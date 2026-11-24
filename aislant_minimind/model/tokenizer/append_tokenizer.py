"""在已有 Hugging Face tokenizer 上追加 Aislant MiniMind token。"""

import argparse
from pathlib import Path

from tokenizers import AddedToken
from transformers import AutoTokenizer

from .exrtra_tokens import ADDITIONAL_TOKENS, BUFFER_TOKENS, SPECIAL_TOKENS


def parse_args():
    parser = argparse.ArgumentParser(description="向已有 tokenizer 追加项目 token")
    parser.add_argument(
        "--base_tokenizer_path",
        type=Path,
        default=Path("model_weights/tokenizers/Llama-2-7b-hf"),
        help="基础 tokenizer 路径或模型名称",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("model_weights/tokenizers/aislant_minimind_llama"),
        help="新 tokenizer 的保存目录",
    )
    parser.add_argument("--special_tokens", nargs="*", default=[], help="额外追加的 special token")
    parser.add_argument("--additional_tokens", nargs="*", default=[], help="额外追加的普通原子 token")
    parser.add_argument(
        "--use_project_core_tokens",
        action="store_true",
        help="将 bos/eos/pad 替换为项目的 im_start/im_end/endoftext",
    )
    return parser.parse_args()


def unique_tokens(tokens):
    """按输入顺序去重，避免同一个 token 被重复添加。"""
    return list(dict.fromkeys(tokens))


def append_tokenizer(
        base_tokenizer_path,
        output_dir,
        special_tokens=None,
        additional_tokens=None,
        use_project_core_tokens=False,
):
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
    original_vocab_size = len(tokenizer)

    # 项目默认 token 始终添加；调用方传入的 token 会追加在默认列表之后。
    project_special_tokens = unique_tokens(
        SPECIAL_TOKENS + BUFFER_TOKENS + list(special_tokens or [])
    )
    project_additional_tokens = unique_tokens(
        ADDITIONAL_TOKENS + list(additional_tokens or [])
    )

    # 保留基础 tokenizer 已有的 additional_special_tokens，并在其后追加项目 token。
    existing_special_tokens = list(tokenizer.additional_special_tokens or [])
    tokenizer.add_special_tokens({
        "additional_special_tokens": unique_tokens(existing_special_tokens + project_special_tokens),
    })

    # think/tool 等标记需要保持原子性，但不能被 skip_special_tokens 自动删除。
    tokenizer.add_tokens([
        AddedToken(token, special=False, normalized=False, lstrip=False, rstrip=False)
        for token in project_additional_tokens
    ])

    if use_project_core_tokens:
        tokenizer.add_special_tokens({
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Base tokenizer: {base_tokenizer_path}")
    print(f"Original vocabulary size: {original_vocab_size}")
    print(f"Added token count: {len(tokenizer) - original_vocab_size}")
    print(f"Final vocabulary size: {len(tokenizer)}")
    print(f"Saved to: {output_dir}")
    return tokenizer


if __name__ == "__main__":
    """
    从项目根目录运行：

    python -m aislant_minimind.model.tokenizer.append_tokenizer
    """
    args = parse_args()
    append_tokenizer(
        base_tokenizer_path=args.base_tokenizer_path,
        output_dir=args.output_dir,
        special_tokens=args.special_tokens,
        additional_tokens=args.additional_tokens,
        use_project_core_tokens=args.use_project_core_tokens,
    )
