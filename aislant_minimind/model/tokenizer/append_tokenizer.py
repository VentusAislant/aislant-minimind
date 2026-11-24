"""在已有 tokenizer 上追加 Aislant MiniMind token。"""

import argparse
import json
from pathlib import Path

from tokenizers import AddedToken
from transformers import AutoTokenizer

from .chat_template import CHAT_TEMPLATE
from .exrtra_tokens import (
    ADDITIONAL_TOKENS,
    BUFFER_TOKENS,
    CORE_SPECIAL_TOKENS,
    OTHER_SPECIAL_TOKENS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="向已有 tokenizer 追加一些常用特殊 token")
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
        "--model_max_length",
        type=int,
        default=32768,
        help=f"tokenizer 最大序列长度，本项目模型计算 RoPE 的最大位置范围为 32768",
    )
    return parser.parse_args()


def unique_tokens(tokens):
    """按输入顺序去重，避免同一个 token 被重复添加。"""
    return list(dict.fromkeys(tokens))


def replace_json_token(value, replacements):
    """递归替换 tokenizer JSON 中的 token 名称，保留原 token ID。"""
    if isinstance(value, dict):
        return {
            replacements.get(key, key): replace_json_token(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_json_token(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def rename_core_tokens(tokenizer, output_dir):
    """将已有核心 token 改为本项目的名称，同时保持其 ID 不变。"""
    replacements = {}
    token_ids = {}
    for role, new_token in CORE_SPECIAL_TOKENS.items():
        old_token = getattr(tokenizer, role)
        old_token_id = getattr(tokenizer, f"{role}_id")
        if old_token is not None:
            replacements[old_token] = new_token
            token_ids[role] = old_token_id

    tokenizer.save_pretrained(output_dir)

    # tokenizer.json 供 fast tokenizer 使用。
    tokenizer_json_path = output_dir / "tokenizer.json"
    if tokenizer_json_path.is_file():
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        tokenizer_data = replace_json_token(tokenizer_data, replacements)
        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    # tokenizer.model 供 Llama slow tokenizer 使用。
    tokenizer_model_path = output_dir / "tokenizer.model"
    if tokenizer_model_path.is_file():
        from sentencepiece import sentencepiece_model_pb2

        model = sentencepiece_model_pb2.ModelProto()
        model.ParseFromString(tokenizer_model_path.read_bytes())
        for role, token_id in token_ids.items():
            model.pieces[token_id].piece = CORE_SPECIAL_TOKENS[role]
        if "bos_token" in token_ids:
            model.trainer_spec.bos_piece = CORE_SPECIAL_TOKENS["bos_token"]
        if "eos_token" in token_ids:
            model.trainer_spec.eos_piece = CORE_SPECIAL_TOKENS["eos_token"]
        if "pad_token" in token_ids:
            model.trainer_spec.pad_piece = CORE_SPECIAL_TOKENS["pad_token"]
        tokenizer_model_path.write_bytes(model.SerializeToString())

    # 两份 Hugging Face 元数据也需要同步名称。
    for filename in ("tokenizer_config.json", "special_tokens_map.json"):
        path = output_dir / filename
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = replace_json_token(json.load(f), replacements)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    renamed_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    for role, expected_id in token_ids.items():
        actual_id = getattr(renamed_tokenizer, f"{role}_id")
        if actual_id != expected_id:
            raise RuntimeError(f"{role} ID 发生变化: {expected_id} -> {actual_id}")
    return renamed_tokenizer


def append_tokenizer(
        base_tokenizer_path,
        output_dir,
        model_max_length,
        special_tokens=None,
        additional_tokens=None,
):
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
    original_vocab_size = len(tokenizer)
    original_core_ids = {
        "bos_token": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token_id,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = rename_core_tokens(tokenizer, output_dir)

    # BOS/EOS 已在原 ID 上改名，不再作为 additional special token 重复添加。
    project_special_tokens = unique_tokens(
        OTHER_SPECIAL_TOKENS + list(special_tokens or [])
    )
    project_additional_tokens = unique_tokens(
        ADDITIONAL_TOKENS + BUFFER_TOKENS + list(additional_tokens or [])
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

    # Llama 1/2 没有 PAD，只有这种缺失的核心角色才需要新增 ID。
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": CORE_SPECIAL_TOKENS["pad_token"]})

    tokenizer.model_max_length = model_max_length
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(output_dir)

    print(f"Base tokenizer: {base_tokenizer_path}")
    print(f"Original vocabulary size: {original_vocab_size}")
    print(f"Added token count: {len(tokenizer) - original_vocab_size}")
    print(f"Final vocabulary size: {len(tokenizer)}")
    print(
        f"BOS: {tokenizer.bos_token} (id={tokenizer.bos_token_id}, "
        f"original id={original_core_ids['bos_token']})"
    )
    print(
        f"EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id}, "
        f"original id={original_core_ids['eos_token']})"
    )
    print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
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
        model_max_length=args.model_max_length,
        special_tokens=args.special_tokens,
        additional_tokens=args.additional_tokens,
    )
