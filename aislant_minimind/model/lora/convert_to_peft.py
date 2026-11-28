"""将 Aislant MiniMind LoRA adapter 转换为 Hugging Face PEFT 格式。"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def _find_peft_key(
    custom_key: str,
    peft_state: dict[str, torch.Tensor],
) -> str:
    """根据模块路径找到自定义参数在 PEFT 模型中的完整参数名。"""
    if custom_key.endswith(".lora_A") or custom_key.endswith(".lora_B"):
        module_name, lora_name = custom_key.rsplit(".", 1)
        marker = f".{lora_name}."
        candidates = [
            key for key in peft_state
            if marker in key and key.split(marker, 1)[0].endswith(module_name)
        ]
    else:
        # bias="all" 或 bias="lora_only" 时还需要转换相应的 bias。
        candidates = [key for key in peft_state if key.endswith(custom_key)]

    if len(candidates) != 1:
        raise ValueError(
            f"参数 {custom_key!r} 应当匹配一个 PEFT 参数，实际匹配到：{candidates}"
        )
    return candidates[0]


def convert_to_peft(
    adapter_path: str | Path,
    output_dir: str | Path,
    base_model_path: str | Path | None = None,
) -> Path:
    """把项目自定义 LoRA adapter 转换并保存成标准 PEFT adapter。"""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise ImportError(
            "转换 LoRA 需要安装 PEFT：pip install peft==0.14.0"
        ) from error

    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)
    config_path = adapter_path / "adapter_config.json"
    weight_path = adapter_path / "adapter_model.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"没有找到 LoRA 配置文件：{config_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"没有找到 LoRA 权重文件：{weight_path}")

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model_path = base_model_path or adapter_config.get("base_model_path")
    if base_model_path is None:
        raise ValueError("没有指定基础模型路径，且 adapter_config.json 中不存在 base_model_path")

    # 由 PEFT 创建目标结构，避免手工假设不同 PEFT 版本的参数名前缀。
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=adapter_config["rank"],
        lora_alpha=adapter_config["alpha"],
        lora_dropout=adapter_config.get("dropout", 0.0),
        bias=adapter_config.get("bias", "none"),
        target_modules=adapter_config["target_modules"],
    )
    peft_model = get_peft_model(base_model, peft_config)
    peft_state = peft_model.state_dict()

    custom_state = torch.load(weight_path, map_location="cpu", weights_only=True)
    converted_state = {}
    for custom_key, value in custom_state.items():
        peft_key = _find_peft_key(custom_key, peft_state)
        if peft_state[peft_key].shape != value.shape:
            raise ValueError(
                f"参数形状不一致：{custom_key} {tuple(value.shape)} -> "
                f"{peft_key} {tuple(peft_state[peft_key].shape)}"
            )
        converted_state[peft_key] = value

    _, unexpected_keys = peft_model.load_state_dict(converted_state, strict=False)
    if unexpected_keys:
        raise ValueError(f"存在 PEFT 模型无法加载的参数：{unexpected_keys}")

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir, safe_serialization=True)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="将 Aislant MiniMind LoRA 转换为 PEFT 格式")
    parser.add_argument("--adapter_path", type=Path, required=True, help="项目自定义 LoRA adapter 目录")
    parser.add_argument("--output_dir", type=Path, required=True, help="PEFT adapter 输出目录")
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=None,
        help="基础模型目录；默认读取 adapter_config.json 中的 base_model_path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = convert_to_peft(
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
    )
    print(f"PEFT adapter 已保存到：{output_dir}")
