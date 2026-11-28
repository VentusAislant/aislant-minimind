# SPDX-License-Identifier: MIT
# Adapted from Microsoft LoRA. Copyright (c) Microsoft Corporation.
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .layers import Embedding, Linear, LoRALayer

__all__ = [
    'mark_only_lora_as_trainable', 'lora_state_dict',
    'add_lora_into_model_by_name', 'load_lora_adapter',
    'merge_lora_weights', 'unmerge_lora_weights',
    'get_all_linear_names', 'get_target_modules'
]


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


# =============================================================================
# LoRA 模块替换工具
# 参考实现：https://github.com/VentusAislant/MSLoRA_CR/blob/main/llava/mslora_utils/lora_utils.py
# =============================================================================


def get_LinearWithLora(
    module: nn.Linear,
    lora_cfg: dict,
    fan_in_fan_out=False
) -> Linear:
    """
    辅助函数, 用于将一个线性层替换为权重相同的带 LoRA 的线性层
    """
    out_module = Linear(
        **lora_cfg,
        in_features=module.in_features,
        out_features=module.out_features,
        fan_in_fan_out=fan_in_fan_out,
        bias=module.bias is not None,
    ).to(device=module.weight.device, dtype=module.weight.dtype)
    out_module.load_state_dict(module.state_dict(), strict=False)
    return out_module


def get_EmbeddingWithLora(module: nn.Embedding, lora_cfg: dict) -> Embedding:
    """
    辅助函数, 用于将一个 Embedding 替换为权重相同的带 LoRA 的 Embedding
    """

    # Embedding LoRA 不使用 dropout；避免通过 **kwargs 继续传给 nn.Embedding。
    embedding_cfg = {
        key: value for key, value in lora_cfg.items()
        if key in {"r", "lora_alpha", "merge_weights"}
    }
    out_module = Embedding(
        num_embeddings=module.num_embeddings,
        embedding_dim=module.embedding_dim,
        **embedding_cfg
    ).to(device=module.weight.device, dtype=module.weight.dtype)
    out_module.load_state_dict(module.state_dict(), strict=False)
    return out_module


def set_module_by_name(
    model: nn.Module,
    name: str,
    target_module: nn.Module
):
    """
    根据模块的完整名称，在模型中找到原来的子模块，将他替换为 target_module
    """
    split_names = name.split('.')

    if len(split_names) == 1:
        if split_names[0].isdigit():
            model[int(split_names[0])] = target_module
        else:
            setattr(model, split_names[0], target_module)
        return

    if split_names[0].isdigit():
        cur_module = model[int(split_names[0])]
    else:
        cur_module = getattr(model, split_names[0])

    for name in split_names[1:-1]:
        if name.isdigit():
            cur_module = cur_module[int(name)]
        else:
            cur_module = getattr(cur_module, name)

    last_name = split_names[-1]
    if last_name.isdigit():
        cur_module[int(last_name)] = target_module
    else:
        setattr(cur_module, last_name, target_module)


def add_lora_into_model_by_name(
        model: nn.Module,
        names: list[str],
        lora_cfg: dict,
        fan_in_fan_out: bool = False
):
    """
    主函数: 负责根据选定的模块名字，进行 lora 的插入
    """
    with tqdm(total=len(names), desc='Adding LoRA modules: ') as pbar:
        for name, module in model.named_modules():
            if name in names:
                if isinstance(module, nn.Linear):
                    set_module_by_name(model, name, get_LinearWithLora(module, lora_cfg, fan_in_fan_out))
                elif isinstance(module, nn.Embedding):
                    set_module_by_name(model, name, get_EmbeddingWithLora(module, lora_cfg))
                pbar.update(1)


@torch.no_grad()
def merge_lora_weights(
        model: nn.Module,
        names: list[str] | None = None,
):
    """合并指定 LoRA 模块；未传 names 时合并模型中的全部 LoRA 模块。"""
    target_names = set(names) if names is not None else None
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer) and (target_names is None or name in target_names):
            module.merge()


@torch.no_grad()
def unmerge_lora_weights(
        model: nn.Module,
        names: list[str] | None = None,
):
    """拆分指定 LoRA 模块；未传 names 时拆分模型中的全部 LoRA 模块。"""
    target_names = set(names) if names is not None else None
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer) and (target_names is None or name in target_names):
            module.unmerge()


def get_all_linear_names(
    model: nn.Module,
    exclude_keywords: tuple[str, ...] = (),
    include_embedding: bool = False,
) -> list[str]:
    names = []

    for name, module in model.named_modules():
        if any(keyword in name for keyword in exclude_keywords):
            continue

        if isinstance(module, nn.Linear):
            names.append(name)
        elif include_embedding and isinstance(module, nn.Embedding):
            names.append(name)

    return names


def get_target_modules(
    model: nn.Module,
    target_modules: list[str] | tuple[str, ...],
    exclude_keywords: tuple[str, ...] = (),
    include_embedding: bool = False,
) -> list[str]:
    """
    根据模块的末级名称筛选 LoRA 目标层，并返回它们在模型中的完整路径。

    例如 target_modules=["q_proj", "v_proj"]，会匹配所有层中的
    self_attn.q_proj 和 self_attn.v_proj。
    """
    target_modules = set(target_modules)
    return [
        name
        for name in get_all_linear_names(
            model,
            exclude_keywords=exclude_keywords,
            include_embedding=include_embedding,
        )
        if name.rsplit(".", 1)[-1] in target_modules
    ]


def load_lora_adapter(
    model: nn.Module,
    adapter_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """向已经加载的基础模型中注入并加载 LoRA adapter。"""
    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    weight_path = adapter_path / "adapter_model.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"没有找到 LoRA 配置文件：{config_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"没有找到 LoRA 权重文件：{weight_path}")

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    target_names = get_target_modules(
        model,
        adapter_config["target_modules"],
    )
    if not target_names:
        raise ValueError(
            f"基础模型中没有找到 LoRA 目标层：{adapter_config['target_modules']}"
        )

    lora_config = {
        "r": adapter_config["rank"],
        "lora_alpha": adapter_config["alpha"],
        "lora_dropout": adapter_config.get("dropout", 0.0),
        "merge_weights": True,
    }
    add_lora_into_model_by_name(
        model,
        names=target_names,
        lora_cfg=lora_config,
    )

    adapter_state = torch.load(
        weight_path,
        map_location=map_location,
        weights_only=True,
    )
    _, unexpected_keys = model.load_state_dict(adapter_state, strict=False)
    if unexpected_keys:
        raise ValueError(f"存在无法加载的 LoRA 参数：{unexpected_keys}")

    # eval() 会触发 LoRA 层的 merge()，推理时不再单独计算低秩分支。
    model.eval()
    return model
