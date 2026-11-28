"""MiniMind 单卡 LoRA SFT。"""

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..dataset.lm_dataset import SFTCollator, SFTDataset
from ..model.lora import add_lora_into_model_by_name, get_target_modules
from ..model.lora import lora_state_dict, mark_only_lora_as_trainable
from ..model.modeling_minimind_llama import MiniMindLlamaForCausalLM
from .utils import Logger, resolve_dtype, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Aislant MiniMind 单卡 LoRA SFT")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型所在的 Hugging Face 目录")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simple_lora_single"))

    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None, help="默认等于 rank，此时 scaling=1")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", choices=["none", "all", "lora_only"], default="none")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="注入 LoRA 的 Linear 层名称",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Aislant-MiniMind-LoRA")
    return parser.parse_args()


if __name__ == "__main__":
    """
    从项目根目录运行：

    python -m aislant_minimind.simple_train.train_lora_single \
        --model_path outputs/simple_full_sft_single \
        --data_path data/lora_medical.jsonl \
        --output_dir outputs/simple_lora_single
    """
    args = parse_args()
    setup_seed(args.seed)

    # 1. 设备与混合精度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dtype = resolve_dtype(args.dtype, device)
    dtype = getattr(torch, args.dtype)
    autocast_ctx = nullcontext() if device.type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    # 2. 加载基础模型、冻结原参数并注入 LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path).to(device)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
    lora_module_names = get_target_modules(model, args.target_modules)
    if not lora_module_names:
        raise ValueError(f"没有找到目标层：{args.target_modules}")

    lora_config = {
        "r": args.lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": args.lora_dropout,
        "merge_weights": True,
    }
    add_lora_into_model_by_name(
        model,
        names=lora_module_names,
        lora_cfg=lora_config,
    )
    mark_only_lora_as_trainable(model, bias=args.lora_bias)
    model.train()

    lora_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in lora_params)
    if not lora_params:
        raise ValueError(f"没有找到目标层：{args.target_modules}")
    Logger(f"LoRA Modules: {len(lora_module_names)}", level="info")
    Logger(
        f"LoRA Params: {trainable_params / 1e6:.3f}M / {total_params / 1e6:.3f}M "
        f"({trainable_params / total_params * 100:.3f}%)",
        level="info",
    )

    # 3. 数据、优化器与学习率
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    collator = SFTCollator(tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=updates_per_epoch * args.epochs,
        eta_min=args.learning_rate * 0.1,
    )

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name="lora-single",
            mode="offline",
        )
        Logger(f"W&B offline 日志目录: {wandb_run.dir}", level="wandb")

    # 4. 训练
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=len(loader) * args.epochs, dynamic_ncols=True)
    for epoch in range(args.epochs):
        tail_steps = len(loader) % args.accumulation_steps

        for step, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            accumulation = args.accumulation_steps
            if tail_steps and step > len(loader) - tail_steps:
                accumulation = tail_steps

            with autocast_ctx:
                outputs = model(input_ids, labels=labels)
                loss = (outputs.loss + outputs.aux_loss) / accumulation

            scaler.scale(loss).backward()

            if step % args.accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            progress.update(1)

            if step % args.log_interval == 0 or step == len(loader):
                log_loss = loss.item() * accumulation
                with tqdm.external_write_mode():
                    Logger(
                        f"Epoch [{epoch + 1}/{args.epochs}] Step [{step}/{len(loader)}] "
                        f"loss={log_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8f}",
                        level="train",
                    )
                if wandb_run:
                    wandb_run.log({
                        "loss": log_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    })
    progress.close()

    # 5. 只保存 LoRA adapter、配置和 tokenizer
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {
        key: value.detach().cpu()
        for key, value in lora_state_dict(model, bias=args.lora_bias).items()
    }
    torch.save(adapter_state, args.output_dir / "adapter_model.pt")
    model.config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    adapter_config = {
        "base_model_path": args.model_path,
        "rank": args.lora_rank,
        "alpha": lora_alpha,
        "dropout": args.lora_dropout,
        "bias": args.lora_bias,
        "target_modules": args.target_modules,
    }
    (args.output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Logger(f"LoRA adapter 已保存到 {args.output_dir}", level="save")
    if wandb_run:
        wandb_run.finish()
