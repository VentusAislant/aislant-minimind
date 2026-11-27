"""MiniMind 多卡 LoRA SFT（DDP，torchrun 启动）。"""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

# 使用 python xxx/train_lora.py 直接运行时，补充当前模块所属的包。
__package__ = "aislant_minimind.simple_train"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround
import torch
import torch.distributed as dist
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..dataset.lm_dataset import SFTCollator, SFTDataset
from ..model.lora import Linear as LoRALinear
from ..model.lora import lora_state_dict, mark_only_lora_as_trainable
from ..model.modeling_minimind_llama import MiniMindLlamaForCausalLM
from .utils import Logger, resolve_dtype, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Aislant MiniMind LoRA SFT（支持多卡）")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型所在的 Hugging Face 目录")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simple_lora"))

    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16, help="每张卡的 batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8, help="每个进程的 worker 数")
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


def init_distributed():
    """初始化分布式环境；没用 torchrun 启动时自动退化为单卡。"""
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def inject_lora(model, target_modules, rank, alpha, dropout):
    """将指定的 nn.Linear 原地替换成 LoRA Linear，并保留预训练权重。"""
    target_modules = set(target_modules)
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name not in target_modules or not isinstance(child, nn.Linear):
                continue
            if isinstance(child, LoRALinear):
                continue

            lora_linear = LoRALinear(
                child.in_features,
                child.out_features,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=child.bias is not None,
                merge_weights=True,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            with torch.no_grad():
                lora_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    lora_linear.bias.copy_(child.bias)
            setattr(parent, child_name, lora_linear)
            replaced.append(f"{parent_name}.{child_name}".lstrip("."))
    return replaced


if __name__ == "__main__":
    """
    从项目根目录运行（4 卡示例）：

    torchrun --nproc_per_node=4 -m aislant_minimind.simple_train.train_lora \
        --model_path outputs/simple_full_sft \
        --data_path data/lora_medical.jsonl \
        --output_dir outputs/simple_lora
    """
    args = parse_args()

    # 0. 分布式初始化
    ddp, rank, local_rank, world_size = init_distributed()
    is_main = rank == 0
    setup_seed(args.seed + rank)

    # 1. 设备与混合精度
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    args.dtype = resolve_dtype(args.dtype, device)
    dtype = getattr(torch, args.dtype)
    autocast_ctx = nullcontext() if device.type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    # 2. 加载基础模型、冻结原参数并注入 LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path).to(device)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
    replaced_modules = inject_lora(
        model,
        target_modules=args.target_modules,
        rank=args.lora_rank,
        alpha=lora_alpha,
        dropout=args.lora_dropout,
    )
    mark_only_lora_as_trainable(model, bias=args.lora_bias)
    model.train()

    lora_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in lora_params)
    if not replaced_modules or not lora_params:
        raise ValueError(f"没有找到目标层：{args.target_modules}")
    if is_main:
        Logger(f"LoRA Modules: {len(replaced_modules)}", level="info")
        Logger(
            f"LoRA Params: {trainable_params / 1e6:.3f}M / {total_params / 1e6:.3f}M "
            f"({trainable_params / total_params * 100:.3f}%)",
            level="info",
        )

    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # 3. 数据、优化器与学习率
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    collator = SFTCollator(tokenizer.pad_token_id)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    ) if ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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
    if args.use_wandb and is_main:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"lora-ddp-{world_size}gpu",
            mode="offline",
        )
        Logger(f"W&B offline 日志目录: {wandb_run.dir}", level="wandb")

    # 4. 训练
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=len(loader) * args.epochs, disable=not is_main, dynamic_ncols=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        tail_steps = len(loader) % args.accumulation_steps

        for step, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            accumulation = args.accumulation_steps
            if tail_steps and step > len(loader) - tail_steps:
                accumulation = tail_steps
            is_update = step % args.accumulation_steps == 0 or step == len(loader)

            sync_ctx = model.no_sync() if (ddp and not is_update) else nullcontext()
            with sync_ctx:
                with autocast_ctx:
                    outputs = model(input_ids, labels=labels)
                    loss = (outputs.loss + outputs.aux_loss) / accumulation
                scaler.scale(loss).backward()

            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if is_main:
                progress.update(1)

            if step % args.log_interval == 0 or step == len(loader):
                log_loss = loss.detach() * accumulation
                if ddp:
                    dist.all_reduce(log_loss, op=dist.ReduceOp.AVG)
                if is_main:
                    with tqdm.external_write_mode():
                        Logger(
                            f"Epoch [{epoch + 1}/{args.epochs}] Step [{step}/{len(loader)}] "
                            f"loss={log_loss.item():.4f} "
                            f"lr={optimizer.param_groups[0]['lr']:.8f}",
                            level="train",
                        )
                    if wandb_run:
                        wandb_run.log({
                            "loss": log_loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1,
                        })
    progress.close()

    # 5. 只保存 LoRA adapter、配置和 tokenizer
    if ddp:
        dist.barrier()
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        adapter_state = {
            key: value.detach().cpu()
            for key, value in lora_state_dict(raw_model, bias=args.lora_bias).items()
        }
        torch.save(adapter_state, args.output_dir / "adapter_model.pt")
        raw_model.config.save_pretrained(args.output_dir)
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

    if ddp:
        dist.destroy_process_group()
