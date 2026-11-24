"""MiniMind 多卡 DPO（DDP，torchrun 启动）。"""

import argparse
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

__package__ = "aislant_minimind.simple_train"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import datasets  # noqa: F401
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..dataset.lm_dataset import DPOCollator, DPODataset
from ..model.modeling_minimind_llama import MiniMindLlamaForCausalLM
from .utils import Logger, resolve_dtype, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Aislant MiniMind 直接偏好优化 DPO（支持多卡）")
    parser.add_argument("--model_path", type=str, required=True, help="SFT 模型所在的 Hugging Face 目录")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simple_dpo"))
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="每张卡的 batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--beta", type=float, default=0.15, help="DPO 偏离参考模型的惩罚强度")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8, help="每个进程的 worker 数")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Aislant-MiniMind-DPO")
    return parser.parse_args()


def init_distributed():
    """初始化分布式环境；非 torchrun 启动时自动退化为单卡。"""
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


def sequence_log_probs(logits, labels, mask):
    """计算每条序列中 assistant 回复部分的 log-prob 总和。"""
    token_log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = torch.gather(token_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def compute_dpo_loss(policy_log_probs, reference_log_probs, beta):
    """batch 前半为 chosen、后半为 rejected。"""
    chosen_policy, rejected_policy = policy_log_probs.chunk(2)
    chosen_reference, rejected_reference = reference_log_probs.chunk(2)
    policy_ratio = chosen_policy - rejected_policy
    reference_ratio = chosen_reference - rejected_reference
    return -F.logsigmoid(beta * (policy_ratio - reference_ratio)).mean()


if __name__ == "__main__":
    """
    从项目根目录运行（4 卡示例）：

    torchrun --nproc_per_node=4 -m aislant_minimind.simple_train.train_dpo \
        --model_path outputs/simple_full_sft \
        --data_path data/dpo.jsonl \
        --output_dir outputs/simple_dpo
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

    # 2. 从同一个 SFT 目录创建策略模型和冻结的参考模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path).to(device).train()
    reference_model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path).to(device).eval()
    reference_model.requires_grad_(False)
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # 3. 数据、优化器与学习率
    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    collator = DPOCollator(tokenizer.pad_token_id)
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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
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
            name=f"dpo-ddp-{world_size}gpu",
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

        for step, batch in enumerate(loader, start=1):
            chosen_x = batch["x_chosen"].to(device)
            rejected_x = batch["x_rejected"].to(device)
            chosen_y = batch["y_chosen"].to(device)
            rejected_y = batch["y_rejected"].to(device)
            chosen_mask = batch["mask_chosen"].to(device)
            rejected_mask = batch["mask_rejected"].to(device)
            input_ids = torch.cat((chosen_x, rejected_x), dim=0)
            labels = torch.cat((chosen_y, rejected_y), dim=0)
            loss_mask = torch.cat((chosen_mask, rejected_mask), dim=0)

            accumulation = args.accumulation_steps
            if tail_steps and step > len(loader) - tail_steps:
                accumulation = tail_steps
            is_update = step % args.accumulation_steps == 0 or step == len(loader)

            sync_ctx = model.no_sync() if (ddp and not is_update) else nullcontext()
            with sync_ctx:
                with autocast_ctx:
                    with torch.no_grad():
                        reference_outputs = reference_model(input_ids)
                    policy_outputs = model(input_ids)
                    reference_log_probs = sequence_log_probs(reference_outputs.logits, labels, loss_mask)
                    policy_log_probs = sequence_log_probs(policy_outputs.logits, labels, loss_mask)
                    dpo_loss = compute_dpo_loss(policy_log_probs, reference_log_probs, args.beta)
                    loss = (dpo_loss + policy_outputs.aux_loss) / accumulation
                scaler.scale(loss).backward()

            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if is_main:
                progress.update(1)

            if step % args.log_interval == 0 or step == len(loader):
                log_loss = loss.detach() * accumulation
                log_dpo_loss = dpo_loss.detach()
                if ddp:
                    dist.all_reduce(log_loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(log_dpo_loss, op=dist.ReduceOp.AVG)
                if is_main:
                    with tqdm.external_write_mode():
                        Logger(
                            f"Epoch [{epoch + 1}/{args.epochs}] Step [{step}/{len(loader)}] "
                            f"loss={log_loss.item():.4f} dpo_loss={log_dpo_loss.item():.4f} "
                            f"lr={optimizer.param_groups[0]['lr']:.8f}",
                            level="train",
                        )
                    if wandb_run:
                        wandb_run.log({
                            "loss": log_loss.item(),
                            "dpo_loss": log_dpo_loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1,
                        })
    progress.close()

    # 5. 只保存训练后的策略模型
    if ddp:
        dist.barrier()
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        Logger(f"模型已保存到 {args.output_dir}", level="save")
        if wandb_run:
            wandb_run.finish()

    if ddp:
        dist.destroy_process_group()
