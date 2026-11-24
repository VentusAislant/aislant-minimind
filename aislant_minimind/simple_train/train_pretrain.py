"""MiniMind 多卡预训练（DDP，torchrun 启动）。"""

import argparse
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

# 使用 python xxx/train_pretrain.py 直接运行时，补充当前模块所属的包。
if not __package__:
    __package__ = "aislant_minimind.simple_train"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround
import torch
import torch.distributed as dist  # [DDP]
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP  # [DDP]
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # [DDP]
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..dataset.lm_dataset import PretrainDataset
from ..model.modeling_minimind_llama import MiniMindLlamaConfig, MiniMindLlamaForCausalLM
from .utils import Logger, resolve_dtype, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 预训练 (支持多卡)")
    parser.add_argument("--model_path", type=str, default=None, help="已有 Hugging Face 模型目录")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="从头训练时使用的 tokenizer 目录")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simple_pretrain_ddp"))

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--use_moe", action="store_true")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128, help="每张卡的 batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8, help="每个进程的 worker 数")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    return parser.parse_args()


# [DDP] 初始化分布式环境；没用 torchrun 启动时自动退化为单卡
def init_distributed():
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


if __name__ == "__main__":
    """
    从项目根目录运行（4 卡示例）：

    torchrun --nproc_per_node=4 -m aislant_minimind.simple_train.train_pretrain \
        --tokenizer_path model_weights/tokenizers/Llama-2-7b-hf \
        --data_path data/pretrain_t2t_mini.jsonl \
        --output_dir outputs/simple_pretrain
    """
    args = parse_args()

    # [DDP] 0. 分布式初始化
    ddp, rank, local_rank, world_size = init_distributed()
    is_main = rank == 0
    setup_seed(args.seed + rank)  # 每个进程种子不同，dropout 等随机性不重复
    # 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
    # 1. 设备与混合精度
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)  # [DDP] 每个进程绑定自己的卡
    else:
        device = torch.device("cpu")
    args.dtype = resolve_dtype(args.dtype, device)
    dtype = getattr(torch, args.dtype)
    autocast_ctx = nullcontext() if device.type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    # 2. 模型与 tokenizer
    if args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path)
    else:
        if args.tokenizer_path is None:
            raise ValueError("从头训练必须指定 tokenizer_path")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        config = MiniMindLlamaConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=args.use_moe,
        )
        model = MiniMindLlamaForCausalLM(config)
    model.to(device).train()

    # [DDP] 包装模型（构造时会把 rank0 的参数广播给其他卡）
    raw_model = model
    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None
        )

    # 3. 数据、优化器与学习率
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # [DDP] 用 DistributedSampler 切分数据，此时 DataLoader 不能再 shuffle=True
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=args.seed) if ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(loader) / args.accumulation_steps)  # len(loader) 已是单卡的步数
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
            name=f"pretrain-ddp-{world_size}gpu-{args.hidden_size}d-{args.num_hidden_layers}l",
            mode="offline",
        )
        Logger(f"W&B offline 日志目录: {wandb_run.dir}", level="wandb")

    # 4. 训练
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=len(loader) * args.epochs,
        disable=not is_main,
        dynamic_ncols=True,
    )
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # [DDP] 每个 epoch 打乱顺序不同，必须调用
        tail_steps = len(loader) % args.accumulation_steps

        for step, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            accumulation = args.accumulation_steps
            if tail_steps and step > len(loader) - tail_steps:
                accumulation = tail_steps

            is_update = step % args.accumulation_steps == 0 or step == len(loader)

            # [DDP] 梯度累积的中间步不做 all-reduce，只在更新步同步，省通信
            sync_ctx = model.no_sync() if (ddp and not is_update) else nullcontext()
            with sync_ctx:
                with autocast_ctx:
                    outputs = model(input_ids, labels=labels)
                    loss = (outputs.loss + outputs.aux_loss) / accumulation
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
                if ddp:  # [DDP] 所有卡都要参与 all_reduce，所以放在 is_main 判断之外
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

    # 5. 只在主进程保存最终 Hugging Face 模型
    if ddp:
        dist.barrier()
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(args.output_dir, safe_serialization=True)  # [DDP] 存 module 而不是 DDP 包装
        tokenizer.save_pretrained(args.output_dir)
        Logger(f"模型已保存到 {args.output_dir}", level="save")
        if wandb_run:
            wandb_run.finish()

    if ddp:
        dist.destroy_process_group()
