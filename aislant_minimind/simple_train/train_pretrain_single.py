"""MiniMind 单卡预训练。"""

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..dataset.lm_dataset import PretrainDataset
from ..model.modeling_minimind_llama import MiniMindLlamaConfig, MiniMindLlamaForCausalLM
from .utils import Logger, resolve_dtype, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 单卡预训练")
    parser.add_argument("--model_path", type=str, default=None, help="已有 Hugging Face 模型目录")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="从头训练时使用的 tokenizer 目录")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simple_pretrain_single"))

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--use_moe", action="store_true")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    return parser.parse_args()


if __name__ == "__main__":
    """
    从项目根目录运行：

    python -m aislant_minimind.simple_train.train_pretrain_single \
        --tokenizer_path model_weights/tokenizers/Llama-2-7b-hf \
        --data_path data/pretrain_t2t_mini.jsonl \
        --output_dir outputs/simple_pretrain_single
    """
    args = parse_args()
    setup_seed(args.seed)

    # 1. 设备与混合精度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 3. 数据、优化器与学习率
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"pretrain-single-{args.hidden_size}d-{args.num_hidden_layers}l",
            mode="offline",
        )
        Logger(f"W&B offline 日志目录: {wandb_run.dir}", level="wandb")

    # 4. 训练
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=len(loader) * args.epochs,
        dynamic_ncols=True,
    )
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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

    # 5. 保存最终 Hugging Face 模型
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    Logger(f"模型已保存到 {args.output_dir}", level="save")
    if wandb_run:
        wandb_run.finish()
