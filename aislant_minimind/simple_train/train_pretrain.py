from pathlib import Path

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import AutoTokenizer

from ..model.modeling_minimind_llama import MiniMindLlamaConfig, MiniMindLlamaForCausalLM
from ..dataset.lm_dataset import PretrainDataset
from .utils import Logger, is_main_process, init_distributed_mode, setup_seed, \
    resolve_dtype

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 1. 基本参数
    # 预训练需要特别指定 tokenizer 路径
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径，None 表示从0训练，传入路径可以基于已有权重继续训练")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="从头训练时使用的 tokenizer 路径")
    parser.add_argument("--data_path", type=str, required=True, help="预训练数据路径，相对路径从项目根目录解析")
    parser.add_argument("--output_dir", type=str, default="outputs/simple_pretrain",
                        help="模型输出目录，相对路径从项目根目录解析")

    # 2. 模型相关
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=4, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 3. 训练相关
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="线性warmup占总步数比例，0为关闭")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    return args


if __name__ == "__main__":
    """
    从项目根目录运行。

    单卡训练：
    python -m aislant_minimind.simple_train.train_pretrain \
        --tokenizer_path model_weights/tokenizers/llama \
        --data_path data/pretrain_t2t_mini.jsonl \
        --output_dir outputs/simple_pretrain

    多卡训练（例如使用 2 张可见 GPU）：
    torchrun --standalone --nproc_per_node=2 \
        -m aislant_minimind.simple_train.train_pretrain \
        --tokenizer_path model_weights/tokenizers/llama \
        --data_path data/pretrain_t2t_mini.jsonl \
        --output_dir outputs/simple_pretrain
    """
    args = parse_args()

    # ========== 1. 初始化环境和随机种子， 配置目录，设置混合精度==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():  # 多卡环境，获取当前卡的设备号
        args.device = f"cuda:{local_rank}"
        setup_seed(args.seed + dist.get_rank())
    else:
        setup_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_type = "cuda" if torch.device(args.device).type == "cuda" else "cpu"
    args.dtype = resolve_dtype(args.dtype, args.device)
    dtype = getattr(torch, args.dtype)
    if device_type == 'cpu':
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

    scaler = torch.amp.GradScaler(device="cuda", enabled=(dtype == torch.float16))

    # ========== 2. 定义模型、数据、优化器 ==========
    if args.model_path is not None:
        # 加载已有 Hugging Face 模型，开始一次新的训练。
        model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    else:
        # 只加载 tokenizer，模型参数随机初始化。
        if args.tokenizer_path is None:
            raise ValueError("从头训练必须指定 tokenizer_path")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        model_config = MiniMindLlamaConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        )
        model = MiniMindLlamaForCausalLM(model_config)

    model = model.to(args.device)

    train_dataset = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
    )
    train_sampler = (
        DistributedSampler(train_dataset)
        if dist.is_initialized()
        else RandomSampler(train_dataset)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # scheduler 每次 optimizer 更新后前进一步，而不是每个 batch 前进一步。
    steps_per_epoch = math.ceil(len(train_sampler) / args.batch_size)
    updates_per_epoch = math.ceil(steps_per_epoch / args.accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = int(total_updates * args.warmup_ratio)
    cosine_updates = max(1, total_updates - warmup_updates)

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_updates,
        eta_min=args.learning_rate * 0.1,
    )
    if warmup_updates > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_updates,
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_updates],
        )
    else:
        lr_scheduler = cosine_scheduler

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 5. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 6. 开始训练 ==========
    model.train()
    for epoch in range(args.epochs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        setup_seed(args.seed + epoch + rank)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        start_time = time.time()
        iters = len(loader)
        tail_steps = iters % args.accumulation_steps
        for step, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss = res.loss + res.aux_loss
                current_accumulation_steps = args.accumulation_steps
                if tail_steps > 0 and step > iters - tail_steps:
                    current_accumulation_steps = tail_steps
                loss = loss / current_accumulation_steps

            scaler.scale(loss).backward()

            optimizer_updated = step % args.accumulation_steps == 0 or step == iters
            if optimizer_updated:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0 or step == iters:
                spend_time = time.time() - start_time
                current_loss = loss.item() * current_accumulation_steps
                current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
                current_logits_loss = current_loss - current_aux_loss
                current_lr = optimizer.param_groups[-1]['lr']
                eta_min = spend_time / step * (iters - step) // 60
                Logger(
                    f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, '
                    f'logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                    f'lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min'
                )
                if wandb:
                    wandb.log({
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    })

            del input_ids, labels, res, loss

    # 最终模型始终保存到 output_dir，供推理或下一阶段训练直接加载。
    if is_main_process():
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        raw_model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        Logger(f"最终模型已保存到 {args.output_dir}")

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
