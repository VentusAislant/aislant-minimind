"""MiniMind 单卡 DPO。"""

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import datasets  # noqa: F401
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..dataset.lm_dataset import DPOCollator, DPODataset
from ..model.modeling_minimind_llama import MiniMindLlamaForCausalLM
from .utils import Logger, resolve_dtype, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Aislant MiniMind 单卡直接偏好优化 DPO")
    parser.add_argument("--model_path", type=str, required=True, help="SFT 模型所在的 Hugging Face 目录")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/simple_dpo_single"))
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--beta", type=float, default=0.15, help="DPO 偏离参考模型的惩罚强度")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Aislant-MiniMind-DPO")
    return parser.parse_args()


def sequence_log_probs(logits, labels, mask):
    """计算每条序列中 assistant 回复部分的 log-prob 总和。"""
    # 通过 log softmax 得到每个词的 log 概率, 形状 [2*bsz, seq_len, vocab_size]
    token_log_probs = F.log_softmax(logits.float(), dim=-1)
    # 取出 label token 对应位置的 log 概率
    """
    例如: bsz=1(chosen 与 rejected 拼接后实际 batch 为 2), seq_len=3, vocab_size=4
        token_log_probs = torch.tensor([
            # ---- 样本 0:chosen ----
            [[-2.3, -0.5, -1.8, -3.0],   # 位置 0
             [-1.2, -2.0, -0.4, -2.5],   # 位置 1
             [-3.1, -1.0, -2.2, -0.3]],  # 位置 2
            # ---- 样本 1:rejected ----
            [[-1.5, -2.2, -0.6, -2.8],   # 位置 0
             [-0.9, -1.7, -2.4, -1.1],   # 位置 1
             [-2.6, -0.8, -3.2, -1.4]],  # 位置 2
        ])  # 形状 [2, 3, 4]

        labels = torch.tensor([
            [1, 2, 3],   # chosen 的目标 token
            [2, 0, 1],   # rejected 的目标 token(回复内容不同,所以 id 不同)
        ])  # 形状 [2, 3]

        labels.unsqueeze(-1) = tensor([
            [[1], [2], [3]],
            [[2], [0], [1]]
        ])  # 形状 [2, 3, 1]

        gather(token_log_probs, 2, labels.unsqueeze(-1)):
            tensor([[[-0.5], [-0.4], [-0.3]],     # chosen
                    [[-0.6], [-0.9], [-0.8]]])    # rejected
            # 形状 [2, 3, 1]

        gather.squeeze(-1):
            tensor([[-0.5, -0.4, -0.3],     # chosen
                    [-0.6, -0.9, -0.8]])    # rejected
            # 形状 [2, 3],每个位置是"正确 token"的 log 概率

        * mask:
            mask = tensor([
                [0, 1, 1],   # chosen:位置 0 是 prompt,位置 1、2 是回复
                [0, 1, 0],   # rejected:位置 0 是 prompt,位置 1 是回复,位置 2 是 padding
            ])  # 形状 [2, 3]

            tensor([[ 0.0, -0.4, -0.3],
                    [ 0.0, -0.9,  0.0]])
            # prompt 和 padding 位置被置 0,不参与累加;形状仍为 [2, 3]

        sum(dim=-1):
            tensor([-0.7, -0.9])
            # chosen = -0.4 + -0.3 = -0.7
            # rejected = -0.9
            # 形状 [2],即 [2*bsz],每条序列的回复部分 log 概率总和
            # 之后在 compute_dpo_loss 里用 chunk(2) 拆成 chosen / rejected
    """
    token_log_probs = torch.gather(token_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def compute_dpo_loss(policy_log_probs, reference_log_probs, beta):
    """batch 前半为 chosen、后半为 rejected。"""
    chosen_policy, rejected_policy = policy_log_probs.chunk(2)
    chosen_reference, rejected_reference = reference_log_probs.chunk(2)
    # log 空间的差就是概率比
    policy_ratio = chosen_policy - rejected_policy
    reference_ratio = chosen_reference - rejected_reference
    """
    loss = - \log{sigmoid{
        beta * (
            \log{\frac{\pi_{\theta}(y_w)}{\pi_{ref}(y_w)}}
            -
            \log{\frac{\pi_{\theta}(y_l)}{\pi_{ref}(y_l)}}
        )
    }}
    y_w 表示接受
    y_l 表示拒绝
    """
    return -F.logsigmoid(beta * (policy_ratio - reference_ratio)).mean()


if __name__ == "__main__":
    """
    从项目根目录运行：

    python -m aislant_minimind.simple_train.train_dpo_single \
        --model_path outputs/simple_full_sft_single \
        --data_path data/dpo.jsonl \
        --output_dir outputs/simple_dpo_single
    """
    args = parse_args()
    setup_seed(args.seed)

    # 1. 设备与混合精度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dtype = resolve_dtype(args.dtype, device)
    dtype = getattr(torch, args.dtype)
    autocast_ctx = nullcontext() if device.type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    # 2. 从同一个 SFT 目录创建策略模型和冻结的参考模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path).to(device).train()
    reference_model = MiniMindLlamaForCausalLM.from_pretrained(args.model_path).to(device).eval()
    reference_model.requires_grad_(False)

    # 3. 数据、优化器与学习率
    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    collator = DPOCollator(tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name="dpo-single",
            mode="offline",
        )
        Logger(f"W&B offline 日志目录: {wandb_run.dir}", level="wandb")

    # 4. 训练
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=len(loader) * args.epochs, dynamic_ncols=True)
    for epoch in range(args.epochs):
        tail_steps = len(loader) % args.accumulation_steps

        for step, batch in enumerate(loader, start=1):
            chosen_x = batch["x_chosen"].to(device)
            rejected_x = batch["x_rejected"].to(device)
            chosen_y = batch["y_chosen"].to(device)
            rejected_y = batch["y_rejected"].to(device)
            chosen_mask = batch["mask_chosen"].to(device)
            rejected_mask = batch["mask_rejected"].to(device)
            # 将 chosen 和 rejected 拼接到一个 batch 中, 因此这里实际 forward 的 batch size是 2 * batch_size
            input_ids = torch.cat((chosen_x, rejected_x), dim=0)
            labels = torch.cat((chosen_y, rejected_y), dim=0)
            loss_mask = torch.cat((chosen_mask, rejected_mask), dim=0)

            accumulation = args.accumulation_steps
            if tail_steps and step > len(loader) - tail_steps:
                accumulation = tail_steps

            with autocast_ctx:
                with torch.no_grad():
                    # 冻结的参考模型
                    reference_outputs = reference_model(input_ids)
                # 正在训练的模型
                policy_outputs = model(input_ids)

                reference_log_probs = sequence_log_probs(reference_outputs.logits, labels, loss_mask)
                policy_log_probs = sequence_log_probs(policy_outputs.logits, labels, loss_mask)
                dpo_loss = compute_dpo_loss(policy_log_probs, reference_log_probs, args.beta)
                loss = (dpo_loss + policy_outputs.aux_loss) / accumulation

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
                        f"loss={log_loss:.4f} dpo_loss={dpo_loss.item():.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8f}",
                        level="train",
                    )
                if wandb_run:
                    wandb_run.log({
                        "loss": log_loss,
                        "dpo_loss": dpo_loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    })
    progress.close()

    # 5. 只保存训练后的策略模型
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    Logger(f"模型已保存到 {args.output_dir}", level="save")
    if wandb_run:
        wandb_run.finish()
