import os
import sys
from pathlib import Path

# 可以支持 python xxx/train_pretrain.py 直接运行
__package__ = "aislant_minimind.elegant_train"
CUR_ROOT = Path(__file__).resolve().parents[2]  # elegant_train -> aislant_minimind -> <project-root>
sys.path.insert(0, str(CUR_ROOT))

import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from tqdm import tqdm
from glob import glob

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from .utils import Logger, SkipBatchSampler, init_distributed_mode, setup_seed, is_main_process
from .arguments import ModelArguments, DataArguments, TrainingArguments


class Trainer(ABC):
    # 训练阶段名称，用于 wandb 实验名，子类覆盖，如 "Pretrain"、"SFT"
    stage = "Train"

    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.local_rank = 0  # 本进程在本机的 GPU 序号（单卡为 0）
        self.device = None
        self.model_config = None  # 模型配置

        # cuda 下的 bfloat16 和 float16 会在 setup_precision 里替换成真正的 autocast
        self.autocast_ctx = nullcontext()  # 混合精度上下文，train_epoch 里 with 使用

        self.scaler = None  # GradScaler，仅 fp16 真正生效
        self.wandb = None  # 实验记录对象，未开启时为 None

        self.model = None  # 训练用模型（compile / DDP 包装之后的版本）
        self.tokenizer = None  # 分词器
        self.optimizer = None  # 优化器
        self.scheduler = None  # 学习率调度器

        self.train_dataset = None  # 训练数据集
        self.train_sampler = None  # 数据集采样器
        self.data_collator = None  # 将变长样本动态补齐并组成 batch

        self.global_step = 0  # 总步数
        self.steps_per_epoch = 0  # 每个 epoch 的步数

    @abstractmethod
    def setup_model(self):
        """
        创建并设置：
            self.tokenizer
            self.model
            self.model_config
        """
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self):
        raise NotImplementedError

    def train(self):
        try:
            self.setup_env()  # 1. 设置环境: 分布式、种子、目录、ckp
            self.setup_precision()  # 2. 设置训练精度: autocast
            self.setup_model()  # 3. 设置 self.tokenizer self.model self.model_config
            self.setup_data()  # 4. 设置数据, train_sampler
            self.setup_optimizer_scheduler()  # 5. 设置训练优化器,学习率调度
            self.setup_logger()  # 6. 设置实验记录: wandb
            start_epoch, start_step = self.setup_resume()  # 7. 从存档恢复状态
            self.wrap_model()  # 8. compile / DDP
            self.fit(start_epoch, start_step)  # 9. epoch 循环

            self.save_model()  # 10. 保存模型
        finally:
            self.cleanup()  # 11.销毁进程组

    def setup_env(self):
        args = self.training_args

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境无法使用 CUDA")

        # ========== 初始化环境和随机种子 ==========
        self.local_rank = init_distributed_mode()  # 处理多卡模式
        if dist.is_initialized():
            self.device = torch.device("cuda", self.local_rank)
            rank = dist.get_rank()
        else:
            rank = 0
            if args.device != "auto":
                self.device = torch.device(args.device)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        setup_seed(args.seed + rank)
        os.makedirs(args.output_dir, exist_ok=True)

    def setup_precision(self):
        # ========== 设置混合精度 ==========
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.training_args.dtype]

        if (
                self.device.type == "cuda"
                and dtype == torch.bfloat16
                and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("当前 GPU 不支持 bfloat16")

        if self.device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
            # autocast (自动混合精度) 可以自动为算子挑选合适的数值精度,
            # 比如 矩阵乘法，线性层等降到16位收益最大，Softmax, LayerNorm 则保持32位
            self.autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

        # fp16 动态范围小，需要 loss scaling；bf16 / fp32 不需要，此时 scaler 的各方法都是 no-op
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.device.type == "cuda" and dtype == torch.float16)
        )

    def setup_data(self):
        self.train_dataset = self.build_dataset()
        if dist.is_initialized():
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                seed=self.training_args.seed,
            )
            num_samples = len(self.train_sampler)  # 每个 rank 分到的样本数
        else:
            self.train_sampler = None
            num_samples = len(self.train_dataset)

        self.steps_per_epoch = math.ceil(num_samples / self.training_args.batch_size)

    def setup_optimizer_scheduler(self):
        args = self.training_args

        self.optimizer = optim.AdamW(
            self.trainable_parameters(),
            lr=args.learning_rate,
        )

        # 学习率是 global_step 的函数：线性 warmup，然后 cosine / linear / constant 衰减到 min_lr
        total = math.ceil(self.steps_per_epoch / args.accumulation_steps) * args.epochs  # 总更新次数
        warmup = int(total * args.warmup_ratio)
        min_ratio = args.min_lr_ratio

        def lr_lambda(update):  # 返回相对 learning_rate 的倍率
            if update < warmup:
                return (update + 1) / warmup
            progress = min((update - warmup) / max(1, total - warmup), 1.0)
            if args.lr_scheduler == "cosine":
                return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            if args.lr_scheduler == "linear":
                return 1 - (1 - min_ratio) * progress
            return 1.0  # constant

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def setup_logger(self):
        # ========== 实验记录: wandb ==========
        args = self.training_args
        if not (args.use_wandb and is_main_process()):
            return  # self.wandb 保持 None

        import wandb  # 如果国内无法访问 wandb 可以使用 import swanlab as wandb
        run_name = (
            f"Aislant-Minimind-{self.stage}"
            f"-Epoch-{args.epochs}"
            f"-BatchSize-{args.batch_size}"
            f"-LearningRate-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project,
            name=run_name
        )
        self.wandb = wandb

    def setup_resume(self):
        """
        checkpoint 断点续训
            返回 (start_epoch, start_step)
            resume_from_checkpoint 为空或找不到可用的 checkpoint 时从头开始
        """
        target = self.training_args.resume_from_checkpoint
        if not target:
            return 0, 0

        if target in {"auto", "latest"}:  # 取最新的完整 checkpoint（trainer_state.pt 最后写入，存在即完整）
            root = os.path.join(self.training_args.output_dir, "checkpoints")
            steps = [
                int(name.split("-")[1])
                for name in (os.listdir(root) if os.path.isdir(root) else [])
                if name.startswith("checkpoint-")
                   and os.path.exists(os.path.join(root, name, "trainer_state.pt"))
            ]
            if not steps:
                return 0, 0
            target = os.path.join(root, f"checkpoint-{max(steps)}")

        state = torch.load(os.path.join(target, "trainer_state.pt"), map_location="cpu")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if state["world_size"] != world_size:
            raise RuntimeError(f"卡数与 checkpoint 不一致: {state['world_size']} vs {world_size}")

        # 模型权重（此时模型还没被 compile / DDP 包装）
        from safetensors.torch import load_file
        weights = {}
        for f in sorted(glob(os.path.join(target, "*.safetensors"))):  # 兼容分片
            weights.update(load_file(f))
        model = self.unwrap_model()
        _, unexpected = model.load_state_dict(weights, strict=False)  # 共享权重会出现在 missing，属正常
        if unexpected:
            raise RuntimeError(f"checkpoint 里有模型不认识的参数: {unexpected[:5]}")
        if hasattr(model, "tie_weights"):
            model.tie_weights()  # save_pretrained 不重复存共享权重，这里补回

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.load_extra_state_dict(state.get("extra", {}))
        self.global_step = state["global_step"]

        Logger(
            f"从 {target} 续训: epoch {state['next_epoch'] + 1}, step {state['next_step']}, global_step {self.global_step}"
        )
        return state["next_epoch"], state["next_step"]

    def wrap_model(self):
        # ========== 编译和分布式包装 ==========
        if self.training_args.use_compile:
            # PyTorch 2.x 的编译功能。它会把模型的前向（以及反向）捕获成计算图，把多个小算子融合成更少的 GPU kernel
            # 减少显存读写和 kernel 启动开销，通常能带来一定比例的加速，具体幅度取决于模型和硬件。
            self.model = torch.compile(self.model)
            Logger('torch.compile enabled')
        if dist.is_initialized():
            # 各卡独立做前向，计算 loss。
            # 反向传播时，DDP 会通过 NCCL 对所有卡的梯度做 all-reduce（求平均），并且和反向计算重叠进行。
            # 所有卡拿到相同的平均梯度，各自执行同样的优化器更新，所以各卡权重始终保持一致
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

    def fit(self, start_epoch, start_step):
        args = self.training_args
        rank = dist.get_rank() if dist.is_initialized() else 0

        # ========== 开始训练 ==========
        for epoch in range(start_epoch, args.epochs):
            setup_seed(args.seed + epoch + rank)

            skip_steps = start_step if epoch == start_epoch else 0
            if skip_steps > 0:
                Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {skip_steps} 个 step，从 step {skip_steps + 1} 开始")

            # 多卡用 train_sampler 打乱并切分数据
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
                sampler = self.train_sampler
            else:
                sampler = RandomSampler(
                    self.train_dataset
                )

            batch_sampler = SkipBatchSampler(sampler, args.batch_size, skip_steps)
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.data_args.num_workers,
                pin_memory=self.device.type == "cuda",
            )

            total_steps = len(train_dataloader) + skip_steps

            self.train_epoch(epoch, train_dataloader, total_steps, skip_steps)

            if args.save_strategy == "epoch" and (epoch + 1) % args.save_interval == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    step=total_steps,
                    total_steps=total_steps,
                )

    def train_epoch(self, epoch, train_dataloader, total_steps, start_step=0):
        args = self.training_args
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            train_dataloader,
            disable=not is_main_process(),
        )
        for step, batch in enumerate(progress, start=start_step + 1):
            with self.autocast_ctx:
                loss, metrics = self.compute_loss(self.to_device(batch))

                # epoch 末尾不足 accumulation_steps 的一组，按实际 batch 数取平均。
                remainder = total_steps % args.accumulation_steps
                is_last_partial_group = (
                        remainder > 0
                        and step > total_steps - remainder
                )
                loss_divisor = remainder if is_last_partial_group else args.accumulation_steps
                backward_loss = loss / loss_divisor

            self.scaler.scale(
                backward_loss
            ).backward()

            should_update = (
                    step % args.accumulation_steps == 0
                    or step == total_steps
            )

            if should_update:
                self.optimizer_step()
                self.global_step += 1
                self.after_update()

                if (
                        args.save_strategy == "steps"
                        and self.global_step
                        % args.save_interval == 0
                ):
                    self.save_checkpoint(
                        epoch=epoch,
                        step=step,
                        total_steps=total_steps,
                    )

            if (
                    step % args.log_interval == 0
                    or step == total_steps
            ):
                self.log_step(
                    epoch=epoch,
                    step=step,
                    total_steps=total_steps,
                    loss=loss,
                    metrics=metrics,
                )

    def compute_loss(self, batch):
        """
        一个 bactch 的前向, 返回 (loss, metrics)
        """
        input_ids, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
        )
        return outputs.loss, {}

    def optimizer_step(self):
        """执行一次通用的参数更新，特殊训练算法可在子类覆写。"""
        args = self.training_args

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters(),
            args.grad_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def log_step(self, epoch, step, total_steps, loss, metrics=None):
        """记录一次训练日志。"""
        values = {
            "loss": loss.detach().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{k: float(v) for k, v in (metrics or {}).items()},
        }

        log_text = " ".join(
            f"{name}: {value:.6f}"
            for name, value in values.items()
        )
        Logger(
            f"Epoch [{epoch + 1}/{self.training_args.epochs}] "
            f"Step [{step}/{total_steps}] {log_text}"
        )

        if self.wandb is not None:
            self.wandb.log(values, step=self.global_step)

    def save_model(self):
        if not is_main_process():
            return

        model = self.unwrap_model()

        model.save_pretrained(
            self.training_args.output_dir,
            safe_serialization=True,
        )

        self.tokenizer.save_pretrained(
            self.training_args.output_dir,
        )

    def save_checkpoint(
            self,
            epoch,
            step,
            total_steps,
    ):
        args = self.training_args

        if step == total_steps:
            next_epoch = epoch + 1
            next_step = 0
        else:
            next_epoch = epoch
            next_step = step

        checkpoint_dir = os.path.join(
            args.output_dir,
            "checkpoints",
            f"checkpoint-{self.global_step}",
        )

        if is_main_process():
            os.makedirs(
                checkpoint_dir,
                exist_ok=True,
            )

            model = self.unwrap_model()

            model.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,
            )

            self.tokenizer.save_pretrained(
                checkpoint_dir,
            )

            trainer_state = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "extra": self.extra_state_dict(),

                "next_epoch": next_epoch,
                "next_step": next_step,
                "global_step": self.global_step,

                "world_size": (
                    dist.get_world_size()
                    if dist.is_initialized()
                    else 1
                ),
            }

            state_path = os.path.join(
                checkpoint_dir,
                "trainer_state.pt",
            )

            temp_path = state_path + ".tmp"

            torch.save(
                trainer_state,
                temp_path,
            )

            os.replace(
                temp_path,
                state_path,
            )

        if dist.is_initialized():
            dist.barrier()

    def cleanup(self):
        # ========== 清理分布进程 ==========
        if self.wandb is not None:
            self.wandb.finish()

        if dist.is_initialized():
            dist.destroy_process_group()

    # ====================
    # = Hooks and Tools
    # ====================
    def after_update(self):
        """每次参数更新之后调用，子类按需重载。"""
        pass

    def extra_state_dict(self) -> dict:
        return {}  # checkpoint 里额外要存的东西，如 PPO 的 critic

    def load_extra_state_dict(self, state: dict):
        pass

    def trainable_parameters(self):
        return self.model.parameters()

    def to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(v) for v in batch)
        return batch

    def unwrap_model(self):
        """剥掉 DDP / compile 外壳，返回原始模型（保存、加载、generate 用）。"""
        m = self.model
        while True:
            if isinstance(m, DistributedDataParallel):
                m = m.module
            elif hasattr(m, "_orig_mod"):
                m = m._orig_mod
            else:
                return m
