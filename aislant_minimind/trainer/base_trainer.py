import os
import sys
from pathlib import Path

# 可以支持 python xxx/train_pretrain.py 直接运行
__package__ = "aislant_minimind.trainer"
CUR_ROOT = Path(__file__).resolve().parents[2]  # trainer -> aislant_minimind -> <project-root>
sys.path.insert(0, str(CUR_ROOT))

import math
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from tqdm import tqdm

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

        self.model_config = None  # 模型结构配置
        self.ckp_data = None  # 续训存档内容，不续训则为 None

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

        self.global_step = 0

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
            self.setup_optimizer()  # 5. 设置训练优化器
            self.setup_logger()  # 6. 设置实验记录: wandb
            start_epoch, start_step = self.setup_resume()  # 7. 从存档恢复状态
            self.wrap_model()  # 8. compile / DDP
            self.fit(start_epoch, start_step)  # 9. epoch 循环

            self.save_model()
        finally:
            self.cleanup()  # 10. 销毁进程组

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

    def setup_data(self):
        self.train_dataset = self.build_dataset()
        if dist.is_initialized():
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                seed=self.training_args.seed,
            )
        else:
            self.train_sampler = None

    def setup_optimizer(self):
        args = self.training_args

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                    self.device.type == "cuda"
                    and args.dtype == "float16"
            ),
        )

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
        """checkpoint 恢复逻辑后续实现，当前从头开始训练。"""
        return 0, 0

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

        # ========== 开始训练 ==========
        for epoch in range(start_epoch, args.epochs):
            rank = dist.get_rank() if dist.is_initialized() else 0
            setup_seed(args.seed + epoch + rank)

            # 多卡用 train_sampler 打乱并切分数据
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
                sampler = self.train_sampler
            else:
                sampler = RandomSampler(
                    self.train_dataset
                )

            skip_steps = start_step if epoch == start_epoch else 0

            batch_sampler = SkipBatchSampler(sampler, args.batch_size, skip_steps)
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.data_args.num_workers,
                pin_memory=self.device.type == "cuda",
            )

            total_steps = len(train_dataloader) + skip_steps

            if self.scheduler is None:
                self.setup_scheduler(total_steps, start_epoch, start_step)

            if skip_steps > 0:
                Logger(
                    f'Epoch [{epoch + 1}/{self.training_args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')

            self.train_epoch(epoch, train_dataloader, total_steps, skip_steps)

            if args.save_strategy == "epoch" and (epoch + 1) % args.save_interval == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    step=total_steps,
                    total_steps=total_steps,
                )

    def setup_scheduler(self, iters_per_epoch, start_epoch, start_step):
        """创建学习率调度器。需要知道每个 epoch 的步数，所以在 fit 里第一个 epoch 的 loader 建好后调用。"""
        args = self.training_args
        sched = torch.optim.lr_scheduler

        # 调度器按"参数更新次数"计步（每 accumulation_steps 个 batch 更新一次）
        updates_per_epoch = math.ceil(iters_per_epoch / args.accumulation_steps)
        total = updates_per_epoch * args.epochs
        warmup = int(total * args.warmup_ratio)
        decay = total - warmup  # 预热之后剩下的步数
        min_lr = args.learning_rate * args.min_lr_ratio

        # 续训时 optimizer 里的 lr 已经是衰减后的值，先还原成初始值，再让调度器"快进"到断点位置
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr'] = args.learning_rate

        if args.lr_scheduler == "cosine":
            main = sched.CosineAnnealingLR(self.optimizer, T_max=decay, eta_min=min_lr)
        elif args.lr_scheduler == "linear":
            main = sched.LinearLR(self.optimizer, start_factor=1.0, end_factor=args.min_lr_ratio, total_iters=decay)
        else:  # constant
            main = sched.ConstantLR(self.optimizer, factor=1.0, total_iters=decay)

        if warmup > 0:  # 先线性预热，再接主调度
            warm = sched.LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup)
            self.scheduler = sched.SequentialLR(self.optimizer, [warm, main], milestones=[warmup])
        else:
            self.scheduler = main

        # 快进：已经完成的参数更新次数。这样续训不需要把调度器存进 checkpoint
        done = start_epoch * updates_per_epoch + math.ceil(start_step / args.accumulation_steps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 屏蔽 "scheduler.step() 早于 optimizer.step()" 的提示
            for _ in range(done):
                self.scheduler.step()

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
                loss = self.train_step(batch)

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
                )

    def train_step(self, batch):
        input_ids, labels = batch

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )
        labels = labels.to(
            self.device,
            non_blocking=True,
        )

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
        )
        return outputs.loss

    def optimizer_step(self):
        """执行一次通用的参数更新，特殊训练算法可在子类覆写。"""
        args = self.training_args

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            args.grad_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def log_step(self, epoch, step, total_steps, loss):
        """记录一次训练日志。"""
        values = {
            "loss": loss.detach().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
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

        model = self.model

        if isinstance(model, DistributedDataParallel):
            model = model.module

        model = getattr(
            model,
            "_orig_mod",
            model,
        )

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

            model = self.model

            if isinstance(model, DistributedDataParallel):
                model = model.module

            model = getattr(
                model,
                "_orig_mod",
                model,
            )

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
