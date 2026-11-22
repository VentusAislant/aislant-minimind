from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingArguments:
    """所有训练阶段共有的运行参数。"""

    output_dir: str = "outputs"

    resume_from_checkpoint: str | None = None

    save_strategy: Literal[
        "steps",
        "epoch",
        "no",
    ] = "steps"
    save_interval: int = 1000

    log_interval: int = 100

    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-4
    accumulation_steps: int = 1
    grad_clip: float = 1.0

    dtype: str = "bfloat16"
    device: str = "auto"
    seed: int = 42

    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.1

    use_compile: bool = False
    use_wandb: bool = False
    wandb_project: str = "Aislant-MiniMind"

    def __post_init__(self):
        if self.save_strategy not in {"steps", "epoch", "no"}:
            raise ValueError(
                "save_strategy 必须是 'steps'、'epoch' 或 'no'"
            )

        if self.save_strategy != "no" and self.save_interval <= 0:
            raise ValueError(
                "启用 checkpoint 保存时，save_interval 必须大于 0"
            )


@dataclass
class PretrainArguments(TrainingArguments):
    output_dir: str = "outputs/pretrain"

    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 5e-4
    accumulation_steps: int = 8

    wandb_project: str = "Aislant-MiniMind-Pretrain"


@dataclass
class SFTArguments(TrainingArguments):
    output_dir: str = "outputs/full_sft"

    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-5
    accumulation_steps: int = 1

    wandb_project: str = "Aislant-MiniMind-SFT"


@dataclass
class DPOArguments(TrainingArguments):
    output_dir: str = "outputs/dpo"
    learning_rate: float = 1e-6

    beta: float = 0.1
