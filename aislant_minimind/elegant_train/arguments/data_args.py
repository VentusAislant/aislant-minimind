from dataclasses import dataclass


@dataclass
class DataArguments:
    """数据集和 DataLoader 参数。"""

    data_path: str | None = None
    max_seq_len: int = 512
    num_workers: int = 8


@dataclass
class SFTDataArguments(DataArguments):
    chat_template: str | None = None