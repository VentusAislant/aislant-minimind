import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭分词器自己的多线程

__all__ = ["PretrainDataset", "PretrainCollator"]


class PretrainCollator:
    """将预训练样本动态补齐到当前 batch 的最长长度"""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return input_ids, labels


class PretrainDataset(Dataset):
    """
    预训练数据集
        数据文件格式:
            {"text": "今天天气很好，适合出去散步。公园里的花都开了……"}
            {"text": "机器学习是人工智能的一个分支，它让计算机从数据中学习规律……"}
            {"text": "The transformer ... become the foundation of modern language models."}
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        """
        # 比较简单的加载数据集方式，但是会把全部数据放入内存，如果数据集过大则不合适
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        """

        """
        # 可以使用 hugging face 的 datasets 库，里面进行了优化，不会一次全部加载，通过映射，模型在需要时才读取数据
            1. data_files 指定实际的文件地址, 可以传入列表
            2. path 指定读取的 data_files 的文件类型, json包含 jsonl
            3. split 介绍如下
                (1) data_files 传入字典指定不同 split
                    load_dataset('json', data_files={'train': 'train.jsonl', 'val': 'val.jsonl'}) 
                (2) field 额外字段来固定 data_files 的 split
                    train = load_dataset('json', data_files='data.json', field='train', split='train')
                    val   = load_dataset('json', data_files='data.json', field='val',   split='train')
                (3) 如果没有指定，认全部数据都在 train 字段下
            4. 返回的是 datasets.Dataset 对象
                不指定 split 返回的是 
                    DatasetDict({
                        train: Dataset({features: ['text'], num_rows: 100000})
                    })
                指定了 split 返回的就是
                    Dataset({features: ['text'], num_rows: 100000})
                
                Dataset 对象可以直接用索引读取
                    ds[0]             # 第 0 行 → {'text': '今天天气很好……'}   （字典）
                    ds[0:3]           # 前 3 行 → {'text': ['…', '…', '…']}     （字典，值是列表）
                    ds['text']        # 整列 → 所有 text 组成的列表
        """
        self.samples = datasets.load_dataset(
            'json',
            data_files=data_path,
            split='train',
        )

    def __len__(self):
        return self.samples.num_rows

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,  # 不让分词器自动添加首尾特殊 token
            max_length=self.max_length - 2,  # 给首尾特殊token 留位置
            truncation=True
        ).input_ids

        # 添加首尾特殊 token
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # Dataset 只返回截断后的变长序列，padding 由 PretrainCollator 完成。
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        return input_ids, labels


if __name__ == '__main__':
    # 从项目根目录运行：python -m aislant_minimind.dataset.lm_dataset.pretrain
    from pathlib import Path
    from transformers import AutoTokenizer

    project_root = Path(__file__).resolve().parents[3]
    tokenizer_path = project_root / "model_weights" / "tokenizers" / "aislant_minimind_llama"
    dataset_path = project_root / "data" / "pretrain_t2t_mini.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    dataset = PretrainDataset(
        data_path=str(dataset_path),
        tokenizer=tokenizer,
        max_length=512,
    )

    print("数据集大小：", len(dataset))
    print(
        "特殊 token id：",
        f"bos={tokenizer.bos_token_id}",
        f"eos={tokenizer.eos_token_id}",
        f"pad={tokenizer.pad_token_id}",
    )

    GREEN, RED_BOLD, RESET = "\033[32m", "\033[1;31m", "\033[0m"
    color = lambda text, on: f"{GREEN}{text}{RESET}" if on else text
    title = lambda text: f"{RED_BOLD}{text}{RESET}"

    samples = [dataset[0], dataset[1]]
    for sample_index, (input_ids, labels) in enumerate(samples):
        ids, labs = input_ids.tolist(), labels.tolist()

        # PretrainDataset 不做 padding，因此当前所有 token 都参与 loss。
        text = color(
            tokenizer.decode(ids, skip_special_tokens=False),
            on=True,
        )

        print(
            f"\n===== 样本 {sample_index}"
            f"（有效 {len(ids)} 个 token，绿色 = 参与 loss）====="
        )
        print(title("\n[labels]"))
        print(" ".join(color(str(label), label != -100) for label in labs))
        print(title("\n[input_ids]"))
        print(" ".join(
            color(str(token_id), label != -100)
            for token_id, label in zip(ids, labs)
        ))
        print(title("\n[input_ids 对应的文本]"))
        print(text)

        assert len(ids) == len(labs) <= 512
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id
        assert all(label != -100 for label in labs)

    collator = PretrainCollator(tokenizer.pad_token_id)
    batch_input_ids, batch_labels = collator(samples)
    print(title("\n[动态 padding 后的 batch]"))
    print("batch shape:", batch_input_ids.shape, batch_labels.shape)
    print("每个样本中被忽略的 label 数:", (batch_labels == -100).sum(dim=1).tolist())
