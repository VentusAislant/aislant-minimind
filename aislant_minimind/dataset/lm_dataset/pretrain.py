import os
import torch
from torch.utils.data import Dataset

import datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭分词器自己的多线程

__all__ = ["PretrainDataset"]


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
        # 添加 pad token 方便组成 batch
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 标签(Ground Truth) token 在 pad 位置不需要计算损失, -100 会被交叉熵损失忽略
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


if __name__ == '__main__':
    # 简洁测试
    from pathlib import Path
    from transformers import AutoTokenizer

    project_root = Path(__file__).resolve().parents[3]
    tokenizer_path = project_root / "model_weights" / "tokenizers" / "llama"
    print(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    print(tokenizer)

    dataset_path = project_root / "data" / "pretrain_t2t_mini.jsonl"
    pretrain_dataset = PretrainDataset(data_path=str(dataset_path), tokenizer=tokenizer)
    print(len(pretrain_dataset))
    print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)  # 三个应互不相同
    input_ids, labels = pretrain_dataset[0]
    print(input_ids.shape, labels.shape)  # 都应是 (512,)
    print((labels == -100).sum().item(), "个 pad 位置")
    print(tokenizer.decode(input_ids[labels != -100]))  # 应该还原出原文，含首尾标记

    input_ids, labels = pretrain_dataset[1]
    print(input_ids.shape, labels.shape)
    print((labels == -100).sum().item(), "个 pad 位置")
    print(tokenizer.decode(input_ids[labels != -100]))
