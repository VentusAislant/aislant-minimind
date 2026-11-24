import os
import random

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

__all__ = ["DPODataset", "DPOCollator"]


class DPOCollator:
    """将 chosen/rejected 六个序列动态补齐到当前 batch 的统一长度。"""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        result = {}
        for key in ("x_chosen", "y_chosen", "x_rejected", "y_rejected"):
            result[key] = pad_sequence(
                [sample[key] for sample in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            )
        for key in ("mask_chosen", "mask_rejected"):
            result[key] = pad_sequence(
                [sample[key] for sample in batch],
                batch_first=True,
                padding_value=0,
            )

        # 训练时 chosen 和 rejected 会沿 batch 维拼接，因此二者序列长度必须一致。
        target_length = max(result["x_chosen"].shape[1], result["x_rejected"].shape[1])
        for key, tensor in result.items():
            if tensor.shape[1] == target_length:
                continue
            padding_value = 0 if key.startswith("mask_") else self.pad_token_id
            padding = tensor.new_full(
                (tensor.shape[0], target_length - tensor.shape[1]),
                padding_value,
            )
            result[key] = torch.cat((tensor, padding), dim=1)
        return result


class DPODataset(Dataset):
    """
    直接偏好优化数据集（Direct Preference Optimization）。

    每行 JSON 数据格式：
        {
            "chosen": [
                {"role": "user", "content": "请介绍一下北京。"},
                {"role": "assistant", "content": "北京是中国的首都……"}
            ],
            "rejected": [
                {"role": "user", "content": "请介绍一下北京。"},
                {"role": "assistant", "content": "北京是一座国外城市。"}
            ]
        }

    chosen 是偏好的回答，rejected 是较差的回答。两条对话分别通过 chat template
    展开，只在 assistant 回复位置生成值为 1 的 loss mask。Dataset 返回截断后的
    变长序列，padding 由 DPOCollator 按 batch 动态完成。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")

        # 与 SFTDataset 使用同一组 assistant 起止标记。
        self.assistant_start_ids = tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        ).input_ids
        self.assistant_end_ids = tokenizer(
            "<|im_end|>\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return self.samples.num_rows

    def generate_loss_mask(self, input_ids):
        """仅标记 assistant 回复和回复结束标记，其他 token 不参与 DPO log-prob。"""
        loss_mask = [0] * len(input_ids)
        index = 0
        while index < len(input_ids):
            if input_ids[index:index + len(self.assistant_start_ids)] != self.assistant_start_ids:
                index += 1
                continue

            start = index + len(self.assistant_start_ids)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.assistant_end_ids)] == self.assistant_end_ids:
                    break
                end += 1

            supervised_end = min(end + len(self.assistant_end_ids), len(input_ids))
            loss_mask[start:supervised_end] = [1] * (supervised_end - start)
            index = supervised_end
        return loss_mask

    def encode_conversation(self, conversations, remove_empty_think):
        """将一条 chosen 或 rejected 对话展开、编码并生成 loss mask。"""
        prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt = self.post_processing_chat(
            prompt,
            remove_empty_think=remove_empty_think,
        )
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        loss_mask = self.generate_loss_mask(input_ids)

        # DPO 训练代码显式使用错开一位的 x/y，并用 mask 选择 assistant token。
        return (
            torch.tensor(input_ids[:-1], dtype=torch.long),
            torch.tensor(input_ids[1:], dtype=torch.long),
            torch.tensor(loss_mask[1:], dtype=torch.long),
        )

    def __getitem__(self, index):
        sample = self.samples[index]

        # chosen/rejected 必须采用相同的空 think 标签增强策略，避免引入格式偏差。
        remove_empty_think = random.random() > 0.2
        x_chosen, y_chosen, mask_chosen = self.encode_conversation(
            sample["chosen"], remove_empty_think
        )
        x_rejected, y_rejected, mask_rejected = self.encode_conversation(
            sample["rejected"], remove_empty_think
        )
        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    @staticmethod
    def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
        """随机移除空的思考标签，避免模型始终学习同一种固定输出格式。"""
        # 默认保留 20% 的空 <think> 标签，移除其余 80%。
        if '<think>\n\n</think>\n\n' in prompt_content:
            if remove_empty_think is None:
                remove_empty_think = random.random() > empty_think_ratio
            if remove_empty_think:
                prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
        return prompt_content


if __name__ == "__main__":
    # 从项目根目录运行：python -m aislant_minimind.dataset.lm_dataset.dpo
    from pathlib import Path
    from transformers import AutoTokenizer

    project_root = Path(__file__).resolve().parents[3]
    tokenizer = AutoTokenizer.from_pretrained(
        project_root / "model_weights" / "tokenizers" / "aislant_minimind_llama",
        local_files_only=True,
    )
    dataset = DPODataset(
        jsonl_path=str(project_root / "data" / "dpo.jsonl"),
        tokenizer=tokenizer,
        max_length=512,
    )

    print("数据集大小：", len(dataset))

    GREEN, RED_BOLD, RESET = "\033[32m", "\033[1;31m", "\033[0m"
    color = lambda text, on: f"{GREEN}{text}{RESET}" if on else text
    title = lambda text: f"{RED_BOLD}{text}{RESET}"

    samples = [dataset[0], dataset[1]]
    for sample_index, sample in enumerate(samples):
        print(f"\n===== 样本 {sample_index} =====")
        for prefix in ("chosen", "rejected"):
            x_ids = sample[f"x_{prefix}"].tolist()
            y_ids = sample[f"y_{prefix}"].tolist()
            masks = sample[f"mask_{prefix}"].tolist()

            # mask 与 y（下一 token 目标）对齐；绿色 token 才参与 DPO log-prob。
            labels = [token_id if enabled else -100 for token_id, enabled in zip(y_ids, masks)]
            text, start = "", 0
            for position in range(1, len(y_ids) + 1):
                if position == len(y_ids) or masks[position] != masks[start]:
                    text += color(
                        tokenizer.decode(y_ids[start:position], skip_special_tokens=False),
                        bool(masks[start]),
                    )
                    start = position

            print(title(
                f"\n[{prefix}] "
                f"（{len(x_ids)} 个 token，绿色 = 参与 DPO log-prob）"
            ))
            print(title("\n[labels：mask=0 的位置等价为 -100]"))
            print(" ".join(
                color(str(label), enabled)
                for label, enabled in zip(labels, masks)
            ))
            print(title("\n[x_input_ids]"))
            print(" ".join(str(token_id) for token_id in x_ids))
            print(title("\n[y_target_ids]"))
            print(" ".join(
                color(str(token_id), enabled)
                for token_id, enabled in zip(y_ids, masks)
            ))
            print(title("\n[y_target_ids 对应的文本]"))
            print(text)

            assert len(x_ids) == len(y_ids) == len(masks) <= 511
            assert any(masks), f"样本 {sample_index} 的 {prefix} 没有 assistant 监督区间"

    batch = DPOCollator(tokenizer.pad_token_id)(samples)
    print(title("\n[动态 padding 后的 batch]"))
    print({key: value.shape for key, value in batch.items()})
