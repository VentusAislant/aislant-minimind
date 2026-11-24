import json
import os

from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

__all__ = ["AgentRLDataset"]


class AgentRLDataset(Dataset):
    """
    Agent 强化学习数据集。

    每行 JSON 数据格式：
        {
            "conversations": [
                {
                    "role": "system",
                    "content": "你是一个 AI 助手。",
                    "tools": "[{...}]"
                },
                {"role": "user", "content": "北京现在几点？"},
                {"role": "assistant", "content": ""}
            ],
            "gt": ["2026-11-24 12:00:00"]
        }

    Dataset 返回待生成的消息、工具定义和标准答案。末尾的空
    assistant 消息只是占位符，返回前会将其移除。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({
            "conversations": [{
                "role": Value("string"),
                "content": Value("string"),
                "reasoning_content": Value("string"),
                "tools": Value("string"),
                "tool_calls": Value("string"),
            }],
            "gt": [Value("string")],
        })
        self.samples = load_dataset(
            "json",
            data_files=jsonl_path,
            split="train",
            features=features,
        )

    def __len__(self):
        return self.samples.num_rows

    @staticmethod
    def parse_conversations(conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = (
                    json.loads(message["tools"])
                    if isinstance(message["tools"], str)
                    else message["tools"]
                )
            messages.append(message)

        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return messages, tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample["conversations"])
        return {
            "messages": messages,
            "tools": tools,
            "gt": sample["gt"],
        }


if __name__ == "__main__":
    # 从项目根目录运行：python -m aislant_minimind.dataset.lm_dataset.agent_rl
    from pathlib import Path
    from transformers import AutoTokenizer

    project_root = Path(__file__).resolve().parents[3]
    tokenizer = AutoTokenizer.from_pretrained(
        project_root / "model_weights" / "tokenizers" / "aislant_minimind_llama",
        local_files_only=True,
    )
    dataset = AgentRLDataset(
        jsonl_path=str(project_root / "data" / "agent_rl.jsonl"),
        tokenizer=tokenizer,
        max_length=512,
    )

    print("数据集大小：", len(dataset))
    for sample_index in range(2):
        sample = dataset[sample_index]
        print(f"\n===== 样本 {sample_index} =====")
        print("messages:", sample["messages"])
        print("tools:", sample["tools"])
        print("gt:", sample["gt"])
