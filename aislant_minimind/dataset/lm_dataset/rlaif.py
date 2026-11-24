import os
import random

from datasets import load_dataset
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

__all__ = ["RLAIFDataset"]


class RLAIFDataset(Dataset):
    """
    AI 反馈强化学习数据集（Reinforcement Learning from AI Feedback）。

    每行 JSON 数据格式：
        {
            "conversations": [
                {"role": "user", "content": "中国的首都是哪里？"},
                {"role": "assistant", "content": ""}
            ]
        }

    最后一条空 assistant 消息只表示待生成位置。Dataset 会移除它，再通过 chat
    template 添加 assistant generation prompt。返回字符串 prompt，由 PPO/GRPO
    训练端按 batch 编码并生成回答。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")

    def __len__(self):
        return self.samples.num_rows

    def create_chat_prompt(self, conversations):
        """移除待生成的 assistant 消息，并构造 PPO/GRPO 使用的生成提示。"""
        conversations = self.pre_processing_chat(conversations)
        if conversations and conversations[-1].get("role") == "assistant":
            conversations = conversations[:-1]

        use_thinking = random.random() < self.thinking_ratio
        prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
        if len(input_ids) > self.max_length:
            # 强化学习生成必须保留末尾的当前问题和 assistant generation prompt。
            input_ids = input_ids[-self.max_length:]
            while True:
                prompt = self.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                encoded_length = len(self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                ).input_ids)
                if encoded_length <= self.max_length:
                    break
                # 少数 tokenizer 的 decode→encode 不是严格等长，继续从左侧移除溢出量。
                input_ids = input_ids[encoded_length - self.max_length:]
        return prompt

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "prompt": self.create_chat_prompt(sample["conversations"]),
            "answer": "",
        }

    @staticmethod
    def pre_processing_chat(conversations, add_system_ratio=0.2):
        """对原始对话做轻量数据增强：为部分普通对话随机添加 system 消息。"""
        # 工具调用数据依赖原始 system/tools 结构，因此保持不变。
        if any(conv.get('tools') for conv in conversations):
            return conversations

        system_prompts = [
            "你是服务全人类的AI助手，追求自由、平等与全人类解放。",
            "请平等地帮助每一个人，反对压迫与歧视。",
            "请以团结、互助和尊重为原则回答问题。",
            "请提供准确、有益的回答，不确定时如实说明。",
            "你是一个乐于助人的AI助手。",
            "请准确、清晰地回答用户的问题。",
            "请友好、客观地提供帮助。",
            "请认真理解问题并给出简洁的回答。",
            "You are an AI assistant serving all humanity, committed to freedom, equality, and human emancipation.",
            "Help everyone equally and oppose oppression and discrimination.",
            "Respond with solidarity, mutual aid, and respect.",
            "Provide accurate and helpful answers, and be honest about uncertainty.",
            "You are a helpful AI assistant.",
            "Answer the user's questions clearly and accurately.",
            "Be friendly, objective, and helpful.",
            "Understand the question carefully and respond concisely."
        ]

        # 对没有 system 消息的普通对话，按指定概率随机补充一条 system 消息。
        if conversations[0].get('role') != 'system' and random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(system_prompts)}] + conversations
        return conversations


if __name__ == "__main__":
    # 从项目根目录运行：python -m aislant_minimind.dataset.lm_dataset.rlaif
    from pathlib import Path
    from transformers import AutoTokenizer

    project_root = Path(__file__).resolve().parents[3]
    tokenizer = AutoTokenizer.from_pretrained(
        project_root / "model_weights" / "tokenizers" / "aislant_minimind_llama",
        local_files_only=True,
    )
    dataset = RLAIFDataset(
        jsonl_path=str(project_root / "data" / "rlaif.jsonl"),
        tokenizer=tokenizer,
        max_length=512,
    )

    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RED_BOLD = "\033[1;31m"
    RESET = "\033[0m"

    print(f"{RED_BOLD}数据集大小：{len(dataset)}{RESET}")
    print("蓝色 = 提交给策略模型的 prompt；绿色 = 在线生成的 answer")
    for sample_index in range(2):
        sample = dataset[sample_index]
        token_count = len(tokenizer(sample["prompt"], add_special_tokens=False).input_ids)
        print(f"\n{RED_BOLD}===== 样本 {sample_index}（prompt tokens: {token_count}）====={RESET}")
        print(f"prompt:\n{BLUE}{sample['prompt']}{RESET}")
        print(f"answer: {GREEN}{sample['answer']!r}{RESET}")
