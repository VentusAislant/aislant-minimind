import os
import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import Features, Value, load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭分词器自己的多线程

__all__ = ["SFTDataset", "SFTCollator"]


class SFTCollator:
    """将监督微调样本动态补齐到当前 batch 的最长长度。"""

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


class SFTDataset(Dataset):
    """
    监督微调数据集。

    每行 JSON 数据格式：
        {
            "conversations": [
                {"role": "system", "content": "你是一个 AI 助手。"},
                {"role": "user", "content": "中国的首都是哪里？"},
                {
                    "role": "assistant",
                    "reasoning_content": "用户询问中国首都，直接回答即可。",
                    "content": "中国的首都是北京。"
                }
            ]
        }

    数据经过 chat template 展开并编码为固定长度序列。labels 中只有 assistant
    回复部分保留真实 token，其余位置设为 -100，因此不会参与交叉熵损失计算。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 显式声明字段类型，使不同样本缺少可选字段时仍能得到一致的数据结构。
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string')
            }]
        })

        # JSONL 文件统一作为 train split 加载，由 datasets 按索引读取样本。
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)

        # 将 chat template 中 assistant 消息的起止标记提前编码，后续据此寻找监督区间。
        self.assistant_start_ids = tokenizer(
            '<|im_start|>assistant\n', add_special_tokens=False
        ).input_ids
        self.assistant_end_ids = tokenizer(
            '<|im_end|>\n', add_special_tokens=False
        ).input_ids

    def __len__(self):
        return self.samples.num_rows

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

    def create_chat_prompt(self, conversations):
        """整理工具调用字段，并使用 tokenizer 的 chat template 展开整段对话。"""
        messages = []
        tools = None
        for message in conversations:
            # datasets 返回的行对象先转成普通字典，方便后续修改字段。
            message = dict(message)

            # tools 通常保存在 system 消息中；JSON 字符串需要先恢复为 Python 对象。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]

            # 同理，将 assistant 的工具调用参数从 JSON 字符串恢复为结构化对象。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)

        # 这里只生成完整训练文本，不追加新的 assistant 生成提示。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """生成 SFT 标签：只监督 assistant 回复，其他 token 使用 -100 屏蔽。"""
        # system、user、padding 等位置默认全部忽略。
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到一段 assistant 回复的开始位置
            if input_ids[i:i + len(self.assistant_start_ids)] == self.assistant_start_ids:
                start = i + len(self.assistant_start_ids)

                # 从当前位置开始扫描， 这个指针不断后移，找到 assistant 回复结束的位置
                end = start

                # 向后寻找当前 assistant 回复的结束标记。
                while end < len(input_ids):
                    if input_ids[end:end + len(self.assistant_end_ids)] == self.assistant_end_ids:
                        break
                    end += 1

                # assistant 正文和结束标记保留真实 token，用于计算语言模型损失。
                for j in range(start, min(end + len(self.assistant_end_ids), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.assistant_end_ids) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # 数据增强后，通过 chat template 将消息列表转换为模型实际看到的文本。
        conversations = self.pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = self.post_processing_chat(prompt)

        # Dataset 只负责编码和截断，padding 由 SFTCollator 按 batch 动态完成。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # labels 与 input_ids 等长，但只有 assistant 回复区间会参与损失计算。
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


if __name__ == '__main__':
    # 从项目根目录运行：python -m aislant_minimind.dataset.lm_dataset.sft
    from pathlib import Path
    from transformers import AutoTokenizer

    project_root = Path(__file__).resolve().parents[3]
    tokenizer_path = project_root / "model_weights" / "tokenizers" / "aislant_minimind_llama"
    dataset_path = project_root / "data" / "sft_t2t_mini.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    dataset = SFTDataset(
        jsonl_path=str(dataset_path),
        tokenizer=tokenizer,
        max_length=512,
    )

    print("数据集大小：", len(dataset))

    GREEN, RED_BOLD, RESET = "\033[32m", "\033[1;31m", "\033[0m"
    color = lambda s, on: f"{GREEN}{s}{RESET}" if on else s  # on=True 时用绿色
    title = lambda s: f"{RED_BOLD}{s}{RESET}"  # 红色加粗标题

    for sample_index in range(2):
        input_ids, labels = dataset[sample_index]
        ids, labs = input_ids.tolist(), labels.tolist()
        n = len(ids)
        while n > 0 and ids[n - 1] == tokenizer.pad_token_id:  # 去掉末尾 padding
            n -= 1

        # 连续的"参与 loss / 不参与"token 合并后再解码，保证中文不被拆坏
        text, start = "", 0
        for pos in range(1, n + 1):
            if pos == n or (labs[pos] != -100) != (labs[start] != -100):
                text += color(tokenizer.decode(ids[start:pos], skip_special_tokens=False), labs[start] != -100)
                start = pos

        print(f"\n===== 样本 {sample_index}（有效 {n} 个 token，padding 已省略，绿色 = 参与 loss）=====")
        print(title("\n[labels]"))
        print(" ".join(color(str(l), l != -100) for l in labs[:n]))
        print(title("\n[input_ids]"))
        print(" ".join(color(str(t), l != -100) for t, l in zip(ids[:n], labs[:n])))
        print(title("\n[input_ids 对应的文本]"))
        print(text)

        assert len(ids) == len(labs) <= 512
        assert any(l != -100 for l in labs), f"样本 {sample_index} 没有找到 assistant 回复，labels 全部为 -100"

    collator = SFTCollator(tokenizer.pad_token_id)
    batch_input_ids, batch_labels = collator([dataset[0], dataset[1]])
    print(title("\n[动态 padding 后的 batch]"))
    print("batch shape:", batch_input_ids.shape, batch_labels.shape)
    print("每个样本中被忽略的 label 数:", (batch_labels == -100).sum(dim=1).tolist())
