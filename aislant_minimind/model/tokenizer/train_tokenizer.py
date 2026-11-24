"""
Tokenizer (分词器)
    - 核心功能是将一个文本序列，划分为离散的 token 序列，每个 token 对应一个索引 (index)
    - 建议直接复用目前主流模型的 tokenizer 会更有模型复用性
    - 此脚本可以用于学习
    - 分词器训练流程:
        1. 提取语料库文本，一般需要语料规模足够多，这样分词器会更加准确
        2. 选择分词算法与预分词方式
            - 算法: BPE（从 256 个字节出发，反复合并出现最频繁的相邻 token 对，直到词表达到目标大小）
            - 预分词: ByteLevel，把文本先转成 UTF-8 字节，任何字符都能表示，不会出现 OOV
        3. 定义特殊 token
            - 核心 token: <|begin_of_text|>、<|end_of_text|>、<|text_pad|>
            - 控制 token: <|im_start|>、<|im_end|> 等，用来标记对话边界
            - 附加 token: <think>、<tool_call> 等，要在解码时保留，所以标记为非 special
            - 预留 buffer token: 以后想加新 token 时不用改词表大小，也不用重训模型
            - 特殊 token 会放在词表最前面（ID 0 ~ N-1）
        4. 训练 BPE（BpeTrainer）
            - vocab_size: 词表大小，是一个权衡
                · 太小: 同样的文本被切成更多 token，序列变长，压缩率低
                · 太大: 后续模型的 embedding 和输出层参数变多，小模型里会占掉大部分参数，低频 token 也学不好
            - initial_alphabet: 256 个字节字符，保证所有输入都能被编码
        5. 保存分词器
            - tokenizer.json: 完整的分词器（词表 + merge 规则 + 特殊 token 等）
            - vocab.json / merges.txt: 词表和合并规则
            - tokenizer_config.json: bos/eos/pad、特殊 token 映射、chat_template
        6. 评估分词器
            - chat template: 渲染一段多轮对话，看格式是否正确
            - 编解码一致性: decode(encode(text)) == text
            - 压缩率: 字符数 / token 数，越大说明词表越高效
            - 流式解码: 一个汉字可能被拆成多个 token，要攒够完整字节再输出，避免乱码 (\ufffd)

    - 主流模型的 tokenizer 对比（数字对应具体版本，新版本可能会变）:

        模型          算法                词表大小       说明
        ----------    ----------------    ----------    ------------------------------------
        Llama 2       SentencePiece BPE   32K           英文为主，中文压缩率低
        Llama 3       tiktoken 体系 BPE   约 128K       100K 来自 tiktoken，另加约 28K 非英语 token
        Qwen3         byte-level BPE      151,669       多语言，数字拆成单个数字，预分词类似 GPT-4

    - 观察:
        1. 主流模型基本都是 BPE，而且大多是 byte-level，核心算法与本脚本相同
        2. 差别主要在词表规模、预分词规则、训练语料的多语言配比
        3. 词表越大，压缩率越高（同样文本 token 更少），但 embedding / 输出层参数也越多
"""

import argparse
import json
from pathlib import Path

from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
from tqdm.auto import tqdm

from .exrtra_tokens import ALL_SPECIAL_TOKENS, ALL_TOKENS, CORE_SPECIAL_TOKENS, OTHER_SPECIAL_TOKENS

DEFAULT_MAX_LINES = 1_000_000

# Tokenizer 评估样例
EVAL_MESSAGES = [
    {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
    {"role": "user", "content": "你来自哪里？"},
    {"role": "assistant", "content": "我来自月球"},
    {"role": "user", "content": "你到底来自哪里？"},
    {"role": "assistant", "content": "我来自地球"},
]

COMPRESSION_TEST_TEXTS = [
    # 中文样例
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
    "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
    # 英文样例
    "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
    "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
    # 中英文混合样例
    "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="训练 Aislant MiniMind tokenizer")
    parser.add_argument(
        "--data_paths",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个 JSONL 数据文件"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("model_weights/tokenizers/aislant_minimind"),
        help="tokenizer 输出目录",
    )
    parser.add_argument("--vocab_size", type=int, default=6400, help="目标词表大小")
    parser.add_argument(
        "--max_lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help=f"最多读取的有效文本数，默认 {DEFAULT_MAX_LINES}，0 表示全部",
    )
    parser.add_argument("--skip_eval", action="store_true", help="训练后跳过 tokenizer 评估")
    return parser.parse_args()


# 1. 提取语料文本
def get_texts(data_paths, max_lines=0):
    """
    逐行产出用于训练 BPE 的文本，兼容两种主线数据格式。

    data_paths: 单个文件路径（str / Path），或由多个路径组成的 list / tuple。
    max_lines:  所有文件合计最多读取多少条有效文本，0 表示全部读取。

    pretrain_t2t(_mini).jsonl 是 {"text": ...}，
    sft_t2t(_mini).jsonl 是 {"conversations": [{"role":..., "content":...}, ...]}。
    读不到任何文本时，BPE 只会基于初始字母表训练，得到一个没有 merge 的词表且不报错，
    所以最后会显式检查并抛出异常。
    """

    # 获取所有输入文件
    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]
    data_paths = [Path(p) for p in data_paths]
    if not data_paths:
        raise ValueError("data_paths 不能为空")
    for p in data_paths:
        if not p.is_file():
            raise FileNotFoundError(f"数据文件不存在: {p}")

    # 将所有输入文件的语料提取到迭代器
    used = 0
    for path in data_paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if max_lines and used >= max_lines:
                    return
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                # 预训练数据直接读取 text；SFT 数据拼接每轮对话的 content。
                if 'text' in data:  # pretrain data
                    text = str(data['text'])
                else:  # sft data
                    contents = [
                        item.get('content')
                        for item in data.get('conversations', [])
                        if isinstance(item, dict) and item.get('content')
                    ]
                    text = "\n".join(contents)
                if text.strip():
                    used += 1
                    yield text

    if used == 0:
        raise ValueError(
            f'{[str(p) for p in data_paths]} 中没有可用文本：每行应为 {{"text": ...}} 或 '
            f'{{"conversations": [{{"role":..., "content":...}}, ...]}}。'
            f'继续训练只会得到一个空词表。'
        )


# 2. 训练 tokenizer
def train_tokenizer(
        data_paths,
        tokenizer_dir,
        vocab_size,
        max_lines
):
    # 使用 ByteLevel 预分词，确保任意 UTF-8 文本都能够被编码。
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 特殊 token 和 ByteLevel 基础字节会占用固定词表空间。
    if vocab_size <= len(ALL_TOKENS) + len(pre_tokenizers.ByteLevel.alphabet()):
        raise ValueError("vocab_size 必须大于特殊 token 数量与 ByteLevel 基础字节数量之和")
    if max_lines < 0:
        raise ValueError("max_lines 不能小于 0")

    # 先流式统计有效序列数量。计数阶段没有预知总数，只显示不断增长的已处理数量；
    # 得到准确数量后再交给 tokenizer，使其训练进度条能够显示正确的总进度。
    sequence_count = sum(
        1 for _ in tqdm(
            get_texts(data_paths, max_lines),
            desc="Counting sequences",
            unit=" sequences",
        )
    )

    # 基于语料迭代器训练 BPE，不需要一次性把全部文本载入内存。
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始字母表：256 个字节字符，保证任何输入都能编码
        special_tokens=ALL_TOKENS  # 特殊 / 附加 / buffer token，会排在词表最前面
    )

    texts = get_texts(data_paths, max_lines)  # 得到文本生成器（边读边产出，不会一次性占满内存）
    tokenizer.train_from_iterator(
        texts,
        trainer=trainer,
        length=sequence_count,
    )  # 开始训练：反复合并最频繁的相邻 token 对
    tokenizer.decoder = decoders.ByteLevel()  # 解码器：把字节字符还原成原始文本，要和预分词配套
    tokenizer.add_special_tokens(ALL_SPECIAL_TOKENS)

    # 同时保存统一的 tokenizer.json 和兼容 BPE 的 vocab/merges 文件。
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.model.save(str(tokenizer_dir))
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in ALL_SPECIAL_TOKENS:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    # 构造 Transformers 加载时使用的特殊 token 元数据。
    added_tokens_decoder = {}
    for token in ALL_TOKENS:
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in ALL_SPECIAL_TOKENS
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": OTHER_SPECIAL_TOKENS,
        "bos_token": CORE_SPECIAL_TOKENS["bos_token"],
        "clean_up_tokenization_spaces": False,
        "eos_token": CORE_SPECIAL_TOKENS["eos_token"],
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": CORE_SPECIAL_TOKENS["pad_token"],
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": None,
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
    }

    with open(tokenizer_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"Tokenizer training completed: {tokenizer_dir}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    # 重新从磁盘加载，验证保存结果能够被 Transformers 正常识别。
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 检查 chat template 渲染和编码、解码的一致性。
    new_prompt = tokenizer.apply_chat_template(
        EVAL_MESSAGES,
        tokenize=False
    )
    print('-' * 100)
    print(new_prompt)
    print('-' * 100)
    print('tokenizer词表长度：', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")
    print('-' * 100)
    print('压缩率测试（Chars/Tokens）：')
    # 分别统计中文、英文和混合文本的字符/token 压缩率。
    total_compression = 0
    for i, text in enumerate(COMPRESSION_TEST_TEXTS):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"样本 {i + 1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")

    print(f"平均压缩率: {total_compression / len(COMPRESSION_TEST_TEXTS):.2f}")
    print('-' * 100)
    print('流式解码（字节缓冲）测试：')
    # ByteLevel token 可能只包含一个汉字的部分字节，需要缓存到能够完整解码后再输出。
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in
                          (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []


if __name__ == '__main__':
    """
    从项目根目录运行：

    python -m aislant_minimind.model.tokenizer.train_tokenizer \
        --data_paths data/pretrain_t2t.jsonl data/sft_t2t_mini.jsonl \
        --vocab_size 6400
    """
    args = parse_args()

    # 1. 训练 tokenizer
    train_tokenizer(
        data_paths=args.data_paths,
        tokenizer_dir=args.output_dir,
        vocab_size=args.vocab_size,
        max_lines=args.max_lines,
    )

    # 2. 评估 tokenizer
    if not args.skip_eval:
        eval_tokenizer(args.output_dir)
