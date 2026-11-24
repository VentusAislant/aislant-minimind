"""简洁的 ChatGPT 风格 Streamlit 对话界面。

从项目根目录运行：
streamlit run aislant_minimind/inference/web/web_demo.py
"""

import gc
import sys
from html import escape
from pathlib import Path
from threading import Thread

import streamlit as st
import torch
from transformers import AutoTokenizer, TextIteratorStreamer

# Streamlit 直接执行此文件时，需要把项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aislant_minimind.model.modeling_minimind_llama import MiniMindLlamaForCausalLM


CUSTOM_MODEL_PATH = "自定义路径…"
COMPLETION_MODE = "补全（Pretrain）"
CHAT_MODE = "Chat（SFT）"


def discover_models():
    """自动发现项目内保存为 Hugging Face 格式的模型目录。"""
    model_paths = []
    for model_root in (PROJECT_ROOT / "outputs", PROJECT_ROOT / "model_weights" / "models"):
        if not model_root.exists():
            continue
        for config_path in model_root.rglob("config.json"):
            model_dir = config_path.parent
            has_weights = (
                (model_dir / "model.safetensors").is_file()
                or (model_dir / "pytorch_model.bin").is_file()
            )
            has_tokenizer = (
                (model_dir / "tokenizer.json").is_file()
                or (model_dir / "tokenizer.model").is_file()
            )
            if has_weights and has_tokenizer:
                model_paths.append(str(model_dir.relative_to(PROJECT_ROOT)))
    return sorted(set(model_paths))


st.set_page_config(
    page_title="Aislant Minimind",
    page_icon=":material/smart_toy:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; color: #202123; }
    [data-testid="stSidebar"] {
        background: #f9f9f9;
        border-right: 1px solid #ececec;
    }
    [data-testid="stSidebar"] * { color: #242424; }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input {
        background: #ffffff;
        border-color: #dedede;
        color: #242424;
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        border: 1px solid #e4e4e4;
        background: #ffffff;
        color: #242424;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #eeeeee;
        border-color: #d8d8d8;
    }
    .block-container {
        max-width: 900px;
        padding-top: 1rem;
        padding-bottom: 7rem;
    }
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.1rem 0 1rem;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 2rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1rem;
        font-weight: 650;
    }
    .model-status {
        color: #6b6b6b;
        font-size: 0.85rem;
    }
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.8rem 0;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        line-height: 1.75;
        font-size: 1rem;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        color: #202123;
        background: #f2f2f2;
        border: 1px solid #dedede;
    }
    [data-testid="stChatMessageAvatarAssistant"] * {
        color: #202123;
    }
    [data-testid="stChatMessageAvatarUser"] {
        position: relative;
        color: #ffffff;
        background: #202123;
        border: 1px solid #202123;
    }
    [data-testid="stChatMessageAvatarUser"] * {
        display: none;
    }
    [data-testid="stChatMessageAvatarUser"]::before {
        content: "";
        position: absolute;
        top: 7px;
        left: 50%;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #ffffff;
        transform: translateX(-50%);
    }
    [data-testid="stChatMessageAvatarUser"]::after {
        content: "";
        position: absolute;
        bottom: 6px;
        left: 50%;
        width: 15px;
        height: 8px;
        border-radius: 9px 9px 4px 4px;
        background: #ffffff;
        transform: translateX(-50%);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        justify-content: flex-start;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        background: #eef6ff;
        border-radius: 18px;
        padding: 0.65rem 1rem;
        width: fit-content;
        flex: 0 1 auto;
        max-width: 72%;
        margin-left: auto;
        margin-right: 0;
    }
    [data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto 1rem;
    }
    [data-testid="stChatInput"] > div {
        background: #f4f4f4 !important;
        border: 1px solid #d7d7d7 !important;
        border-radius: 26px !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"]:focus-within > div {
        border-color: #9b9b9b !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        background: transparent !important;
        border: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;
    }
    .welcome {
        text-align: center;
        padding: 13vh 1rem 5rem;
    }
    .welcome h1 {
        font-size: 2rem;
        margin-bottom: 0.7rem;
    }
    .welcome p {
        color: #777;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载模型…")
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MiniMindLlamaForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.half()
    return model.eval().to(device), tokenizer, device


def split_response(text: str):
    """把 <think>...</think> 与最终回答分开展示。"""
    if "</think>" not in text:
        return "", text
    thinking, answer = text.split("</think>", 1)
    return thinking.removeprefix("<think>").strip(), answer.strip()


def clear_chat():
    st.session_state.messages = []


def switch_model():
    """切换模型时释放旧模型，并避免混用上一个模型的对话。"""
    clear_chat()
    load_model.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


with st.sidebar:
    st.markdown("## Aislant Minimind")
    st.caption("本地智能助手")
    st.divider()

    with st.expander("⚙️ 模型与生成设置", expanded=True):
        available_models = discover_models()
        model_options = available_models + [CUSTOM_MODEL_PATH]
        default_model = (
            "outputs/simple_pretrain_single"
            if "outputs/simple_pretrain_single" in available_models
            else model_options[0]
        )
        selected_model = st.selectbox(
            "选择模型",
            model_options,
            index=model_options.index(default_model),
            on_change=switch_model,
            key="selected_model",
        )
        if selected_model == CUSTOM_MODEL_PATH:
            model_path = st.text_input(
                "模型路径",
                value="outputs/simple_pretrain_single",
                on_change=switch_model,
                key="custom_model_path",
            )
        else:
            model_path = selected_model
        # Pretrain 模型做文本补全；后续 SFT、DPO、RL 等模型都使用 Chat 模板。
        inference_mode = (
            COMPLETION_MODE
            if "pretrain" in str(model_path).lower()
            else CHAT_MODE
        )
        st.caption(f"推理模式：{inference_mode}")
        max_new_tokens = st.slider("最大生成长度", 64, 4096, 512, step=64)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.05)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, step=0.05)
        show_thinking = st.toggle("显示思考过程", value=True)

    st.button("🗑️ 清空对话记录", on_click=clear_chat, use_container_width=True)

    st.divider()
    st.caption("模型在本机运行 · 对话不会上传")


st.markdown(
    f"""
    <div class="chat-header">
        <div class="brand">Aislant Minimind</div>
        <div class="model-status">{escape(Path(model_path).name)} · {escape(inference_mode)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome">
            <h1>有什么可以帮忙的？</h1>
            <p>与 Aislant Minimind 开始一段对话</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("thinking") and show_thinking:
            with st.expander("思考过程", expanded=False):
                st.markdown(message["thinking"])
        st.markdown(message["content"])

input_placeholder = "输入需要续写的文本" if inference_mode == COMPLETION_MODE else "给 Aislant Minimind 发送消息"
prompt = st.chat_input(input_placeholder)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        model, tokenizer, device = load_model(model_path)
    except Exception as error:
        st.error(f"模型加载失败：{error}")
        st.stop()

    if inference_mode == CHAT_MODE:
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        input_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=show_thinking,
        )
    else:
        # Pretrain 模型没有学习对话模板，直接对当前文本做续写。
        input_text = prompt
    # chat template 已经负责安排特殊 token，因此这里不让 tokenizer 重复添加。
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=False,
    )
    if inference_mode == COMPLETION_MODE and tokenizer.bos_token_id is not None:
        # PretrainDataset 训练时手工在原文前添加 BOS，补全推理必须保持相同格式。
        bos = torch.full(
            (inputs.input_ids.shape[0], 1),
            tokenizer.bos_token_id,
            dtype=inputs.input_ids.dtype,
        )
        inputs["input_ids"] = torch.cat((bos, inputs.input_ids), dim=1)
        inputs["attention_mask"] = torch.cat(
            (torch.ones_like(bos), inputs.attention_mask),
            dim=1,
        )
    inputs = inputs.to(device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    # SFT 的每轮 assistant 回复以 <|im_end|> 结束；补全模型才使用普通 EOS。
    # 如果 Chat 模式仍使用 eos_token_id，模型生成 <|im_end|> 后不会停止，
    # 后续无监督内容会继续生成并污染下一轮对话历史。
    if inference_mode == CHAT_MODE:
        stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if stop_token_id == tokenizer.unk_token_id:
            raise ValueError("当前 tokenizer 缺少 Chat 模式所需的 <|im_end|> token")
    else:
        stop_token_id = tokenizer.eos_token_id

    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_token_id,
        "streamer": streamer,
    }

    Thread(target=model.generate, kwargs=generation_kwargs, daemon=True).start()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        raw_response = ""
        for text in streamer:
            raw_response += text
            visible_answer = (
                split_response(raw_response)[1]
                if inference_mode == CHAT_MODE
                else raw_response
            )
            placeholder.markdown(visible_answer or "思考中…")

        if inference_mode == CHAT_MODE:
            thinking, answer = split_response(raw_response)
        else:
            thinking, answer = "", raw_response
        placeholder.markdown(answer)
        if thinking and show_thinking:
            with st.expander("思考过程", expanded=False):
                st.markdown(thinking)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
    })
