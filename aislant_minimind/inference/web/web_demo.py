"""简洁的 ChatGPT 风格 Streamlit 对话界面。

从项目根目录运行：
streamlit run aislant_minimind/inference/web/web_demo.py
"""

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


st.set_page_config(
    page_title="Aislant Minimind",
    page_icon="🤖",
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
    .brand-mark {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        color: #202123;
        background: #eeeeee;
        font-size: 1rem;
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
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
        background: #eef6ff;
        border-radius: 18px;
        padding: 0.65rem 1rem;
        max-width: 72%;
        margin-left: auto;
    }
    [data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto 1rem;
    }
    .welcome {
        text-align: center;
        padding: 13vh 1rem 5rem;
    }
    .robot-mark {
        width: 58px;
        height: 58px;
        display: grid;
        place-items: center;
        margin: 0 auto 1rem;
        border-radius: 18px;
        background: #f0f0f0;
        border: 1px solid #e3e3e3;
        font-size: 1.7rem;
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


with st.sidebar:
    st.markdown("## 🤖 Aislant Minimind")
    st.caption("本地智能助手")
    st.divider()

    with st.expander("⚙️ 模型与生成设置", expanded=True):
        model_path = st.text_input("模型路径", value="outputs/simple_pretrain_single")
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
        <div class="brand"><span class="brand-mark">🤖</span>Aislant Minimind</div>
        <div class="model-status">{escape(Path(model_path).name)}</div>
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
            <div class="robot-mark">🤖</div>
            <h1>有什么可以帮忙的？</h1>
            <p>与 Aislant Minimind 开始一段对话</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "🙂"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("thinking") and show_thinking:
            with st.expander("思考过程", expanded=False):
                st.markdown(message["thinking"])
        st.markdown(message["content"])

prompt = st.chat_input("给 Aislant Minimind 发送消息")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)

    try:
        model, tokenizer, device = load_model(model_path)
    except Exception as error:
        st.error(f"模型加载失败：{error}")
        st.stop()

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
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    Thread(target=model.generate, kwargs=generation_kwargs, daemon=True).start()

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        raw_response = ""
        for text in streamer:
            raw_response += text
            _, visible_answer = split_response(raw_response)
            placeholder.markdown(visible_answer or "思考中…")

        thinking, answer = split_response(raw_response)
        placeholder.markdown(answer)
        if thinking and show_thinking:
            with st.expander("思考过程", expanded=False):
                st.markdown(thinking)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
    })
