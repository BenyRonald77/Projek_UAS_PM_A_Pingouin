import os
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import traceback

# ================================
# STREAMLIT PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Mirota Kampus Chatbot (LoRA)",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chatbot Mirota Kampus (Fine-tuned LoRA)")
st.caption("Multi-turn chat • Avatar seperti contoh • Skenario uji dari dataset • Siap Streamlit Cloud")


try:
    tokenizer, model = load_model_and_tokenizer(BASE_MODEL, ADAPTER_DIR)
except Exception:
    st.error("❌ Crash saat load model/adapter. Detail error:")
    st.code(traceback.format_exc())
    st.stop()


# ================================
# SETTINGS (ENV OVERRIDE)
# ================================
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")  # gated
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "output_lora")                    # hasil fine-tuning

USER_AVATAR = os.getenv("USER_AVATAR", "👥")
BOT_AVATAR  = os.getenv("BOT_AVATAR", "🤖")

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Kamu adalah chatbot layanan informasi Mirota Kampus. "
    "Jawab sesuai informasi yang kamu ketahui dari data fine-tuning. "
    "Jika informasi bisa berbeda per cabang, jelaskan bahwa kebijakan bisa berbeda tiap cabang."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ================================
# SESSION STATE INIT
# ================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}
if "queued_user_text" not in st.session_state:
    st.session_state.queued_user_text = ""


# ================================
# TOKEN (SAFE: lokal tanpa secrets.toml tidak crash)
# ================================
def get_hf_token() -> str | None:
    # 1) Coba dari secrets (kalau file secrets.toml ada)
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok and str(tok).strip():
            return str(tok).strip()
    except Exception:
        pass

    # 2) Fallback env var
    tok_env = os.getenv("HF_TOKEN", None)
    if tok_env and str(tok_env).strip():
        return str(tok_env).strip()

    return None


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(base_model: str, adapter_dir: str):
    hf_token = get_hf_token()
    if hf_token is None:
        st.error(
            "HF_TOKEN belum diset.\n\n"
            "Lokal: set environment variable HF_TOKEN\n"
            "atau buat file .streamlit/secrets.toml berisi:\n"
            "HF_TOKEN=\"hf_xxx\""
        )
        st.stop()

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, token=hf_token)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, use_auth_token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Base model
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token, torch_dtype=DTYPE)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(base_model, use_auth_token=hf_token, torch_dtype=DTYPE)

    # Adapter
    if not os.path.exists(adapter_dir):
        st.error(
            f"Folder adapter '{adapter_dir}' tidak ditemukan.\n\n"
            "Pastikan hasil fine-tuning LoRA kamu ada, contoh:\n"
            "output_lora/adapter_config.json\n"
            "output_lora/adapter_model.safetensors"
        )
        st.stop()

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = True

    return tokenizer, model


def build_messages(system_prompt: str, history: list[dict], max_context_messages: int) -> list[dict]:
    # simpan konteks terakhir (tanpa system)
    history_trim = history[-max_context_messages:] if max_context_messages > 0 else history
    msgs = [{"role": "system", "content": system_prompt.strip()}]
    msgs.extend(history_trim)
    return msgs


@torch.inference_mode()
def generate_reply(
    tokenizer,
    model,
    messages: list[dict],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(DEVICE)

    gen_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-6),
        top_p=min(max(top_p, 1e-6), 1.0),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = gen_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# ================================
# SIDEBAR: SETTINGS + 5 SKENARIO UJI (DATASET KAMU)
# ================================
with st.sidebar:
    st.subheader("⚙️ Settings")

    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=140)

    max_context_messages = st.slider("Max context messages (history)", 2, 30, 12, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 512, 256, 32)

    deterministic = st.checkbox("Deterministic (no sampling)", value=False)
    if deterministic:
        do_sample = False
        temperature = 0.0
        top_p = 1.0
    else:
        do_sample = True
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.subheader("🧪 5 Skenario Pengujian (dataset Mirota)")

    scenarios = [
        ("S1 — Definisi Mirota", "Apa itu Mirota Kampus?"),
        ("S2 — Jam Operasional", "Mirota Kampus buka jam berapa?"),
        ("S3 — Print/Fotokopi", "Di Mirota Kampus bisa print atau fotokopi?"),
        ("S4 — Pembayaran QRIS", "Bisa bayar pakai QRIS?"),
        ("S5 — Keluhan stok", "Tadi saya ke Mirota tapi barang yang dicari nggak ada."),
    ]

    for title, prompt in scenarios:
        if st.button(title, use_container_width=True):
            # klik tombol -> prompt otomatis terkirim pada rerun berikutnya
            st.session_state.queued_user_text = prompt

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.queued_user_text = ""
            st.rerun()
    with colB:
        st.caption(f"DEVICE: `{DEVICE}`")

    st.caption(f"Base: `{BASE_MODEL}`")
    st.caption(f"Adapter: `{ADAPTER_DIR}`")


# ================================
# LOAD MODEL
# ================================
tokenizer, model = load_model_and_tokenizer(BASE_MODEL, ADAPTER_DIR)


# ================================
# RENDER CHAT HISTORY (AVATAR)
# ================================
for m in st.session_state.messages:
    avatar = USER_AVATAR if m["role"] == "user" else BOT_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])


# ================================
# INPUT (AUTO-SEND dari tombol skenario)
# ================================
user_text = st.chat_input("Tulis pesan…")

# kalau user klik skenario: auto kirim prompt walau chat_input kosong
if (user_text is None or user_text.strip() == "") and st.session_state.queued_user_text:
    user_text = st.session_state.queued_user_text
    st.session_state.queued_user_text = ""

if user_text is None or user_text.strip() == "":
    st.stop()


# ================================
# APPEND USER MESSAGE
# ================================
st.session_state.messages.append({"role": "user", "content": user_text})

with st.chat_message("user", avatar=USER_AVATAR):
    st.markdown(user_text)


# ================================
# GENERATE ASSISTANT
# ================================
with st.chat_message("assistant", avatar=BOT_AVATAR):
    with st.spinner("Generating..."):
        msgs = build_messages(system_prompt, st.session_state.messages, max_context_messages)
        reply = generate_reply(
            tokenizer=tokenizer,
            model=model,
            messages=msgs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        st.markdown(reply)

st.session_state.messages.append({"role": "assistant", "content": reply})
