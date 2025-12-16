import os
import traceback
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Chatbot Mirota Kampus (Fine-tuned LoRA)",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chatbot Mirota Kampus (Fine-tuned LoRA)")
st.caption("Multi-turn chat • Avatar • Skenario uji dari dataset • Streamlit Cloud")


# ================================
# SETTINGS
# ================================
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")  # gated
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "output_lora")

USER_AVATAR = os.getenv("USER_AVATAR", "👤")
BOT_AVATAR  = os.getenv("BOT_AVATAR", "🤖")

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Kamu adalah chatbot layanan informasi Mirota Kampus. "
    "Jawab dengan jelas, sopan, dan singkat (maks 2–3 kalimat). "
    "Jika informasi bisa berbeda antar cabang, jelaskan bahwa kebijakan dapat berbeda antar cabang."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "auto"  # hemat RAM & aman di Streamlit Cloud


# ================================
# SESSION STATE INIT
# ================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queued_user_text" not in st.session_state:
    st.session_state.queued_user_text = ""


# ================================
# TOKEN LOADER (SAFE)
# ================================
def get_hf_token() -> str | None:
    try:
        tok = st.secrets.get("HF_TOKEN", None)
        if tok and str(tok).strip():
            return str(tok).strip()
    except Exception:
        pass

    tok_env = os.getenv("HF_TOKEN", None)
    if tok_env and str(tok_env).strip():
        return str(tok_env).strip()

    return None


# ================================
# MODEL LOADER (CACHED)
# ================================
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(base_model: str, adapter_dir: str):
    hf_token = get_hf_token()
    if hf_token is None:
        raise RuntimeError("HF_TOKEN belum diset (Secrets atau env var).")

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, token=hf_token)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, use_auth_token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Base model (hemat RAM)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            use_auth_token=hf_token,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )

    # Adapter
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Folder adapter '{adapter_dir}' tidak ditemukan di repo.")

    if len(os.listdir(adapter_dir)) == 0:
        raise FileNotFoundError(f"Folder '{adapter_dir}' kosong. Upload adapter LoRA ke repo.")

    model = PeftModel.from_pretrained(base, adapter_dir)

    model.to(DEVICE)
    model.eval()
    model.config.use_cache = True

    # Kadang membantu mengurangi lag di environment kecil
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    return tokenizer, model


def build_messages(system_prompt: str, history: list[dict], max_context_messages: int) -> list[dict]:
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
    top_p: float
) -> str:
    # Pakai apply_chat_template -> text -> tokenize, supaya dapat attention_mask
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # ✅ hilangkan warning & lebih reliable
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-6),
        top_p=min(max(top_p, 1e-6), 1.0),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = gen_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.subheader("⚙️ Settings")

    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=140)

    max_context_messages = st.slider("Max context messages (history)", 2, 30, 12, 1)

    # ✅ default diperkecil biar lebih cepat
    max_new_tokens = st.slider("Max new tokens", 32, 256, 128, 16)

    deterministic = st.checkbox("Deterministic (no sampling)", value=True)
    if deterministic:
        do_sample = False
        temperature = 0.0
        top_p = 1.0
    else:
        do_sample = True
        temperature = st.slider("Temperature", 0.1, 1.5, 0.5, 0.1)
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
    st.caption(f"dtype: `{DTYPE}`")


# ================================
# LOAD MODEL (CLEAR ERROR)
# ================================
try:
    tokenizer, model = load_model_and_tokenizer(BASE_MODEL, ADAPTER_DIR)
except Exception:
    st.error("❌ Gagal load model/adapter. Detail error:")
    st.code(traceback.format_exc())
    st.stop()


# ================================
# RENDER HISTORY (AVATAR)
# ================================
for m in st.session_state.messages:
    avatar = USER_AVATAR if m["role"] == "user" else BOT_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])


# ================================
# INPUT (AUTO SEND FROM SCENARIO)
# ================================
user_text = st.chat_input("Tulis pesan…")

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
        reply = generate_reply(tokenizer, model, msgs, max_new_tokens, do_sample, temperature, top_p)
        st.markdown(reply)

st.session_state.messages.append({"role": "assistant", "content": reply})
