# Chatbot Mirota Kampus (Fine-tuned LoRA)

Chatbot Streamlit yang memuat **base LLM (gated)** + **LoRA adapter hasil fine-tuning**, lalu menjalankan **multi-turn chat** dengan pengaturan generation (max history, max tokens, temperature, top-p) dan tombol **5 skenario pengujian** dari dataset.

---

## Ringkasan Sistem

### Model & Pipeline
| Komponen | Nilai |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` (gated) |
| Adapter | LoRA via **PEFT** (folder `output_lora/`) |
| Task | Chat / text-generation |
| Framework | `transformers` + `peft` + `torch` |
| App | Streamlit |

### Fitur Utama di App
- **Load gated model** menggunakan `HF_TOKEN` (Streamlit Secrets / Environment Variable).
- **Attach LoRA adapter** dari folder `output_lora/`.
- **Multi-turn chat history** (pesan sebelumnya dipakai sebagai konteks).
- **Pengaturan generation** via sidebar:
  - Max context messages (history)
  - Max new tokens
  - Deterministic (no sampling)
  - Temperature
  - Top-p
- **5 skenario pengujian** (klik → auto kirim prompt) khusus dataset Mirota.
- UI chat dengan **avatar** user & bot.

---