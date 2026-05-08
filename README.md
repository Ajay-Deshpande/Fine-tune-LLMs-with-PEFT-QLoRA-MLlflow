# Fine-Tuning LLMs with QLoRA, PEFT & MLflow

Parameter-efficient fine-tuning of large language models using QLoRA (4-bit quantization) and PEFT, with MLflow for experiment tracking — enabling high-quality model adaptation on consumer-grade GPU hardware.

---

## 🎯 The Problem This Solves

Full fine-tuning of LLMs requires tens of gigabytes of VRAM and hours of compute — inaccessible without expensive infrastructure. This project demonstrates how to fine-tune large transformer models to near full-performance quality using a fraction of the memory, making LLM customization practical at scale.

---

## 🧠 Key Techniques

### QLoRA — Quantized Low-Rank Adaptation
- Base model loaded in **4-bit precision** (NF4 quantization via BitsandBytes), dramatically reducing VRAM footprint
- LoRA adapters trained in **16-bit precision** on top of the frozen quantized base — preserving training quality while keeping memory usage low
- Result: fine-tune a 7B+ parameter model on a single GPU (P100) that would otherwise require multi-GPU setup

### PEFT — Parameter-Efficient Fine-Tuning
Three adapter strategies explored:
- **LoRA (Low-Rank Adaptation)** — injects trainable low-rank matrices into attention layers; only ~0.1–1% of parameters are trained
- **Adapter Layers** — inserts small bottleneck modules between transformer blocks
- **Prompt Tuning** — learns soft prompt embeddings prepended to input, leaving model weights untouched

### MLflow — Experiment Tracking
- Logs hyperparameters (rank, alpha, dropout, learning rate, batch size) per run
- Tracks training/validation loss curves and evaluation metrics
- Stores fine-tuned checkpoints with versioning for reproducible comparison

---

## ⚙️ Pipeline

```
Pre-trained Base Model (HuggingFace)
        │
        ▼
4-bit Quantization (BitsandBytes NF4)
        │
        ▼
PEFT Configuration (LoRA rank, alpha, target modules)
        │
        ▼
Fine-Tuning on Task-Specific Dataset
        │
        ├──▶ MLflow: log hyperparams, metrics, checkpoints
        │
        ▼
Evaluation → compare adapter variants
        │
        ▼
Merged Model (optional: merge LoRA weights into base)
```

---

## 📊 Why This Matters for Production

| Approach | VRAM Required | Training Parameters |
|---|---|---|
| Full Fine-Tuning (7B model) | ~112 GB | 7 billion |
| QLoRA (this project) | ~8 GB | ~4–40 million |
| Prompt Tuning | ~6 GB | ~10 thousand |

QLoRA achieves comparable task performance to full fine-tuning at ~7% of the memory cost.

---

## 🛠 Stack

- **HuggingFace Transformers** — base model loading and training loop
- **PEFT** — LoRA, adapter, and prompt tuning implementations
- **BitsandBytes** — 4-bit and 8-bit quantization
- **MLflow** — experiment tracking and model registry
- **PyTorch** — training framework
- **Kaggle GPU (P100)** — execution environment

---

## 🚀 Getting Started

**On Kaggle (recommended — no setup required):**
1. Open the notebook in Kaggle
2. Set Accelerator → GPU (P100)
3. Run all cells

**Locally:**
```bash
pip install transformers peft bitsandbytes mlflow torch
jupyter notebook finetuning_Qlora_mlflow.ipynb
```

---

## 📌 Related Projects

- [LLM-RAG-sandbox](https://github.com/Ajay-Deshpande/LLM-RAG-sandbox) — RAG architectures and retrieval optimization
