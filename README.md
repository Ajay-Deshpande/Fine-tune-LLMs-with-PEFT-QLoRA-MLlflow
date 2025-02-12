# 🚀 Fine-Tuning Transformers with QLoRA, PEFT & MLflow  

A **Parameter-Efficient Fine-Tuning (PEFT)** implementation using **QLoRA**, **BitsandBytes Quantization**, and **MLflow** to optimize transformer models efficiently. This project was executed on **Kaggle** using a **GPU P100 session**, requiring minimal setup.  

---

## 📌 Features  
- ✅ **QLoRA (Quantized LoRA)** for fine-tuning large models with low VRAM  
- ✅ **BitsandBytes Quantization (4-bit & 8-bit)** to reduce GPU memory usage  
- ✅ **PEFT (LoRA, Adapters, Prompt Tuning)** for efficient fine-tuning  
- ✅ **MLflow Integration** for experiment tracking and model versioning  
- ✅ **Hugging Face Transformers** for NLP fine-tuning  

---

## 🛠️ Tech Stack  
- **Python**  
- **Hugging Face Transformers**  
- **QLoRA** (Quantized LoRA for efficient fine-tuning)  
- **BitsandBytes** (Quantization for memory-efficient training)  
- **MLflow** (Experiment Tracking & Model Management)  
- **PEFT (LoRA, Adapters, Prompt Tuning)**  
- **PyTorch**  

---

## 🚀 Running on Kaggle  

### **1️⃣ Open Kaggle Notebook**  
[Click here to open in Kaggle](#)  

### **2️⃣ Change Runtime to GPU P100**  
- Navigate to **Settings** in Kaggle Notebook.  
- Under **Accelerator**, select **GPU (P100)**.  

### **3️⃣ Install Dependencies**  
```bash
pip install transformers peft bitsandbytes mlflow
```

🔍 How It Works
- Fine-Tuning with QLoRA & PEFT
  - QLoRA (Quantized LoRA) allows fine-tuning large models with minimal VRAM.
  - BitsandBytes Quantization (4-bit & 8-bit) significantly reduces memory footprint.
  - LoRA, Adapters, and Prompt Tuning ensure efficient model adaptation.
- MLflow for Experiment Tracking
  - Logs hyperparameters, model metrics, and fine-tuned checkpoints.
  - Enables easy model versioning and comparison.
 
✅ **Kaggle Execution** → No elaborate setup required, just switch to **GPU P100**.  
✅ **Minimal Installation** → Only requires running a few `pip install` commands.  
