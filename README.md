---
# 🚀 main.py

(imports and main logic as above)

---
# 📄 README.md

```markdown
# 🧬 Protein Language Model Fine-Tuning Pipeline

This project provides a full pipeline to fine-tune protein language models like **ProtGPT2**, **ProGen**, and **RITA** using your own FASTA data. It supports preprocessing, hyperparameter search with Optuna, LoRA fine-tuning, evaluation, and generation.

---

## 📦 Models Supported
- `protgpt2`
- `progen2-small`, `progen2-medium`, `progen2-large`, `progen2-xlarge`
- `RITA_s`, `RITA_m`, `RITA_l`, `RITA_xl`

---

## 🧪 Requirements
```bash
pip install -r requirements.txt
```

**Or manually:**
```bash
pip install transformers datasets peft optuna torch
```

---

## 🚀 Run Full Pipeline
```bash
python main.py \
  --model protgpt2 \
  --fasta data/sequences.fasta \
  --output_dir ./outputs/protgpt2 \
  --n_trials 5 \
  --generate_prompt "MGLSA"
```

This will:
1. Load the selected model
2. Preprocess the FASTA file
3. Run Optuna for hyperparameter search
4. Fine-tune using LoRA
5. Generate a new sequence from prompt (optional)

---

## 🔍 Evaluate the Model
```bash
python evaluate.py \
  --model_dir ./outputs/protgpt2/final_model \
  --test_file ./outputs/protgpt2/test.txt
```

---

## 🧬 Generate New Sequence
```bash
python generate.py \
  --model protgpt2 \
  --model_dir ./outputs/protgpt2/final_model \
  --prompt "MAKVQ" \
  --save_path ./results/generated.txt
```

---

## 📁 Folder Structure
```
project/
│
├── main.py
├── model_loader.py
├── preprocess.py
├── trainer.py
├── optuna_search.py
├── evaluate.py
├── generate.py
├── data/
│   └── sequences.fasta
├── outputs/
│   └── [model_name]/...
└── results/
    └── generated.txt
```
