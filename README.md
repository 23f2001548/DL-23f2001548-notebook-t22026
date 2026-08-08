# DL & GenAI Project — Smart MCQ Solver

**Name:** Gaurav Jha  
**Roll No:** 23f2001548  
**Term:** T2-2026  
**Course:** BSDA2001P — Introduction to DL and GenAI  

---

## Problem Statement

This project is built for the [Smart MCQ Solver Kaggle Competition](https://www.kaggle.com/competitions/smart-mcq-solver-challenge). Given a question prompt and five answer options (A to E), the model must rank the most likely correct answers. Evaluation metric is **mAP@3** (Mean Average Precision at 3).

The dataset is deliberately challenging — options are closely related concepts that require understanding precise definitions, not just broad topic knowledge.

---

## Results

| Model | Val mAP@3 | Val Accuracy | Kaggle Score |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9905 | 0.9811 | 0.74896 |
| DeBERTa-v3-base (fine-tuned) | 0.9981 | 0.9962 | **0.75644** |
| BiLSTM from Scratch | 0.9811 | 0.9697 | ~0.748 |
| Ensemble (0.2/0.6/0.2) | — | — | 0.751 |

**Best Kaggle Score: 0.75644** (cutoff: 0.73) ✅

---

## Models

### Model 1 — TF-IDF + Logistic Regression (Baseline)
Classical ML baseline. Prompt and each option are concatenated and converted to TF-IDF bigram vectors. A binary Logistic Regression classifier scores each option, and all five are ranked by probability.

### Model 2 — DeBERTa-v3-base (Fine-tuned Pretrained Transformer)
Fine-tuned using `AutoModelForMultipleChoice` from HuggingFace. DeBERTa's disentangled attention mechanism separately encodes content and position, giving it stronger contextual understanding than standard BERT. All five options are scored simultaneously in one forward pass.

### Model 3 — BiLSTM from Scratch (Custom DL / RNN)
Built entirely from scratch using PyTorch — no pretrained weights. A BiLSTMEncoder embeds words, passes through bidirectional LSTM (256 hidden units), and returns mean-pooled output. Prompt and each option are encoded independently, concatenated, and scored via a linear layer.

---

## Repository Structure

```
├── notebooks/
│   ├── DL-23f2001548-notebook-t22026.ipynb   # Main notebook — all 3 models + ensemble + submission
│   ├── milestone1.ipynb                       # NLP Foundation & Semantic Similarity
│   ├── milestone2.ipynb                       # Transformers & Pretrained Embeddings
│   ├── milestone3.ipynb                       # Context Augmentation with RAG
│   ├── milestone4.ipynb                       # MCQ Formatting & Fine-tuning
│   └── milestone5.ipynb                       # Ensembling & Final Predictions
├── reports/
│   └── 23f2001548_DG_T22026.pdf               # Final project report
├── app.py                                     # Streamlit deployment app
├── requirements.txt                           # Dependencies
└── README.md
```

---

## Experiment Tracking

All experiments tracked on **Weights & Biases**:  
Project: [DL-GenAi-2026-T2](https://wandb.ai/23f2001548-dl-genai-project/DL-GenAi-2026-T2)

---

## Deployment

**HuggingFace Hub:** [gauravjhaiitm/deberta-mcq-solver](https://huggingface.co/gauravjhaiitm/deberta-mcq-solver)  
**Streamlit App:** [deberta-mcq-solver.streamlit.app](https://deberta-mcq-solver.streamlit.app/)

---

## Key Hyperparameters

**DeBERTa:**
- Learning rate: 1e-5
- Batch size: 4 (effective 16 with gradient accumulation)
- Epochs: 6 with `load_best_model_at_end=True`
- Max sequence length: 256
- fp16: False (T4 GPU incompatibility)

**BiLSTM:**
- Embedding dim: 128, Hidden dim: 256 (bidirectional → 512)
- Dropout: 0.4, Epochs: 10
- Optimizer: Adam lr=1e-3, Grad clipping: 1.0
- Vocabulary size: 3804, Max length: 128

---

## Setup

```bash
pip install transformers==4.44.2 accelerate==0.34.2 tokenizers==0.19.1
pip install torch wandb scikit-learn pandas numpy
```

---

## Links

- Kaggle Notebook: [DL-23f2001548-notebook-t22026](https://www.kaggle.com/code/gauravjhaiitm/dl-23f2001548-notebook-t22026)
- HuggingFace Model: [gauravjhaiitm/deberta-mcq-solver](https://huggingface.co/gauravjhaiitm/deberta-mcq-solver)
- W&B Project: [DL-GenAi-2026-T2](https://wandb.ai/23f2001548-dl-genai-project/DL-GenAi-2026-T2)
- Streamlit App: [deberta-mcq-solver.streamlit.app](https://deberta-mcq-solver.streamlit.app/)
