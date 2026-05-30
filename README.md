# MHMisinfo Dashboard

**AI-Driven Visual Analytics for Real-Time Monitoring of Mental Health Misinformation on Social Media**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mhmisinfo-dashboard.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/Paulst7)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Project Description

This project detects and visualises mental health misinformation in social media comments using fine-tuned transformer models (RoBERTa and MentalBERT). It was developed as part of a Master of Information Technology thesis at Charles Darwin University (CDU), Group 2, 2026.

The system addresses two key gaps identified in the literature:
- No open-source, comment-level misinformation detection tool exists for mental health content.
- No interactive monitoring interface allows researchers or platform moderators to explore model outputs and platform-level trends in real time.

**Dataset:** MHMisinfo Full — 582,362 comments from YouTube and Bitchute (Nguyen et al., 2024). DOI: [10.5281/zenodo.13191247](https://doi.org/10.5281/zenodo.13191247)

**Live Dashboard:** [https://mhmisinfo-dashboard.streamlit.app](https://mhmisinfo-dashboard.streamlit.app)

### Research Questions

- **RQ1:** How do misinformation patterns differ between YouTube and Bitchute at the comment level?
- **RQ2:** How accurately can fine-tuned RoBERTa and MentalBERT detect misinformation in short, colloquial comment text?
- **RQ3:** How does an interactive visual analytics dashboard improve interpretability and real-time monitoring?

### Key Results

| Model | Accuracy | Misinformation F1 |
|-------|----------|-------------------|
| RoBERTa | 75% | 0.76 |
| MentalBERT | 77% | 0.77 |

---

## Environment Setup

### Option A — pip (recommended for quick start)

```bash
# Clone the repository
git clone https://github.com/Paul-Flores-Sinche/mhmisinfo-dashboard.git
cd mhmisinfo-dashboard

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B — conda

```bash
# Create environment
conda create -n mhmisinfo python=3.12
conda activate mhmisinfo

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Key packages (see `requirements.txt` for full list):

```
streamlit
torch
transformers>=4.x
pandas
scikit-learn
matplotlib
plotly
huggingface_hub
```

**Training environment:** Google Colab with NVIDIA Tesla T4 GPU (15.6 GB VRAM), Python 3.12, PyTorch 2.x, HuggingFace Transformers 5.3.0, pandas 3.0.1, scikit-learn 1.8.0, matplotlib 3.x.

---

## Running the Dashboard Locally

### 1. Set up your HuggingFace token

The dashboard loads models from HuggingFace Hub. You need a free HuggingFace account and access token.

Create a file called `.streamlit/secrets.toml` in the project root:

```toml
HF_TOKEN = "your_huggingface_token_here"
```

> **Note:** Never commit this file to GitHub. It is already listed in `.gitignore`.

### 2. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| Overview | Dataset summary, class distribution, temporal trends, platform comparison |
| Model Performance | Metrics (accuracy, precision, recall, F1), confusion matrices, training loss curves |
| Pattern Analysis | Misinformation category breakdown, keyword frequency, platform rates |
| Live Classifier | Real-time comment classification with confidence scores and history |

---

## Training Parameter Settings

Models were fine-tuned on a balanced 10,000-comment subset (5,000 misinformation / 5,000 legitimate) drawn from the MHMisinfo Full dataset. Training was performed in Google Colab.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Platform | Google Colab | Free GPU access, reproducible environment |
| GPU | NVIDIA Tesla T4 (15.6 GB) | Sufficient VRAM for batch size 32 |
| Optimiser | AdamW | Standard for transformer fine-tuning |
| Learning rate | 2e-5 | Recommended for BERT-family models (Devlin et al., 2019) |
| Epochs | 3 | Loss decreased each epoch; convergence confirmed |
| Batch size | 32 | Balance between memory and training stability |
| Max sequence length | 128 tokens | Covers most comment lengths; longer sequences truncated |
| Train/test split | 80/20 stratified | Preserves class balance across splits |

### Training Loss per Epoch

| Epoch | RoBERTa Loss | MentalBERT Loss |
|-------|-------------|-----------------|
| 1 / 3 | 0.5906 | 0.5426 |
| 2 / 3 | 0.4741 | 0.3783 |
| 3 / 3 | 0.3796 | 0.2149 |

### Class Imbalance Handling

The original dataset contained 88.6% legitimate comments and 11.4% misinformation (≈8:1 ratio). Without balancing, the model achieved 95% accuracy but F1=0.00 for the misinformation class (accuracy paradox). Undersampling was applied to create a balanced 50/50 training set.

---

## Loading Models from HuggingFace

Both fine-tuned models are publicly available on HuggingFace Hub.

| Model | HuggingFace ID | Macro F1 |
|-------|----------------|----------|
| RoBERTa | [Paulst7/roberta-mhmisinfo](https://huggingface.co/Paulst7/roberta-mhmisinfo) | 0.76 |
| MentalBERT | [Paulst7/mentalbert-mhmisinfo](https://huggingface.co/Paulst7/mentalbert-mhmisinfo) | 0.77 |

### Load with HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load MentalBERT (or swap with roberta-mhmisinfo)
model_id = "Paulst7/mentalbert-mhmisinfo"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

# Classify a comment
comment = "Doctors are hiding the cure for depression."

inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits

probabilities = torch.softmax(logits, dim=-1)
predicted_class = torch.argmax(probabilities).item()

labels = {0: "Legitimate", 1: "Misinformation"}
print(f"Label: {labels[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class].item():.1%}")
```

### Load with HuggingFace Pipeline (simpler)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Paulst7/mentalbert-mhmisinfo",
    tokenizer="Paulst7/mentalbert-mhmisinfo"
)

result = classifier("Doctors are hiding the cure for depression.")
print(result)
# [{'label': 'LABEL_1', 'score': 0.85}]
```

> If the models are private, pass your token: `pipeline(..., token="your_hf_token")`

---

## Repository Structure

```
mhmisinfo-dashboard/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # HuggingFace token (not committed)
└── README.md
```

---

## Citation

If you use this work, please cite:

```
Flores Sinche, P.S., Saha, N., Talukder, M.A.U.Z., & Islam Abir, M.A. (2026).
AI-Driven Visual Analytics for Real-Time Monitoring of Mental Health Misinformation
on Social Media. Master of Information Technology Thesis, Charles Darwin University.
```

**Dataset:**
```
Nguyen, V.C. (2024). MHMisinfo — Video-based Mental Health Misinformation Dataset (Version 1).
Zenodo. https://doi.org/10.5281/zenodo.13191247
```

---

## Authors

- Paul Steven Flores Sinche (S386377)
- Niloy Saha (S388945)
- Md Asif Uz Zaman Talukder (S389003)
- MD Arafath Islam Abir (S381312)

**Supervisor:** Md Rafiqul Islam  
**Institution:** Charles Darwin University, Faculty of Science and Technology  
**Unit:** PRT840 — Master of Information Technology Thesis, 2026
