# CLAUDE.md — MHMisinfo Dashboard Project
> This file is the persistent memory for Claude Code. Read it at the start of every session.

---

## 1. Project Overview

**Thesis title:** AI-Driven Visual Analytics for Real-Time Monitoring of Mental Health Misinformation on Social Media
**Unit:** PRT840 IT Thesis — Charles Darwin University (CDU)
**Research Group:** Group 2
**Supervisor:** Md Rafiqul Islam
**Student:** Paul Steven Flores Sinche (S386377)
**Teammates:** Niloy Saha (S388945), Md Asif Uz Zaman Talukder (S389003), MD Arafath Islam Abir (S381312)
**Target graduation:** November 2026

---

## 2. Research Questions

- **RQ1:** How do misinformation patterns differ between YouTube and Bitchute? *(statistical analysis)*
- **RQ2:** How effective are transformer models at detecting mental health misinformation? *(model implementation)*
- **RQ3:** How do interactive visualisations improve interpretation of that information? *(the dashboard — main focus here)*

---

## 3. Models

| Model | Base | F1 Score | Notes |
|-------|------|----------|-------|
| RoBERTa | roberta-base | 0.76 | Fine-tuned on MHMisinfo |
| MentalBERT | mental/mental-bert-base-uncased | 0.77 | Fine-tuned on MHMisinfo |

- Both models fine-tuned on the **MHMisinfo dataset** (Zenodo DOI: 10.5281/zenodo.13191247)
- Class imbalance handled via **50/50 undersampling**
- Training done in **Google Colab**
- Models saved to **Google Drive**
- Key finding: YouTube shows higher misinformation rate (11.6%) than Bitchute (5.5%)

---

## 4. Dataset & Files

```
Coding/
├── comments_MHMisinfo_Full.csv      # ~90 MB — full comments dataset
├── comments_MHMisinfo_Gold.csv      # ~21 MB — gold-labelled comments
├── videos_MHMisinfo_Full.csv        # ~6.7 MB — full videos dataset
├── videos_MHMisinfo_Gold.csv        # ~772 KB — gold-labelled videos
├── exploring_dataset.ipynb          # EDA notebook
├── thesis_roberta_mentalbert.ipynb  # Main model training notebook
├── misinfo_patterns.png             # Pre-generated chart
├── pattern_analysis.png             # Pre-generated chart
└── platform_comparison.png          # Pre-generated chart
```

Platforms in dataset: **YouTube** and **Bitchute**
Labels: binary — misinformation (1) vs. credible (0)

---

## 5. Dashboard — Current State

**Framework:** Streamlit
**Theme:** Dark UI (`#0a0e1a` background, `#00d4b4` accent teal, `#4f8fff` blue)
**Fonts:** Space Mono (headings) + Inter (body) via Google Fonts

### Dashboard sections already built:
- Page config & CSS theme (dark, card-based layout)
- Sidebar with filters
- RoBERTa vs MentalBERT live inference (text input → classification)
- Model performance comparison charts (F1, precision, recall)
- Platform comparison visualisations (YouTube vs Bitchute)
- Temporal trend analysis
- Confusion matrix display

### Main file: `dashboard_app.py`

### Key imports used:
```python
import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
from sklearn.metrics import confusion_matrix
```

---

## 6. Design System

```css
--bg-primary:   #0a0e1a
--bg-secondary: #111827
--bg-card:      #1a2235
--accent:       #00d4b4   /* teal */
--accent-2:     #4f8fff   /* blue */
--danger:       #ff4b6e
--warning:      #f59e0b
--text-primary: #e2e8f0
--text-muted:   #64748b
--border:       #1e293b
```

---

## 7. Model Loading (path config)

Models are loaded from Google Drive via mounted paths in Colab, or locally when running on laptop. The dashboard has a path config section:

```python
ROBERTA_PATH    = '...'   # path to saved RoBERTa model
MENTALBERT_PATH = '...'   # path to saved MentalBERT model
RESULTS_PATH    = '...'   # path to evaluation results
```

> When running locally on Windows, update these paths to wherever the models are saved.

---

## 8. How to Run

```bash
# Install dependencies
pip install streamlit torch transformers pandas numpy plotly scikit-learn

# Run dashboard
streamlit run dashboard_app.py
```

---

## 9. Context About the Developer

- **Name:** Paul (S386377)
- **Background:** Communications, not CS — prefers plain-language explanations before technical content
- **Environment:** Windows 11 (Lenovo Legion 7), PowerShell, Python via pip
- **IDE preference:** Working via Claude Code in terminal + VS Code
- **Colab used for training** (GPU access), local machine for dashboard
- **Language:** Communicates in Spanish; professional outputs in English

---

## 10. Supervisor Feedback (May 2026)

### Research Report
- Remove RQ1/RQ2 labels from research questions
- Exclude introduction section from final report
- Focus more on model sections, less on dataset explanation
- Figures have inconsistent results — left vs right diagrams show different outcomes for same models
- MentalBERT appears more accurate than RoBERTa in figures (needs correction/clarification)

### Dashboard Requirements (UPDATED)
- **Replace dual model display with single model selector** — dropdown to choose RoBERTa OR MentalBERT
- Dashboard visualizations must update based on selected model
- Showing both models simultaneously doesn't scale — remove that approach
- Live classifier must work with both models via the selector
- **Migrate from Google Colab to permanent hosting** — Streamlit Community Cloud preferred (lifetime hosting, CV portfolio value)

### Current Dashboard Status
- Live classifier functional with RoBERTa and MentalBERT
- Features: live inference, model performance metrics, pattern analysis, platform comparison
- Currently runs on Google Colab (requires manual execution for links)
- Needs migration to Streamlit Community Cloud

### Timeline
- **Week 10:** Complete dashboard + make it live
- Send dashboard to another student group for testing (4-5 usability questions)
- **Week 12:** Public release target
- Optional: IT Code Fair (November 5th, submission by September)

### Evaluation Plan
- Send to another student group for testing
- 4-5 evaluation questions about usability and accuracy
- Calculate success rate from ratings (e.g. 4/5 = 80% success)

---

## 11. What to Do When Paul Opens a Session

1. Read this file first
2. Check which files exist in the `Coding/` folder
3. Ask what he wants to work on today (dashboard feature, bug fix, new section, etc.)
4. Keep explanations clear — explain *what* and *why* before writing code
5. Always run `streamlit run dashboard_app.py` to verify changes work
