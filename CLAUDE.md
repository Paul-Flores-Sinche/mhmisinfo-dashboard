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

| Model | Base | F1 Score | HuggingFace Hub |
|-------|------|----------|-----------------|
| RoBERTa | roberta-base | 0.76 | `Paulst7/roberta-mhmisinfo` |
| MentalBERT | mental/mental-bert-base-uncased | 0.77 | `Paulst7/mentalbert-mhmisinfo` |

- Both models fine-tuned on the **MHMisinfo dataset** (Zenodo DOI: 10.5281/zenodo.13191247)
- Class imbalance handled via **50/50 undersampling**
- Training done in **Google Colab**
- Models published to **HuggingFace Hub** (loaded via `from_pretrained()` at runtime)
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

**Status: LIVE** at https://mhmisinfo-dashboard.streamlit.app
**Hosting:** Streamlit Community Cloud (permanent, free)
**Framework:** Streamlit
**Theme:** Vigil — deep-black dark UI (`#09090E` background, `#00D4A8` teal accent, `#8B6CF8` purple)
**Fonts:** IBM Plex Mono (monospace) via Google Fonts

### Tabs built:
- **Overview** — KPI cards, platform comparison bar charts, comment/video label pie charts
- **Model Performance** — per-class metrics, confusion matrix, training loss curve, side-by-side comparison
- **Pattern Analysis** — misinformation category breakdown, temporal trend, sample comments
- **Live Classifier** — real inference via sidebar model selector (RoBERTa or MentalBERT); quick example buttons (2 misinfo + 2 legit from real dataset, auto-fill + auto-classify on click); classification history table (comment, label, confidence %, model; persists per session; Clear button)

### Architecture (`dashboard_app.py`):
- Each tab is a standalone function: `render_overview(fc, fv)`, `render_model_tab(selected_model, m)`, `render_pattern_tab(fc)`, `render_classifier_tab(selected_model, m, ex_comments)`
- HTML helpers: `vkpi()`, `alert_bar()`, `vfinding()`, `legend_row()`, `prob_bar()`, `clean_text()`
- CSS component classes: `.vkpi`, `.vcard`, `.valert`, `.vfind`, `.vbadge`, `.vcls-result`, `.vcls-ms`, `.vpbar`, `.vcm-ok`, `.vcm-ng`
- Model loading uses `@st.cache_resource` — models load once and stay in memory
- Data loading uses `@st.cache_data` — falls back through: full CSV → sample CSV → synthetic data
- Classifier state keys in `st.session_state`: `clf_history`, `auto_classify`, `classifier_input`, `last_result`

### Key imports used:
```python
import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, re
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
)
```

---

## 6. Design System (Vigil Theme — applied 2026-05-14)

### Color palette
```css
--bg:           #09090E   /* near-black page background */
--surface:      #0E0E16   /* card/panel surface */
--border:       #1C1C2E   /* subtle border */
--teal:         #00D4A8   /* primary accent — correct, credible */
--purple:       #8B6CF8   /* secondary accent — model/AI elements */
--pink:         #FF6B9D   /* danger / misinformation */
--orange:       #FF9040   /* warning */
--text:         #E8E8F0   /* primary text */
--text-muted:   #6B6B8A   /* secondary text */
```

### CSS component classes
| Class | Purpose |
|-------|---------|
| `.vkpi` | KPI stat card with teal top-border |
| `.vcard` | Generic content card (surface bg) |
| `.valert` | Alert/finding bar with left accent strip |
| `.vfind` | Key finding callout block |
| `.vbadge` | Small label badge (color via inline style) |
| `.vcls-result` | Classifier result card |
| `.vcls-ms` | Mini metric strip inside result card |
| `.vpbar` | Probability bar (misinfo vs legit) |
| `.vcm-ok` | Confusion matrix correct-prediction cell (teal) |
| `.vcm-ng` | Confusion matrix error-prediction cell (pink) |

---

## 7. Model Loading

Models are loaded from HuggingFace Hub at runtime using `@st.cache_resource`:

```python
MODEL_HUB_IDS = {
    "RoBERTa":    "Paulst7/roberta-mhmisinfo",
    "MentalBERT": "Paulst7/mentalbert-mhmisinfo",
}
```

- Auth token read from environment variable `HF_TOKEN`
- On Streamlit Cloud: set `HF_TOKEN` under app Settings → Secrets
- Locally: `$env:HF_TOKEN = "hf_xxx"` before running
- If the Hub repo is public, `HF_TOKEN` is not required

---

## 8. How to Run

```powershell
# Install dependencies
pip install streamlit torch transformers pandas numpy plotly scikit-learn

# Set HuggingFace token (if repo is private)
$env:HF_TOKEN = "hf_xxxxxxxxxxxx"

# Run dashboard (use python -m streamlit — streamlit not in PATH on this machine)
python -m streamlit run dashboard_app.py
```

> Note: `streamlit` is not in the Windows PATH. Always use `python -m streamlit run`.
> A `~/.streamlit/credentials.toml` and `~/.streamlit/config.toml` are configured to skip first-run prompts.

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

### Current Dashboard Status ✓ COMPLETE — ready for evaluation
- **Live at:** https://mhmisinfo-dashboard.streamlit.app
- Models: `Paulst7/roberta-mhmisinfo` and `Paulst7/mentalbert-mhmisinfo` (HuggingFace Hub)
- All four tabs complete: Overview, Model Performance, Pattern Analysis, Live Classifier
- Live Classifier fully featured: real inference, quick example buttons, classification history table
- Hosted on Streamlit Community Cloud (permanent, no manual execution needed)

### Timeline
- **Week 10:** ✓ Dashboard complete and live
- Send dashboard to another student group for testing (4-5 usability questions)
- **Week 12:** Public release target
- Optional: IT Code Fair (November 5th, submission by September)

### Evaluation Plan
- Send to another student group for testing
- 4-5 evaluation questions about usability and accuracy
- Calculate success rate from ratings (e.g. 4/5 = 80% success)

---

## 11. Changelog

### 2026-05-14 — Vigil redesign + 6 bug fixes

**Vigil theme applied** (full visual redesign of `dashboard_app.py`):
- New deep-black colour palette replacing the old purple-navy scheme
- IBM Plex Mono font replacing Inter
- Custom CSS component system (`.vkpi`, `.vcard`, `.valert`, etc. — see Section 6)
- Helper functions renamed/replaced: `vkpi()`, `alert_bar()`, `vfinding()`, `legend_row()`, `prob_bar()`
- Tab function signatures updated to include filtered dataframes and `ex_comments`

**Bug fixes (commit `9749071`):**

| # | Description | Fix |
|---|-------------|-----|
| #1 | Result card vanished on any Streamlit rerun | Stored result in `st.session_state.last_result`; rendered from state |
| #3 | Removing all platforms from filter showed empty charts silently | Added `st.warning` + `st.stop()` guard after filter computation |
| #5 | Confusion matrix had no annotation on diagonal cells | Added TPR / TNR percentages below each diagonal cell |
| #6 | Platform bar widths were hardcoded (`rate * 5`) — broke if data changed | Now data-driven: `rate / max_rate * 90` |
| #8 | **Crash**: `render_classifier_tab` call site passed 2 args; signature required 3 | Call site updated to pass `comments_df` as third arg |
| #9 | `clean_text(None)` raised `AttributeError` on NaN comment values | Added `if text is None: return ""` guard at top of function |

---

## 12. What to Do When Paul Opens a Session

1. Read this file first
2. Check which files exist in the `Coding/` folder
3. Ask what he wants to work on today (dashboard feature, bug fix, report writing, evaluation, etc.)
4. Keep explanations clear — explain *what* and *why* before writing code
5. To test locally: `python -m streamlit run dashboard_app.py` (not `streamlit run`)
6. Dashboard is already live — next priorities are usability evaluation and thesis report
