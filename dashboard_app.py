import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import re
import torch
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MHMisinfo · Vigil",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS — Vigil dark theme
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
*, *::before, *::after {
    font-family: 'Outfit', system-ui, sans-serif !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #09090E !important;
    color: #CDD6F4 !important;
}
[data-testid="stMainBlockContainer"] {
    background-color: #09090E !important;
    padding-top: 1rem !important;
}
[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0B0D17 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] * { color: #CDD6F4 !important; }
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 12px 0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: .05em !important;
    color: #8892B0 !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}
section[data-testid="stSidebar"] small { color: #3D4A6B !important; }

/* Sidebar selectbox */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #111420 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #CDD6F4 !important; }
[data-baseweb="popover"] [role="option"] { background-color: #111420 !important; }
[data-baseweb="popover"] [role="option"]:hover { background-color: rgba(0,212,168,0.1) !important; }

/* Sidebar multiselect */
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: rgba(0,212,168,0.12) !important;
    border: 1px solid rgba(0,212,168,0.3) !important;
    border-radius: 4px !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 400 !important;
    color: #8892B0 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 16px !important;
    letter-spacing: 0 !important;
}
[data-baseweb="tab"]:hover {
    color: #CDD6F4 !important;
    background: rgba(255,255,255,0.03) !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #CDD6F4 !important;
    font-weight: 500 !important;
    background: transparent !important;
    border-bottom: 2px solid #00D4A8 !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5 {
    color: #CDD6F4 !important;
    font-weight: 600 !important;
}
p, li, label, .stMarkdown { color: #CDD6F4 !important; }
small { color: #8892B0 !important; }
code {
    background: rgba(0,212,168,0.1) !important;
    color: #00D4A8 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── st.metric ── */
[data-testid="metric-container"] {
    background: #111420 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: none !important;
}
[data-testid="metric-container"] label {
    color: #8892B0 !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #CDD6F4 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricDeltaIcon-Up"]   path { fill: #00D4A8 !important; }
[data-testid="stMetricDeltaIcon-Down"] path { fill: #FF9040 !important; }

/* ── Inputs ── */
textarea, [data-baseweb="textarea"] textarea {
    background-color: #080A12 !important;
    color: #CDD6F4 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-family: 'Outfit', sans-serif !important;
}
textarea:focus { border-color: rgba(0,212,168,0.3) !important; outline: none !important; }

/* ── Buttons ── */
[data-testid="baseButton-primary"] {
    background: #00D4A8 !important;
    color: #000000 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    padding: 7px 18px !important;
}
[data-testid="baseButton-primary"]:hover { opacity: .88 !important; }
[data-testid="baseButton-primary"]:disabled {
    background: rgba(255,255,255,0.07) !important;
    color: #8892B0 !important;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #8892B0 !important;
    border-radius: 6px !important;
    font-size: 11px !important;
}

/* ── st.info / st.warning ── */
[data-testid="stAlert"] {
    background: rgba(0,212,168,0.06) !important;
    border: 1px solid rgba(0,212,168,0.2) !important;
    border-radius: 8px !important;
    color: #CDD6F4 !important;
}
[data-testid="stAlert"] svg { display: none !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe {
    background: #111420 !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }

/* ══════════════════════════════════════════
   VIGIL COMPONENTS
══════════════════════════════════════════ */

/* KPI Cards */
.vkpi {
    background: #111420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
}
.vkpi-lbl {
    font-size: 10px;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #8892B0;
    margin-bottom: 10px;
}
.vkpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    line-height: 1;
    color: #CDD6F4;
}
.vkpi-delta { font-size: 10px; margin-top: 6px; font-family: 'IBM Plex Mono', monospace; }
.vkpi-sub   { font-size: 10px; color: #3D4A6B; margin-top: 8px; }

/* Chart card */
.vcard {
    background: #111420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.vcard-title { font-size: 13px; font-weight: 600; color: #CDD6F4; margin-bottom: 2px; }
.vcard-sub   { font-size: 10px; color: #8892B0; margin-bottom: 14px; }

/* Alert bar */
.valert {
    background: rgba(255,144,64,0.08);
    border: 1px solid rgba(255,144,64,0.25);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 12px 0 20px;
    font-size: 12px;
    color: #CDD6F4;
}

/* Finding cards */
.vfind {
    background: #111420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 12px 14px;
    border-left-width: 3px;
    border-left-style: solid;
    margin-bottom: 10px;
}
.vfind-lbl { font-size: 10px; color: #8892B0; margin-bottom: 3px; }
.vfind-val { font-size: 14px; font-weight: 600; }
.vfind-sub { font-size: 9px; color: #3D4A6B; margin-top: 2px; }

/* Model comparison table */
.vmdl {
    background: #111420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 18px;
}
.vtable {
    width: 100%;
    font-size: 12px;
    color: #CDD6F4;
    border-collapse: collapse;
    line-height: 2;
}
.vtable-hdr {
    color: #8892B0;
    font-size: 9px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    text-transform: uppercase;
    letter-spacing: .08em;
}

/* Classifier result */
.vcls-result {
    background: rgba(0,212,168,0.08);
    border: 1px solid rgba(0,212,168,0.25);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}
.vcls-ms {
    background: rgba(255,85,85,0.08);
    border: 1px solid rgba(255,85,85,0.25);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
}
.vcls-label { font-size: 16px; font-weight: 700; margin-bottom: 4px; }
.vcls-conf  { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 800; line-height: 1; }
.vcls-sub   { font-size: 10px; color: #8892B0; margin-top: 4px; }

/* Prob bars */
.vpbar {
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px;
    margin: 4px 0 8px;
    overflow: hidden;
}
.vpbar-fill { height: 100%; border-radius: 2px; }

/* Comment cards */
.vcomment-ms {
    background: rgba(255,107,157,0.05);
    border-left: 3px solid #FF6B9D;
    border-radius: 0 6px 6px 0;
    padding: 9px 13px;
    margin: 5px 0;
    font-size: 12px;
    color: #CDD6F4;
    line-height: 1.55;
}
.vcomment-lg {
    background: rgba(0,212,168,0.05);
    border-left: 3px solid #00D4A8;
    border-radius: 0 6px 6px 0;
    padding: 9px 13px;
    margin: 5px 0;
    font-size: 12px;
    color: #CDD6F4;
    line-height: 1.55;
}

/* Badge */
.vbadge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}
.vbadge-teal   { background: rgba(0,212,168,0.12); color: #00D4A8; border: 1px solid rgba(0,212,168,0.3); }
.vbadge-purple { background: rgba(139,108,248,0.12); color: #8B6CF8; border: 1px solid rgba(139,108,248,0.3); }

/* Platform bar */
.vpb-track { height: 6px; background: rgba(255,255,255,0.07); border-radius: 3px; overflow: hidden; margin: 5px 0 3px; }
.vpb-fill  { height: 100%; border-radius: 3px; }

/* CM cells */
.vcm-ok { background: rgba(0,212,168,0.12); border: 1px solid rgba(0,212,168,0.25); color: #00D4A8; }
.vcm-ng { background: rgba(255,85,85,0.05); border: 1px solid rgba(255,255,255,0.07); color: #8892B0; }
.vcm-cell {
    padding: 20px 10px;
    text-align: center;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS — Vigil color palette
# ═══════════════════════════════════════════════════════════════════
MODEL_RESULTS = {
    "RoBERTa": {
        "base":       "roberta-base",
        "macro_f1":   0.76,
        "accuracy":   0.75,
        "color":      "#8B6CF8",
        "misinfo":    {"precision": 0.72, "recall": 0.79, "f1": 0.76, "support": 1000},
        "legit":      {"precision": 0.77, "recall": 0.70, "f1": 0.73, "support": 1000},
        "cm":         [[720, 280], [250, 750]],
        "train_loss": [0.5906, 0.4741, 0.3796],
    },
    "MentalBERT": {
        "base":       "mental/mental-bert-base-uncased",
        "macro_f1":   0.77,
        "accuracy":   0.77,
        "color":      "#00D4A8",
        "misinfo":    {"precision": 0.78, "recall": 0.76, "f1": 0.77, "support": 1000},
        "legit":      {"precision": 0.77, "recall": 0.79, "f1": 0.78, "support": 1000},
        "cm":         [[760, 240], [210, 790]],
        "train_loss": [0.5426, 0.3783, 0.2149],
    },
}

MISINFO_CATEGORIES = {
    "Stigma / Dismissal":       {"count": 6572, "color": "#FF6B9D"},
    "Conspiracy / Anti-estab.": {"count": 3575, "color": "#8B6CF8"},
    "Harmful Advice":           {"count": 1939, "color": "#00D4A8"},
    "Anti-Medication":          {"count": 1170, "color": "#FF9040"},
    "Alternative Treatment":    {"count": 665,  "color": "#4CAF8E"},
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#CDD6F4", family="'IBM Plex Mono', monospace", size=10),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.07)",
        borderwidth=1,
        font=dict(size=11, family="'Outfit', sans-serif"),
    ),
)
GRID = dict(
    gridcolor="rgba(255,255,255,0.05)",
    linecolor="rgba(255,255,255,0.07)",
    tickfont=dict(size=9, family="'IBM Plex Mono', monospace", color="#8892B0"),
    zerolinecolor="rgba(255,255,255,0.05)",
)

MODEL_HUB_IDS = {
    "RoBERTa":    "Paulst7/roberta-mhmisinfo",
    "MentalBERT": "Paulst7/mentalbert-mhmisinfo",
}

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING  (unchanged)
# ═══════════════════════════════════════════════════════════════════
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_sample_data():
    rng = np.random.default_rng(42)
    misinfo_texts = [
        "Doctors don't want you to know this natural cure for depression.",
        "Big pharma is hiding the real treatments for mental illness.",
        "Mental illness is fake, just think positive and you'll be fine.",
        "Natural remedies can cure schizophrenia without dangerous medication.",
        "Psychiatrists are corrupt and just want to drug your children.",
        "Antidepressants are more dangerous than the depression itself.",
        "This herb cures ADHD in 30 days — doctors won't tell you.",
        "Stop taking your meds and try this simple detox instead.",
        "Mental health industry is a scam designed to control the population.",
        "Essential oils and diet can replace all psychiatric medications.",
        "Big pharma profits from keeping you sick — seek natural cures.",
        "Meditation alone cures bipolar disorder, no pills needed.",
        "Lithium is poison — your psychiatrist won't tell you the truth.",
        "ADHD isn't real, it's just sugar and screens causing the problem.",
        "Rewire your brain naturally — antidepressants destroy neural pathways.",
    ]
    legit_texts = [
        "I've been in therapy for 6 months and it has genuinely helped my anxiety.",
        "My doctor prescribed medication and it really changed my life for the better.",
        "Mental health awareness is so important — thank you for sharing this.",
        "If you're struggling, please reach out to a mental health professional.",
        "CBT helped me manage my depression. Highly recommend it.",
        "I was diagnosed with ADHD last year and the right medication made a huge difference.",
        "Therapy isn't a sign of weakness — it takes real courage to seek help.",
        "I relate to this so much. Living with OCD is really challenging.",
        "The stigma around mental health needs to end. We need to talk more openly.",
        "My psychiatrist has been amazing. Finding the right support changed everything.",
        "Recovery is not linear, but it is possible. Keep going.",
        "This video helped me understand what my friend is going through.",
        "Mental health days should be normalized just like physical sick days.",
        "Just started my mental health journey — nervous but hopeful.",
        "Six months of DBT and I barely recognise how much I've grown.",
        "Please don't suffer in silence — help is out there.",
        "My anxiety improved so much with the right combination of therapy and meds.",
        "I love how this community supports each other through tough times.",
        "Remember: asking for help is a sign of strength, not weakness.",
        "The research on this topic is really fascinating and encouraging.",
    ]
    n = 500
    platforms = rng.choice(["Youtube", "Bitchute"], size=n, p=[0.90, 0.10])
    labels = [-1 if rng.random() < (0.116 if p == "Youtube" else 0.055) else 0 for p in platforms]
    texts  = [rng.choice(misinfo_texts) if lbl == -1 else rng.choice(legit_texts) for lbl in labels]
    start_ts = int(pd.Timestamp("2021-01-01").timestamp())
    end_ts   = int(pd.Timestamp("2023-12-31").timestamp())
    comments = pd.DataFrame({
        "text":                           texts,
        "commenter_channel_display_name": [f"user_{i}" for i in range(n)],
        "comment_publish_date":           rng.integers(start_ts, end_ts, size=n).astype(float),
        "video_id":                       rng.choice([f"vid_{i}" for i in range(50)], size=n),
        "platform":                       platforms,
        "label":                          labels,
    })
    vid_platforms = rng.choice(["Youtube", "Bitchute"], size=50, p=[0.86, 0.14])
    vid_labels    = rng.choice([-1, 0], size=50, p=[0.16, 0.84])
    videos = pd.DataFrame({
        "video_id":            [f"vid_{i}" for i in range(50)],
        "video_title":         [f"Mental Health Video {i}" for i in range(50)],
        "label":               vid_labels,
        "platform":            vid_platforms,
        "video_comment_count": rng.integers(10, 500, size=50),
    })
    return comments, videos


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    full_c = os.path.join(DATA_DIR, "comments_MHMisinfo_Gold.csv")
    full_v = os.path.join(DATA_DIR, "videos_MHMisinfo_Gold.csv")
    samp_c = os.path.join(DATA_DIR, "data", "sample_comments.csv")
    samp_v = os.path.join(DATA_DIR, "data", "sample_videos.csv")
    if os.path.exists(full_c) and os.path.exists(full_v):
        comments = pd.read_csv(full_c)
        videos   = pd.read_csv(full_v)
        mode     = "full"
    elif os.path.exists(samp_c) and os.path.exists(samp_v):
        comments = pd.read_csv(samp_c)
        videos   = pd.read_csv(samp_v)
        mode     = "sample"
    else:
        comments, videos = generate_sample_data()
        mode = "synthetic"
    comments["date"]       = pd.to_datetime(comments["comment_publish_date"], unit="s", errors="coerce")
    comments["year_month"] = comments["date"].dt.to_period("M").astype(str)
    comments["is_misinfo"] = comments["label"] == -1
    videos["is_misinfo"]   = videos["label"] == -1
    return comments, videos, mode


@st.cache_resource(show_spinner="Loading model weights…")
def load_classifier(model_name: str):
    hub_id = MODEL_HUB_IDS[model_name]
    token  = os.environ.get("HF_TOKEN") or None
    try:
        if model_name == "RoBERTa":
            tokenizer = RobertaTokenizer.from_pretrained(hub_id, token=token)
            model     = RobertaForSequenceClassification.from_pretrained(hub_id, token=token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(hub_id, token=token)
            model     = AutoModelForSequenceClassification.from_pretrained(hub_id, token=token)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


def clean_text(text, max_len=220):
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"http\S+", "", text).strip()
    return text[:max_len] + ("…" if len(text) > max_len else "")


# ═══════════════════════════════════════════════════════════════════
# VIGIL HTML COMPONENTS
# ═══════════════════════════════════════════════════════════════════
def vkpi(value, label, sub, delta=None, delta_up=True):
    d_color = "#00D4A8" if delta_up else "#FF9040"
    d_arrow = "↑" if delta_up else "↓"
    delta_html = (
        f'<div class="vkpi-delta" style="color:{d_color}">{d_arrow} {delta}</div>'
        if delta else ""
    )
    return f"""
<div class="vkpi">
  <div class="vkpi-lbl">{label}</div>
  <div class="vkpi-val">{value}</div>
  {delta_html}
  <div class="vkpi-sub">{sub}</div>
</div>"""


def alert_bar(msg):
    return f"""
<div class="valert">
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
       stroke="#FF9040" stroke-width="2" style="flex-shrink:0">
    <path d="m10.29 3.86-8.69 15A2 2 0 0 0 3.27 22h17.46a2 2 0 0 0 1.67-3.14l-8.69-15a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
  <span>{msg}</span>
  <span style="margin-left:auto;color:#FF9040;font-size:11px;cursor:pointer">Analyse →</span>
</div>"""


def vfinding(label, value, sub, color):
    return f"""
<div class="vfind" style="border-left-color:{color}">
  <div class="vfind-lbl">{label}</div>
  <div class="vfind-val" style="color:{color}">{value}</div>
  <div class="vfind-sub">{sub}</div>
</div>"""


def legend_row(items):
    """items = list of (color, label)"""
    dots = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#8892B0">'
        f'<div style="width:8px;height:8px;border-radius:2px;background:{c}"></div>{l}</div>'
        for c, l in items
    )
    return f'<div style="display:flex;gap:16px;margin-bottom:12px">{dots}</div>'


def prob_bar(label, value_pct, color):
    return f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;font-size:11px">
    <span style="color:#8892B0">{label}</span>
    <span style="font-family:'IBM Plex Mono',monospace;color:{color}">{value_pct:.1f}%</span>
  </div>
  <div class="vpbar">
    <div class="vpbar-fill" style="width:{value_pct}%;background:{color}"></div>
  </div>
</div>"""


# ═══════════════════════════════════════════════════════════════════
# TAB RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def render_overview(fc, fv):
    total       = len(fc)
    n_misinfo   = int(fc["is_misinfo"].sum())
    misinfo_pct = n_misinfo / total * 100 if total else 0
    n_platforms = len(fc["platform"].unique())
    n_vid_ms    = int(fv["is_misinfo"].sum())
    vid_ms_pct  = n_vid_ms / len(fv) * 100 if len(fv) else 0

    # ── Page header ────────────────────────────────────────────────
    st.markdown(
        '<h1 style="font-size:22px;font-weight:700;margin-bottom:4px">Misinformation overview</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:11px;color:#8892B0;margin-bottom:0">'
        '<span style="color:#00D4A8">●</span> Active · MHMisinfo Gold Dataset · '
        '582,362 comments · YouTube &amp; Bitchute</p>',
        unsafe_allow_html=True,
    )

    # ── Alert bar ──────────────────────────────────────────────────
    st.markdown(
        alert_bar(
            "Platform disparity detected — Bitchute shows 2.1× higher misinformation rate · 11.6% vs 5.5%"
        ),
        unsafe_allow_html=True,
    )

    # ── KPI row ────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(vkpi(f"{total:,}", "Comments Analysed", "From 582,362 total (Gold)",
                         delta="stratified sample", delta_up=True), unsafe_allow_html=True)
    with k2:
        st.markdown(vkpi(f"{misinfo_pct:.1f}%", "Misinfo Rate", "Sample dataset",
                         delta="vs 11.4% full dataset", delta_up=False), unsafe_allow_html=True)
    with k3:
        st.markdown(vkpi(f"{n_misinfo:,}", "Misinfo Comments", "5 categories identified",
                         delta=f"from 66,537 total", delta_up=True), unsafe_allow_html=True)
    with k4:
        st.markdown(vkpi(f"{vid_ms_pct:.0f}%", "Video Misinfo Rate",
                         f"{n_vid_ms} of {len(fv)} videos",
                         delta="2.7× vs comment rate", delta_up=False), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown(
            '<div class="vcard-title">Comment volume over time</div>'
            '<div class="vcard-sub">Monthly distribution · Gold dataset · Jan 2022 – Jan 2024</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            legend_row([("#00D4A8", "Legitimate"), ("#FF6B9D", "Misinformation")]),
            unsafe_allow_html=True,
        )
        valid_fc = fc.dropna(subset=["date"])
        if len(valid_fc) > 0:
            trend = (
                valid_fc.groupby(["year_month", "is_misinfo"])
                .size().reset_index(name="count").sort_values("year_month")
            )
            t_legit   = trend[~trend["is_misinfo"]]
            t_misinfo = trend[trend["is_misinfo"]]
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=t_legit["year_month"], y=t_legit["count"],
                name="Legitimate", mode="lines",
                line=dict(color="#00D4A8", width=2), fill="tozeroy",
                fillcolor="rgba(0,212,168,0.1)",
            ))
            fig_trend.add_trace(go.Scatter(
                x=t_misinfo["year_month"], y=t_misinfo["count"],
                name="Misinformation", mode="lines",
                line=dict(color="#FF6B9D", width=2), fill="tozeroy",
                fillcolor="rgba(255,107,157,0.12)",
            ))
            fig_trend.update_layout(
                **CHART_LAYOUT, height=220, showlegend=False,
                xaxis=dict(**GRID), yaxis=dict(**GRID, title=""),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No valid date data for the current filter selection.")

    with col_r:
        st.markdown(
            '<div class="vcard-title">Category distribution</div>'
            '<div class="vcard-sub">Full dataset · 66,537 flagged</div>',
            unsafe_allow_html=True,
        )
        cats_names  = list(MISINFO_CATEGORIES.keys())
        cats_counts = [v["count"] for v in MISINFO_CATEGORIES.values()]
        cats_colors = [v["color"] for v in MISINFO_CATEGORIES.values()]
        fig_donut = go.Figure(go.Pie(
            values=cats_counts, labels=cats_names, hole=0.58,
            marker=dict(colors=cats_colors, line=dict(color="#111420", width=3)),
            textinfo="none",
        ))
        fig_donut.add_annotation(
            text="<b>66,537</b><br><span style='font-size:9px'>TOTAL FLAGS</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#CDD6F4", size=14, family="'IBM Plex Mono', monospace"),
        )
        fig_donut.update_layout(**CHART_LAYOUT, height=200, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

        # Category legend
        total_mis = sum(cats_counts)
        for name, data in MISINFO_CATEGORIES.items():
            pct = data["count"] / total_mis * 100
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
                f'<div style="width:6px;height:6px;border-radius:50%;background:{data["color"]};flex-shrink:0"></div>'
                f'<span style="font-size:10px;color:#8892B0;flex:1">{name}</span>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#CDD6F4">{data["count"]:,}</span>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:{data["color"]};width:36px;text-align:right">{pct:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Platform comparison ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown(
            '<div class="vcard-title">Comment composition by platform (%)</div>'
            '<div class="vcard-sub">Sample dataset · selected platforms</div>',
            unsafe_allow_html=True,
        )
        plat = (
            fc.groupby("platform")
            .agg(legit=("is_misinfo", lambda x: (~x).sum()), misinfo=("is_misinfo", "sum"))
            .reset_index()
        )
        plat["total"]       = plat["legit"] + plat["misinfo"]
        plat["legit_pct"]   = plat["legit"]   / plat["total"] * 100
        plat["misinfo_pct"] = plat["misinfo"] / plat["total"] * 100
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            name="Legitimate", x=plat["platform"], y=plat["legit_pct"],
            marker_color="#00D4A8", opacity=.85,
            text=plat["legit_pct"].map("{:.1f}%".format), textposition="inside",
            textfont=dict(color="#000", size=11),
        ))
        fig_stack.add_trace(go.Bar(
            name="Misinformation", x=plat["platform"], y=plat["misinfo_pct"],
            marker_color="#FF6B9D", opacity=.85,
            text=plat["misinfo_pct"].map("{:.1f}%".format), textposition="inside",
            textfont=dict(color="#fff", size=11),
        ))
        fig_stack.update_layout(
            **CHART_LAYOUT, barmode="stack", height=260,
            xaxis=GRID, yaxis=dict(**GRID, title="% of Comments", range=[0, 100]),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    with pc2:
        st.markdown(
            '<div class="vcard-title">Misinformation rate by platform (full dataset)</div>'
            '<div class="vcard-sub">Source: 582,362 comments</div>',
            unsafe_allow_html=True,
        )
        fig_rate = go.Figure(go.Bar(
            x=["YouTube", "Bitchute"], y=[11.6, 5.5],
            marker_color=["#00D4A8", "#8B6CF8"], opacity=.85,
            text=["11.6%", "5.5%"], textposition="outside",
            textfont=dict(color="#CDD6F4", size=13),
        ))
        fig_rate.update_layout(
            **CHART_LAYOUT, height=260,
            xaxis=GRID, yaxis=dict(**GRID, title="Misinfo rate (%)", range=[0, 16]),
        )
        st.plotly_chart(fig_rate, use_container_width=True)


def render_model_tab(selected_model, m, badge_cls):
    other_name = "MentalBERT" if selected_model == "RoBERTa" else "RoBERTa"
    other      = MODEL_RESULTS[other_name]
    badge_cls_v = "vbadge-teal" if selected_model == "MentalBERT" else "vbadge-purple"

    # ── Header ────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="font-size:22px;font-weight:700;margin-bottom:8px">Model performance</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="vbadge {badge_cls_v}">{selected_model}</span>&nbsp; '
        f'<code>{m["base"]}</code>&nbsp;·&nbsp;'
        f'<small>Test set: 2,000 samples · 3 epochs · lr=2e-5</small>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Macro F1", f"{m['macro_f1']:.2f}",
                  delta=f"{m['macro_f1']-other['macro_f1']:+.2f} vs {other_name}")
    with mc2:
        st.metric("Accuracy", f"{m['accuracy']:.2f}",
                  delta=f"{m['accuracy']-other['accuracy']:+.2f} vs {other_name}")
    with mc3:
        st.metric("Misinfo Precision", f"{m['misinfo']['precision']:.2f}",
                  delta=f"{m['misinfo']['precision']-other['misinfo']['precision']:+.2f}")
    with mc4:
        st.metric("Misinfo Recall", f"{m['misinfo']['recall']:.2f}",
                  delta=f"{m['misinfo']['recall']-other['misinfo']['recall']:+.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-class + Confusion Matrix ──────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            '<div class="vcard-title">Per-class performance</div>'
            '<div class="vcard-sub">Precision · Recall · F1-Score by class</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            legend_row([("#FF6B9D", "Misinfo"), ("#00D4A8", "Legit")]),
            unsafe_allow_html=True,
        )
        metrics   = ["Precision", "Recall", "F1-Score"]
        m_misinfo = [m["misinfo"]["precision"], m["misinfo"]["recall"], m["misinfo"]["f1"]]
        m_legit   = [m["legit"]["precision"],   m["legit"]["recall"],   m["legit"]["f1"]]
        fig_cls = go.Figure()
        fig_cls.add_trace(go.Bar(
            name="Misinfo", x=metrics, y=m_misinfo,
            marker_color="#FF6B9D", opacity=.85, marker_cornerradius=3,
            text=[f"{v:.2f}" for v in m_misinfo], textposition="outside",
            textfont=dict(color="#CDD6F4", size=10),
        ))
        fig_cls.add_trace(go.Bar(
            name="Legit", x=metrics, y=m_legit,
            marker_color="#00D4A8", opacity=.85, marker_cornerradius=3,
            text=[f"{v:.2f}" for v in m_legit], textposition="outside",
            textfont=dict(color="#CDD6F4", size=10),
        ))
        fig_cls.update_layout(
            **CHART_LAYOUT, barmode="group", height=300, showlegend=False,
            xaxis=GRID, yaxis=dict(**GRID, range=[0, 0.98]),
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    with col_b:
        st.markdown(
            f'<div class="vcard-title">Confusion matrix</div>'
            f'<div class="vcard-sub">Test set (2,000 samples) · {selected_model}</div>',
            unsafe_allow_html=True,
        )
        cm = m["cm"]
        # Vigil-style HTML confusion matrix
        st.markdown(f"""
<div style="display:grid;grid-template-columns:80px 1fr 1fr;gap:8px;margin-top:16px">
  <div></div>
  <div style="font-size:9px;color:#3D4A6B;text-align:center;padding-bottom:4px">Pred: Misinfo</div>
  <div style="font-size:9px;color:#3D4A6B;text-align:center;padding-bottom:4px">Pred: Legit</div>
  <div style="font-size:9px;color:#3D4A6B;text-align:right;padding-right:8px;display:flex;align-items:center;justify-content:flex-end">Act: Misinfo</div>
  <div class="vcm-cell vcm-ok">{cm[0][0]:,}</div>
  <div class="vcm-cell vcm-ng">{cm[0][1]:,}</div>
  <div style="font-size:9px;color:#3D4A6B;text-align:right;padding-right:8px;display:flex;align-items:center;justify-content:flex-end">Act: Legit</div>
  <div class="vcm-cell vcm-ng">{cm[1][0]:,}</div>
  <div class="vcm-cell vcm-ok">{cm[1][1]:,}</div>
</div>""", unsafe_allow_html=True)

    # ── Training Loss ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="vcard-title">Training loss — all epochs</div>'
        '<div class="vcard-sub">RoBERTa vs MentalBERT · 3 epochs · lr=2e-5</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        legend_row([("#8B6CF8", "RoBERTa"), ("#00D4A8", "MentalBERT")]),
        unsafe_allow_html=True,
    )
    fig_loss = go.Figure()
    for name, res in MODEL_RESULTS.items():
        is_sel = name == selected_model
        fig_loss.add_trace(go.Scatter(
            x=[1, 2, 3], y=res["train_loss"],
            name=name, mode="lines+markers",
            line=dict(color=res["color"], width=2.5 if is_sel else 1.5),
            marker=dict(size=8 if is_sel else 5, color=res["color"]),
            opacity=1.0 if is_sel else 0.35,
        ))
    fig_loss.update_layout(
        **CHART_LAYOUT, height=180, showlegend=False,
        xaxis=dict(**GRID, title="Epoch", tickvals=[1, 2, 3]),
        yaxis=dict(**GRID, title="Loss"),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # ── Side-by-side comparison ────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="vcard-title" style="margin-bottom:12px">Side-by-side comparison</div>',
        unsafe_allow_html=True,
    )
    comp1, comp2 = st.columns(2)
    for col, (name, res) in zip([comp1, comp2], MODEL_RESULTS.items()):
        bc    = "vbadge-teal" if name == "MentalBERT" else "vbadge-purple"
        hl    = f"border: 1px solid {res['color']}55;" if name == selected_model \
                else "border: 1px solid rgba(255,255,255,0.07);"
        arrow = f'<small style="color:{res["color"]};margin-left:6px">▲ selected</small>' \
                if name == selected_model else ""
        with col:
            st.markdown(f"""
<div style="background:#111420;{hl}border-radius:10px;padding:18px">
  <span class="vbadge {bc}">{name}</span>{arrow}
  <br><br>
  <table class="vtable">
    <tr class="vtable-hdr"><td>Metric</td><td style="text-align:right">Misinfo</td><td style="text-align:right">Legit</td></tr>
    <tr><td>Precision</td>
      <td style="text-align:right;font-family:'IBM Plex Mono',monospace">{res['misinfo']['precision']:.2f}</td>
      <td style="text-align:right;font-family:'IBM Plex Mono',monospace">{res['legit']['precision']:.2f}</td></tr>
    <tr><td>Recall</td>
      <td style="text-align:right;font-family:'IBM Plex Mono',monospace">{res['misinfo']['recall']:.2f}</td>
      <td style="text-align:right;font-family:'IBM Plex Mono',monospace">{res['legit']['recall']:.2f}</td></tr>
    <tr><td>F1-Score</td>
      <td style="text-align:right;font-family:'IBM Plex Mono',monospace">{res['misinfo']['f1']:.2f}</td>
      <td style="text-align:right;font-family:'IBM Plex Mono',monospace">{res['legit']['f1']:.2f}</td></tr>
    <tr style="font-weight:700;color:{res['color']};border-top:1px solid rgba(255,255,255,0.07)">
      <td>Macro F1</td>
      <td style="text-align:right;font-family:'IBM Plex Mono',monospace" colspan="2">{res['macro_f1']:.2f}</td>
    </tr>
  </table>
</div>""", unsafe_allow_html=True)


def render_pattern_tab(fc):
    st.markdown(
        '<h1 style="font-size:22px;font-weight:700;margin-bottom:4px">Pattern analysis</h1>'
        '<p style="font-size:11px;color:#8892B0">Category breakdown, platform distribution, '
        'temporal patterns · Full dataset · 582,362 comments</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    total_misinfo_full = 66537
    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        st.markdown(
            '<div class="vcard-title">Category breakdown</div>'
            '<div class="vcard-sub">Full dataset · 5 misinformation classes · sorted by frequency</div>',
            unsafe_allow_html=True,
        )
        cats   = list(MISINFO_CATEGORIES.keys())
        counts = [v["count"] for v in MISINFO_CATEGORIES.values()]
        colors = [v["color"] for v in MISINFO_CATEGORIES.values()]
        pcts   = [c / total_misinfo_full * 100 for c in counts]
        fig_cats = go.Figure(go.Bar(
            x=counts, y=cats, orientation="h",
            marker_color=colors, marker_cornerradius=3,
            text=[f"{c:,}  ({p:.1f}%)" for c, p in zip(counts, pcts)],
            textposition="outside",
            textfont=dict(color="#CDD6F4", size=10),
        ))
        fig_cats.update_layout(
            **CHART_LAYOUT, height=300,
            xaxis=dict(**GRID, title="Number of comments", range=[0, 8400]),
            yaxis=dict(**GRID, autorange="reversed"),
        )
        st.plotly_chart(fig_cats, use_container_width=True)

    with col_p2:
        st.markdown(
            '<div class="vcard-title">Key findings</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            vfinding("Total misinfo (full dataset)", "66,537",
                     "out of 582,362 comments · 11.4%", "#FF6B9D") +
            vfinding("Largest category", "Stigma / Dismissal",
                     "9.9% · most prevalent type", "#FF9040") +
            vfinding("Medication mentions", "1.5× more",
                     "Misinfo 1.2% vs Legit 0.8%", "#00D4A8") +
            vfinding("Platform gap", "Bitchute 2.1×",
                     "Higher misinfo rate vs YouTube", "#8B6CF8"),
            unsafe_allow_html=True,
        )

    # ── Platform bars ─────────────────────────────────────────────
    st.markdown("---")
    pb1, pb2 = st.columns(2)
    with pb1:
        st.markdown(
            '<div class="vcard-title">Platform misinformation rate</div>'
            '<div class="vcard-sub">Full dataset · 582,362 comments</div>',
            unsafe_allow_html=True,
        )
        for platform, rate, color, n in [
            ("YouTube", 11.6, "#00D4A8", "~400K comments"),
            ("Bitchute", 5.5, "#8B6CF8", "~182K comments"),
        ]:
            st.markdown(f"""
<div style="margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:12px">
    <span style="color:#CDD6F4">{platform}</span>
    <span style="font-family:'IBM Plex Mono',monospace;color:{color};font-weight:600">{rate}%</span>
  </div>
  <div class="vpb-track"><div class="vpb-fill" style="width:{rate*5}%;background:{color}"></div></div>
  <div style="font-size:9px;color:#3D4A6B;margin-top:3px">{n}</div>
</div>""", unsafe_allow_html=True)

    with pb2:
        st.markdown(
            '<div class="vcard-title">Temporal trend</div>'
            '<div class="vcard-sub">Monthly · Gold dataset · Jan 2022 – Jan 2024</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            legend_row([("#00D4A8", "Legitimate"), ("#FF6B9D", "Misinformation")]),
            unsafe_allow_html=True,
        )
        valid_fc = fc.dropna(subset=["date"])
        if len(valid_fc) > 0:
            trend = (
                valid_fc.groupby(["year_month", "is_misinfo"])
                .size().reset_index(name="count").sort_values("year_month")
            )
            t_legit   = trend[~trend["is_misinfo"]]
            t_misinfo = trend[trend["is_misinfo"]]
            fig_t2 = go.Figure()
            fig_t2.add_trace(go.Scatter(
                x=t_legit["year_month"], y=t_legit["count"],
                name="Legitimate", mode="lines",
                line=dict(color="#00D4A8", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,168,0.08)",
            ))
            fig_t2.add_trace(go.Scatter(
                x=t_misinfo["year_month"], y=t_misinfo["count"],
                name="Misinformation", mode="lines",
                line=dict(color="#FF6B9D", width=2),
                fill="tozeroy", fillcolor="rgba(255,107,157,0.12)",
            ))
            fig_t2.update_layout(
                **CHART_LAYOUT, height=200, showlegend=False,
                xaxis=dict(**GRID), yaxis=dict(**GRID),
            )
            st.plotly_chart(fig_t2, use_container_width=True)

    # ── Sample comments ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="vcard-title" style="margin-bottom:12px">Sample comments</div>',
        unsafe_allow_html=True,
    )
    cs1, cs2 = st.columns(2)
    with cs1:
        st.markdown(
            '<span style="color:#FF6B9D;font-weight:600;font-size:12px">⚠ Misinformation</span>',
            unsafe_allow_html=True,
        )
        n_avail    = int(fc["is_misinfo"].sum())
        sample_mis = fc[fc["is_misinfo"]]["text"].dropna().sample(min(5, n_avail), random_state=42)
        for t in sample_mis:
            st.markdown(
                f'<div class="vcomment-ms">{clean_text(t)}</div>',
                unsafe_allow_html=True,
            )
    with cs2:
        st.markdown(
            '<span style="color:#00D4A8;font-weight:600;font-size:12px">✓ Legitimate</span>',
            unsafe_allow_html=True,
        )
        n_legit    = int((~fc["is_misinfo"]).sum())
        sample_leg = fc[~fc["is_misinfo"]]["text"].dropna().sample(min(5, n_legit), random_state=77)
        for t in sample_leg:
            st.markdown(
                f'<div class="vcomment-lg">{clean_text(t)}</div>',
                unsafe_allow_html=True,
            )


def render_classifier_tab(selected_model, m, badge_cls):
    tokenizer, clf_model, load_error = load_classifier(selected_model)
    model_ready = tokenizer is not None
    badge_cls_v = "vbadge-teal" if selected_model == "MentalBERT" else "vbadge-purple"

    # Session state
    for key, default in [
        ("clf_history", []),
        ("auto_classify", False),
        ("classifier_input", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Header ────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="font-size:22px;font-weight:700;margin-bottom:6px">Live classifier</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="vbadge {badge_cls_v}">{selected_model}</span>&nbsp;'
        f'<code>{m["base"]}</code>&nbsp;·&nbsp;'
        f'<code>{MODEL_HUB_IDS[selected_model]}</code>&nbsp;·&nbsp;'
        f'<span style="color:#00D4A8">● realtime</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model load error ──────────────────────────────────────────
    if not model_ready:
        err_msg = load_error or "Set HF_TOKEN environment variable and restart."
        st.markdown(
            f'<div class="valert" style="background:rgba(255,85,85,0.08);border-color:rgba(255,85,85,0.25)">'
            f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#FF5555" stroke-width="2" style="flex-shrink:0">'
            f'<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
            f'<span><b>Model not loaded</b> — {err_msg}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick examples ────────────────────────────────────────────
    st.markdown(
        '<div class="vcard-title">Quick examples</div>'
        '<div class="vcard-sub" style="margin-bottom:8px">Click an example to auto-fill and classify</div>',
        unsafe_allow_html=True,
    )
    misinfo_pool = comments_df[comments_df["is_misinfo"]]["text"].dropna()
    legit_pool   = comments_df[~comments_df["is_misinfo"]]["text"].dropna()
    ex_mis = misinfo_pool.sample(min(2, len(misinfo_pool)), random_state=99).tolist()
    ex_leg = legit_pool.sample(min(2, len(legit_pool)), random_state=99).tolist()
    examples = []
    for mis_t, leg_t in zip(ex_mis, ex_leg):
        examples.append({"text": mis_t, "tag": "Misinformation"})
        examples.append({"text": leg_t, "tag": "Legitimate"})

    ex_cols = st.columns(2)
    for i, ex in enumerate(examples[:4]):
        is_ms  = ex["tag"] == "Misinformation"
        color  = "#FF5555" if is_ms else "#00D4A8"
        prefix = "⚠ MISINFO" if is_ms else "✓ LEGIT"
        short  = ex["text"][:55].rstrip() + "…"
        with ex_cols[i % 2]:
            st.markdown(
                f'<div style="font-size:9px;color:{color};font-weight:600;margin-bottom:2px">{prefix}</div>',
                unsafe_allow_html=True,
            )
            if st.button(f'"{short}"', key=f"ex_{i}", use_container_width=True):
                st.session_state.classifier_input = ex["text"]
                st.session_state.auto_classify    = True

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input + classify ──────────────────────────────────────────
    user_text = st.text_area(
        "Enter a social media comment to classify:",
        placeholder="e.g. 'Doctors are hiding the cure for depression…' or 'I've been in therapy and it really helped…'",
        height=110,
        key="classifier_input",
    )

    act_col, info_col = st.columns([1, 4])
    with act_col:
        classify = st.button(
            "Classify →", type="primary",
            disabled=not model_ready or not user_text.strip(),
        )
    with info_col:
        st.markdown(
            f'<small>{"Model loaded ✓" if model_ready else "Model not loaded"} · '
            f'Max token length: 128 · Classes: Misinformation / Legitimate</small>',
            unsafe_allow_html=True,
        )

    # ── Inference ─────────────────────────────────────────────────
    run_inference = classify or st.session_state.auto_classify
    if st.session_state.auto_classify:
        st.session_state.auto_classify = False

    if run_inference and user_text.strip() and model_ready:
        with st.spinner("Running inference…"):
            inputs = tokenizer(
                user_text, return_tensors="pt",
                truncation=True, max_length=128, padding=True,
            )
            with torch.no_grad():
                logits = clf_model(**inputs).logits
            probs      = torch.softmax(logits, dim=1)[0]
            pred       = torch.argmax(logits, dim=1).item()
            label      = "Misinformation" if pred == 0 else "Legitimate"
            confidence = probs[pred].item() * 100
            ms_prob    = probs[0].item() * 100
            lg_prob    = probs[1].item() * 100

        if pred == 0:
            st.markdown(
                f'<div class="vcls-ms">'
                f'<div class="vcls-label" style="color:#FF5555">⚠ {label}</div>'
                f'<div class="vcls-conf" style="color:#CDD6F4">{confidence:.1f}%</div>'
                f'<div class="vcls-sub">{selected_model} confidence</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="vcls-result">'
                f'<div class="vcls-label" style="color:#00D4A8">✓ {label}</div>'
                f'<div class="vcls-conf" style="color:#CDD6F4">{confidence:.1f}%</div>'
                f'<div class="vcls-sub">{selected_model} confidence</div></div>',
                unsafe_allow_html=True,
            )

        # Probability bars
        st.markdown(
            '<div style="margin-top:12px;font-size:9px;text-transform:uppercase;'
            'letter-spacing:.1em;color:#3D4A6B;margin-bottom:8px">Class scores</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            prob_bar("Misinformation", ms_prob, "#FF5555") +
            prob_bar("Legitimate", lg_prob, "#00D4A8"),
            unsafe_allow_html=True,
        )

        short_comment = user_text[:80].rstrip() + ("…" if len(user_text) > 80 else "")
        st.session_state.clf_history.insert(0, {
            "Comment":    short_comment,
            "Label":      label,
            "Confidence": f"{confidence:.1f}%",
            "Model":      selected_model,
        })

    # ── Classification history ────────────────────────────────────
    st.markdown("---")
    hist_hdr, clear_col = st.columns([5, 1])
    with hist_hdr:
        st.markdown(
            '<div class="vcard-title">Classification history</div>',
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.button("Clear", key="clear_history"):
            st.session_state.clf_history = []

    if st.session_state.clf_history:
        df_hist = pd.DataFrame(st.session_state.clf_history)
        st.dataframe(
            df_hist, use_container_width=True, hide_index=True,
            column_config={
                "Comment":    st.column_config.TextColumn(width="large"),
                "Label":      st.column_config.TextColumn(width="small"),
                "Confidence": st.column_config.TextColumn(width="small"),
                "Model":      st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.markdown(
            '<div style="color:#3D4A6B;font-size:11px;padding:8px 0">'
            'No classifications yet this session.</div>',
            unsafe_allow_html=True,
        )

    # ── Model reference ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="background:rgba(139,108,248,0.08);border:1px solid rgba(139,108,248,0.2);'
        f'border-left:3px solid #8B6CF8;border-radius:0 8px 8px 0;padding:12px 16px">'
        f'<div style="font-size:12px;font-weight:600;margin-bottom:4px">{selected_model} quick reference</div>'
        f'<div style="font-size:11px;color:#8892B0">'
        f'Macro F1: <b style="color:#CDD6F4;font-family:\'IBM Plex Mono\',monospace">{m["macro_f1"]:.2f}</b>&nbsp;·&nbsp;'
        f'Accuracy: <b style="color:#CDD6F4;font-family:\'IBM Plex Mono\',monospace">{m["accuracy"]:.2f}</b><br>'
        f'Catches <b style="color:#00D4A8">{int(m["misinfo"]["recall"]*100)}%</b> of real misinformation '
        f'(recall) with <b style="color:#00D4A8">{int(m["misinfo"]["precision"]*100)}%</b> precision</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════
comments_df, videos_df, DATA_MODE = load_data()

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — Vigil style
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
        '<div style="width:28px;height:28px;border-radius:6px;background:rgba(0,212,168,.12);'
        'border:1px solid rgba(0,212,168,.2);display:flex;align-items:center;justify-content:center">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00D4A8" stroke-width="2">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div>'
        '<div><div style="font-size:13px;font-weight:700;color:#CDD6F4">MHMisinfo</div>'
        '<div style="font-size:9px;color:#3D4A6B">mh-misinfo · ops</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Model")
    selected_model = st.selectbox(
        "Select classifier",
        options=list(MODEL_RESULTS.keys()),
        index=1,
        help="All model-performance sections update to reflect the selected model.",
    )
    m = MODEL_RESULTS[selected_model]
    badge_cls   = "badge-purple" if selected_model == "RoBERTa" else "badge-teal"
    badge_cls_v = "vbadge-purple" if selected_model == "RoBERTa" else "vbadge-teal"
    st.markdown(
        f'<span class="vbadge {badge_cls_v}">{selected_model}</span>'
        f'&nbsp;<small>Macro F1 = {m["macro_f1"]:.2f}</small>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Filters")
    platform_filter = st.multiselect(
        "Platform",
        options=["Youtube", "Bitchute"],
        default=["Youtube", "Bitchute"],
    )
    st.markdown("---")

    st.markdown("### Dataset")
    if DATA_MODE == "full":
        c_rows, c_mis, c_leg = "135,445", "8,025", "127,420"
        v_rows, v_mis, v_leg = "739", "120", "619"
        c_label, v_label     = "Gold Comments", "Gold Videos"
    elif DATA_MODE == "sample":
        c_rows = f"{len(comments_df):,}"
        c_mis  = str(int(comments_df['is_misinfo'].sum()))
        c_leg  = str(int((~comments_df['is_misinfo']).sum()))
        v_rows = f"{len(videos_df):,}"
        v_mis  = str(int(videos_df['is_misinfo'].sum()))
        v_leg  = str(int((~videos_df['is_misinfo']).sum()))
        c_label, v_label = "Comments (stratified sample)", "Videos (stratified sample)"
    else:
        c_rows = f"{len(comments_df):,}"
        c_mis  = str(int(comments_df['is_misinfo'].sum()))
        c_leg  = str(int((~comments_df['is_misinfo']).sum()))
        v_rows = f"{len(videos_df):,}"
        v_mis  = str(int(videos_df['is_misinfo'].sum()))
        v_leg  = str(int((~videos_df['is_misinfo']).sum()))
        c_label, v_label = "Comments (synthetic)", "Videos (synthetic)"

    st.markdown(
        f'<div style="background:rgba(0,212,168,.06);border:1px solid rgba(0,212,168,.15);'
        f'border-left:3px solid #00D4A8;border-radius:0 6px 6px 0;padding:10px 12px;margin-bottom:8px">'
        f'<div style="font-size:11px;font-weight:600;margin-bottom:3px">{c_label}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;font-weight:600;color:#CDD6F4">{c_rows}</div>'
        f'<div style="font-size:10px;color:#3D4A6B;margin-top:2px">{c_mis} misinfo · {c_leg} legit</div></div>'
        f'<div style="background:rgba(139,108,248,.06);border:1px solid rgba(139,108,248,.15);'
        f'border-left:3px solid #8B6CF8;border-radius:0 6px 6px 0;padding:10px 12px">'
        f'<div style="font-size:11px;font-weight:600;margin-bottom:3px">{v_label}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;font-weight:600;color:#CDD6F4">{v_rows}</div>'
        f'<div style="font-size:10px;color:#3D4A6B;margin-top:2px">{v_mis} misinfo · {v_leg} legit</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("PRT840 IT Thesis · CDU · 2026\nPaul S. Flores Sinche (S386377)")

# ═══════════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════════
fc = comments_df[comments_df["platform"].isin(platform_filter)] if platform_filter else comments_df
fv = videos_df[videos_df["platform"].isin(platform_filter)]     if platform_filter else videos_df

# ── Data mode banner ──────────────────────────────────────────────
if DATA_MODE == "sample":
    st.info(
        "**Stratified sample** — showing a representative 1,000-comment / 200-video sample "
        "drawn from the real MHMisinfo Gold dataset. Class proportions reflect the full dataset.",
        icon="📊",
    )
elif DATA_MODE == "synthetic":
    st.warning(
        "**Demo mode** — CSV files not found. Displaying synthetic data. "
        "Run locally with CSV files for real statistics.",
        icon="⚠️",
    )

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Model Performance",
    "Pattern Analysis",
    "Live Classifier",
])

with tab1:
    render_overview(fc, fv)

with tab2:
    render_model_tab(selected_model, m, badge_cls)

with tab3:
    render_pattern_tab(fc)

with tab4:
    render_classifier_tab(selected_model, m, badge_cls)
