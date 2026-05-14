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
    page_title="MHMisinfo Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS THEME — Dark CRM / Purple-Navy
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
*, *::before, *::after { font-family: 'Inter', sans-serif !important; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0f0e17 !important;
    color: #e2e8f0;
}
[data-testid="stMainBlockContainer"] { background-color: #0f0e17 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #1a1830 !important;
    border-right: 1px solid rgba(124, 58, 237, 0.2);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(124, 58, 237, 0.2) !important; }

/* ── Headings ── */
h1, h2, h3, h4, h5 { color: #e2e8f0 !important; font-weight: 600 !important; }
p, li, label, .stMarkdown { color: #e2e8f0; }
hr { border-color: rgba(124, 58, 237, 0.2) !important; }
small { color: #94a3b8; }
code { background: rgba(124, 58, 237, 0.15) !important; color: #a78bfa !important; border-radius: 4px; padding: 1px 5px; }

/* ── Metric containers (Tab 2) ── */
[data-testid="metric-container"] {
    background: #1a1830 !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 12px;
    padding: 18px 16px !important;
    box-shadow: 0 0 16px rgba(124, 58, 237, 0.08);
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #7c3aed !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="metric-delta"] { font-size: 0.78rem !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(124, 58, 237, 0.2);
    gap: 4px;
}
[data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border-radius: 8px 8px 0 0;
    padding: 8px 16px !important;
}
[data-baseweb="tab"]:hover { color: #a78bfa !important; background: rgba(124, 58, 237, 0.08) !important; }
[data-baseweb="tab"][aria-selected="true"] {
    color: #7c3aed !important;
    background: rgba(124, 58, 237, 0.12) !important;
    border-bottom: 2px solid #7c3aed !important;
}

/* ── Inputs ── */
[data-baseweb="select"] > div {
    background-color: #1a1830 !important;
    border-color: rgba(124, 58, 237, 0.3) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] span { color: #e2e8f0 !important; }
[data-baseweb="popover"] [role="option"] { background-color: #1a1830 !important; }
[data-baseweb="popover"] [role="option"]:hover { background-color: rgba(124, 58, 237, 0.15) !important; }
textarea, [data-baseweb="textarea"] textarea {
    background-color: #1a1830 !important;
    color: #e2e8f0 !important;
    border-color: rgba(124, 58, 237, 0.3) !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7c3aed, #06b6d4) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="baseButton-primary"]:disabled {
    background: #1a1830 !important;
    color: #94a3b8 !important;
}

/* ── Multiselect tags ── */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: rgba(124, 58, 237, 0.2) !important;
    border: 1px solid rgba(124, 58, 237, 0.4) !important;
}

/* ── Info/warn alerts ── */
[data-testid="stAlert"] { background-color: #1a1830 !important; border-radius: 8px; }

/* ══ KPI CARDS ══ */
.kpi-gradient {
    background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
    border: 1px solid rgba(124, 58, 237, 0.5);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    margin-top: 6px;
    box-shadow: 0 0 24px rgba(124, 58, 237, 0.2), 0 2px 8px rgba(0,0,0,0.4);
}
.kpi-solid {
    background: #1e1b4b;
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    margin-top: 6px;
    box-shadow: 0 0 16px rgba(124, 58, 237, 0.1), 0 2px 8px rgba(0,0,0,0.3);
}
.kpi-value-light {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}
.kpi-value-accent {
    font-size: 2rem;
    font-weight: 700;
    color: #06b6d4;
    margin: 0;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.7);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
.kpi-label-dark {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ══ GENERIC CARD ══ */
.card {
    background: #1a1830;
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 0 16px rgba(124, 58, 237, 0.08);
}

/* ══ BADGES ══ */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.35); }
.badge-teal   { background: rgba(6,182,212,0.15);  color: #22d3ee; border: 1px solid rgba(6,182,212,0.35); }

/* ══ INFO / WARN BOXES ══ */
.info-box {
    background: rgba(124, 58, 237, 0.08);
    border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #e2e8f0;
}
.warn-box {
    background: rgba(236, 72, 153, 0.08);
    border-left: 3px solid #ec4899;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #e2e8f0;
}

/* ══ COMMENT SAMPLE CARDS ══ */
.comment-card-pink {
    background: rgba(236, 72, 153, 0.06);
    border-left: 3px solid #ec4899;
    border-radius: 0 6px 6px 0;
    padding: 9px 13px;
    margin: 5px 0;
    font-size: 0.82rem;
    color: #e2e8f0;
    line-height: 1.55;
}
.comment-card-teal {
    background: rgba(6, 182, 212, 0.06);
    border-left: 3px solid #06b6d4;
    border-radius: 0 6px 6px 0;
    padding: 9px 13px;
    margin: 5px 0;
    font-size: 0.82rem;
    color: #e2e8f0;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════
MODEL_RESULTS = {
    "RoBERTa": {
        "base":       "roberta-base",
        "macro_f1":   0.76,
        "accuracy":   0.75,
        "color":      "#7c3aed",
        "misinfo":    {"precision": 0.72, "recall": 0.79, "f1": 0.76, "support": 1000},
        "legit":      {"precision": 0.77, "recall": 0.70, "f1": 0.73, "support": 1000},
        "cm":         [[790, 210], [300, 700]],
        "train_loss": [0.5906, 0.4741, 0.3796],
    },
    "MentalBERT": {
        "base":       "mental/mental-bert-base-uncased",
        "macro_f1":   0.77,
        "accuracy":   0.77,
        "color":      "#06b6d4",
        "misinfo":    {"precision": 0.78, "recall": 0.76, "f1": 0.77, "support": 1000},
        "legit":      {"precision": 0.77, "recall": 0.79, "f1": 0.78, "support": 1000},
        "cm":         [[760, 240], [210, 790]],
        "train_loss": [0.5426, 0.3783, 0.2149],
    },
}

MISINFO_CATEGORIES = {
    "Stigma / Dismissal":       {"count": 6572, "color": "#ec4899"},
    "Conspiracy / Anti-estab.": {"count": 3575, "color": "#7c3aed"},
    "Harmful Advice":           {"count": 1939, "color": "#06b6d4"},
    "Anti-Medication":          {"count": 1170, "color": "#a78bfa"},
    "Alternative Treatment":    {"count": 665,  "color": "#22d3ee"},
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter"),
    margin=dict(l=10, r=10, t=16, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(124,58,237,0.2)", borderwidth=1),
)
GRID = dict(gridcolor="#2d2b55", linecolor="#2d2b55")

# ─── HuggingFace Hub model IDs ───
MODEL_HUB_IDS = {
    "RoBERTa":    "Paulst7/roberta-mhmisinfo",
    "MentalBERT": "Paulst7/mentalbert-mhmisinfo",
}

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_sample_data():
    """Synthetic data used when CSV files are absent (e.g. Streamlit Cloud)."""
    rng = np.random.default_rng(42)

    misinfo_texts = [
        "Doctors don't want you to know this natural cure for depression.",
        "Big pharma is hiding the real treatments for mental illness.",
        "Vaccines cause autism — proven by independent research.",
        "Mental illness is fake, just think positive and you'll be fine.",
        "Natural remedies can cure schizophrenia without dangerous medication.",
        "The government is putting chemicals in water to cause mental illness.",
        "Psychiatrists are corrupt and just want to drug your children.",
        "Antidepressants are more dangerous than the depression itself.",
        "This herb cures ADHD in 30 days — doctors won't tell you.",
        "Stop taking your meds and try this simple detox instead.",
        "Mental health industry is a scam designed to control the population.",
        "They're using 5G to trigger mental illness in the population.",
        "Essential oils and diet can replace all psychiatric medications.",
        "Big pharma profits from keeping you sick — seek natural cures.",
        "Meditation alone cures bipolar disorder, no pills needed.",
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
        "Remember: asking for help is a sign of strength, not weakness.",
        "The research on this topic is really fascinating and encouraging.",
        "My anxiety improved so much with the right combination of therapy and meds.",
        "I love how this community supports each other through tough times.",
        "Six months of DBT and I barely recognise how much I've grown.",
        "Please don't suffer in silence — help is out there.",
    ]

    n = 500
    platforms = rng.choice(["Youtube", "Bitchute"], size=n, p=[0.90, 0.10])
    labels = [
        -1 if rng.random() < (0.116 if p == "Youtube" else 0.055) else 0
        for p in platforms
    ]
    texts = [
        rng.choice(misinfo_texts) if lbl == -1 else rng.choice(legit_texts)
        for lbl in labels
    ]

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
            model = RobertaForSequenceClassification.from_pretrained(hub_id, token=token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(hub_id, token=token)
            model = AutoModelForSequenceClassification.from_pretrained(hub_id, token=token)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


def clean_text(text, max_len=220):
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"http\S+", "", text).strip()
    return text[:max_len] + ("…" if len(text) > max_len else "")


# ═══════════════════════════════════════════════════════════════════
# HELPERS — reusable HTML components
# ═══════════════════════════════════════════════════════════════════
def kpi_gradient(value, label):
    return (
        f'<div class="kpi-gradient">'
        f'<p class="kpi-value-light">{value}</p>'
        f'<p class="kpi-label">{label}</p></div>'
    )


def kpi_solid(value, label):
    return (
        f'<div class="kpi-solid">'
        f'<p class="kpi-value-accent">{value}</p>'
        f'<p class="kpi-label-dark">{label}</p></div>'
    )


# ═══════════════════════════════════════════════════════════════════
# TAB RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def render_overview(fc, fv):
    total       = len(fc)
    n_misinfo   = int(fc["is_misinfo"].sum())
    misinfo_pct = n_misinfo / total * 100 if total else 0
    n_platforms = len(fc["platform"].unique())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_gradient(f"{total:,}", "Comments Analysed"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_gradient(f"{misinfo_pct:.1f}%", "Misinformation Rate"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_solid(f"{n_misinfo:,}", "Misinfo Comments"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_solid(str(n_platforms), "Platforms"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Comment Composition by Platform (%)")
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
            marker_color="#06b6d4", opacity=0.9,
            text=plat["legit_pct"].map("{:.1f}%".format), textposition="inside",
            textfont=dict(color="#ffffff"),
        ))
        fig_stack.add_trace(go.Bar(
            name="Misinformation", x=plat["platform"], y=plat["misinfo_pct"],
            marker_color="#ec4899", opacity=0.9,
            text=plat["misinfo_pct"].map("{:.1f}%".format), textposition="inside",
            textfont=dict(color="#ffffff"),
        ))
        fig_stack.update_layout(
            **CHART_LAYOUT, barmode="stack", height=300,
            xaxis=GRID,
            yaxis=dict(**GRID, title="% of Comments", range=[0, 100]),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    with col_r:
        st.markdown("#### Misinformation Rate by Platform (Full Dataset)")
        fig_rate = go.Figure(go.Bar(
            x=["YouTube", "Bitchute"],
            y=[11.6, 5.5],
            marker_color=["#06b6d4", "#7c3aed"],
            text=["11.6%", "5.5%"],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=14),
        ))
        fig_rate.update_layout(
            **CHART_LAYOUT, height=300,
            yaxis=dict(**GRID, title="Misinfo Rate (%)", range=[0, 16]),
            xaxis=GRID,
        )
        fig_rate.add_annotation(
            text="Source: Full dataset · 582,362 comments",
            xref="paper", yref="paper", x=0.5, y=-0.14,
            showarrow=False, font=dict(size=10, color="#94a3b8"),
        )
        st.plotly_chart(fig_rate, use_container_width=True)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### Comment Labels")
        fig_c = go.Figure(go.Pie(
            values=[int((~fc["is_misinfo"]).sum()), n_misinfo],
            labels=["Legitimate", "Misinformation"],
            hole=0.55,
            marker=dict(colors=["#06b6d4", "#ec4899"]),
            textinfo="label+percent",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_c.update_layout(**CHART_LAYOUT, height=260)
        st.plotly_chart(fig_c, use_container_width=True)

    with col_d2:
        st.markdown("#### Video Labels")
        fig_v = go.Figure(go.Pie(
            values=[int((~fv["is_misinfo"]).sum()), int(fv["is_misinfo"].sum())],
            labels=["Legitimate", "Misinformation"],
            hole=0.55,
            marker=dict(colors=["#7c3aed", "#ec4899"]),
            textinfo="label+percent",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_v.update_layout(**CHART_LAYOUT, height=260)
        st.plotly_chart(fig_v, use_container_width=True)


def render_model_tab(selected_model, m, badge_cls):
    other_name = "MentalBERT" if selected_model == "RoBERTa" else "RoBERTa"
    other      = MODEL_RESULTS[other_name]

    st.markdown(f"#### {selected_model} — Performance Metrics")
    st.markdown(
        f'<span class="badge {badge_cls}">{selected_model}</span> &nbsp; '
        f'<code>{m["base"]}</code> &nbsp;·&nbsp; '
        f'<small>Test set: 2,000 samples · 3 epochs · lr=2e-5</small>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Macro F1",          f"{m['macro_f1']:.2f}",
                  delta=f"{m['macro_f1'] - other['macro_f1']:+.2f} vs {other_name}")
    with mc2:
        st.metric("Accuracy",          f"{m['accuracy']:.2f}",
                  delta=f"{m['accuracy'] - other['accuracy']:+.2f} vs {other_name}")
    with mc3:
        st.metric("Misinfo Precision", f"{m['misinfo']['precision']:.2f}",
                  delta=f"{m['misinfo']['precision'] - other['misinfo']['precision']:+.2f}")
    with mc4:
        st.metric("Misinfo Recall",    f"{m['misinfo']['recall']:.2f}",
                  delta=f"{m['misinfo']['recall'] - other['misinfo']['recall']:+.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Per-Class Metrics")
        metrics   = ["Precision", "Recall", "F1-Score"]
        m_misinfo = [m["misinfo"]["precision"], m["misinfo"]["recall"], m["misinfo"]["f1"]]
        m_legit   = [m["legit"]["precision"],   m["legit"]["recall"],   m["legit"]["f1"]]

        fig_cls = go.Figure()
        fig_cls.add_trace(go.Bar(
            name="Misinformation", x=metrics, y=m_misinfo,
            marker_color="#ec4899", opacity=0.9,
            text=[f"{v:.2f}" for v in m_misinfo], textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_cls.add_trace(go.Bar(
            name="Legitimate", x=metrics, y=m_legit,
            marker_color=m["color"], opacity=0.9,
            text=[f"{v:.2f}" for v in m_legit], textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_cls.update_layout(
            **CHART_LAYOUT, barmode="group", height=330,
            xaxis=GRID,
            yaxis=dict(**GRID, range=[0, 0.96]),
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    with col_b:
        st.markdown("##### Confusion Matrix")
        cm     = m["cm"]
        z_text = [[str(v) for v in row] for row in cm]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: Misinfo", "Predicted: Legit"],
            y=["Actual: Misinfo", "Actual: Legit"],
            colorscale=[[0, "#1e1b4b"], [1, m["color"]]],
            showscale=False,
            text=z_text,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=20, color="#e2e8f0"),
        ))
        fig_cm.update_layout(
            **CHART_LAYOUT, height=330,
            xaxis=dict(**GRID, side="top"),
            yaxis=dict(**GRID, autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("##### Training Loss — All Epochs")
    fig_loss = go.Figure()
    for name, res in MODEL_RESULTS.items():
        is_sel = name == selected_model
        fig_loss.add_trace(go.Scatter(
            x=[1, 2, 3], y=res["train_loss"],
            name=name, mode="lines+markers",
            line=dict(color=res["color"], width=3 if is_sel else 1.5),
            marker=dict(size=9 if is_sel else 6),
            opacity=1.0 if is_sel else 0.3,
        ))
    fig_loss.update_layout(
        **CHART_LAYOUT, height=220,
        xaxis=dict(**GRID, title="Epoch", tickvals=[1, 2, 3]),
        yaxis=dict(**GRID, title="Training Loss"),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Side-by-Side Comparison")
    comp1, comp2 = st.columns(2)
    for col, (name, res) in zip([comp1, comp2], MODEL_RESULTS.items()):
        bc    = "badge-purple" if name == "RoBERTa" else "badge-teal"
        glow  = f"box-shadow: 0 0 20px {res['color']}33;" if name == selected_model else ""
        hl    = f"border: 1px solid {res['color']}60;" if name == selected_model else \
                "border: 1px solid rgba(124,58,237,0.2);"
        arrow = "▲ selected" if name == selected_model else ""
        with col:
            st.markdown(f"""
<div style="background:#1a1830; {hl} {glow} border-radius:12px; padding:18px;">
<span class="badge {bc}">{name}</span>
<small style="color:{res['color']}; margin-left:8px">{arrow}</small>
<br><br>
<table style="width:100%; font-size:0.85rem; color:#e2e8f0; border-collapse:collapse; line-height:2">
<tr style="color:#94a3b8; font-size:0.72rem; border-bottom:1px solid #2d2b55;">
  <td>Metric</td><td style="text-align:right">Misinfo</td><td style="text-align:right">Legit</td>
</tr>
<tr><td>Precision</td>
  <td style="text-align:right">{res['misinfo']['precision']:.2f}</td>
  <td style="text-align:right">{res['legit']['precision']:.2f}</td></tr>
<tr><td>Recall</td>
  <td style="text-align:right">{res['misinfo']['recall']:.2f}</td>
  <td style="text-align:right">{res['legit']['recall']:.2f}</td></tr>
<tr><td>F1-Score</td>
  <td style="text-align:right">{res['misinfo']['f1']:.2f}</td>
  <td style="text-align:right">{res['legit']['f1']:.2f}</td></tr>
<tr style="font-weight:700; color:{res['color']}; border-top:1px solid #2d2b55;">
  <td>Macro F1</td>
  <td style="text-align:right" colspan="2">{res['macro_f1']:.2f}</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)


def render_pattern_tab(fc):
    st.markdown("#### Misinformation Pattern Analysis")
    total_misinfo_full = 66537

    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        st.markdown("##### Category Breakdown (Full Dataset)")
        cats   = list(MISINFO_CATEGORIES.keys())
        counts = [v["count"] for v in MISINFO_CATEGORIES.values()]
        colors = [v["color"] for v in MISINFO_CATEGORIES.values()]
        pcts   = [c / total_misinfo_full * 100 for c in counts]

        fig_cats = go.Figure(go.Bar(
            x=counts, y=cats, orientation="h",
            marker_color=colors,
            text=[f"{c:,}  ({p:.1f}%)" for c, p in zip(counts, pcts)],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
        ))
        fig_cats.update_layout(
            **CHART_LAYOUT, height=310,
            xaxis=dict(**GRID, title="Number of Comments", range=[0, 8400]),
            yaxis=dict(**GRID, autorange="reversed"),
        )
        st.plotly_chart(fig_cats, use_container_width=True)

    with col_p2:
        st.markdown("##### Key Findings")
        st.markdown("""
<div class="info-box">
<b>Total misinfo (full dataset)</b><br>
<span style="font-size:1.5rem; font-weight:700; color:#ec4899">66,537</span><br>
<span style="color:#94a3b8; font-size:0.8rem">out of 582,362 comments (11.4%)</span>
</div>
<div class="info-box" style="margin-top:8px">
<b>Largest misinfo category</b><br>
Stigma / Dismissal<br>
<span style="color:#94a3b8; font-size:0.8rem">9.9% of all misinfo comments</span>
</div>
<div class="info-box" style="margin-top:8px; border-color:#06b6d4; background:rgba(6,182,212,0.08)">
<b>Medication mentions: 1.5× more</b><br>
Misinfo 1.2% vs Legit 0.8%<br>
<span style="color:#94a3b8; font-size:0.8rem">Highest cross-category ratio</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Temporal Trend — Comments Over Time (Gold Dataset)")

    valid_fc = fc.dropna(subset=["date"])
    if len(valid_fc) > 0:
        trend = (
            valid_fc.groupby(["year_month", "is_misinfo"])
            .size()
            .reset_index(name="count")
        )
        trend     = trend.sort_values("year_month")
        t_legit   = trend[~trend["is_misinfo"]]
        t_misinfo = trend[trend["is_misinfo"]]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=t_legit["year_month"], y=t_legit["count"],
            name="Legitimate", mode="lines",
            line=dict(color="#06b6d4", width=2),
            fill="tozeroy", fillcolor="rgba(6,182,212,0.08)",
        ))
        fig_trend.add_trace(go.Scatter(
            x=t_misinfo["year_month"], y=t_misinfo["count"],
            name="Misinformation", mode="lines",
            line=dict(color="#ec4899", width=2),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.10)",
        ))
        fig_trend.update_layout(
            **CHART_LAYOUT, height=250,
            xaxis=dict(**GRID, title="Month"),
            yaxis=dict(**GRID, title="Comment Count"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No valid date data for the current filter selection.")

    st.markdown("---")
    st.markdown("##### Sample Comments")
    cs1, cs2 = st.columns(2)

    with cs1:
        st.markdown(
            '<span style="color:#ec4899; font-weight:600; font-size:0.9rem">Misinformation</span>',
            unsafe_allow_html=True,
        )
        n_avail    = int(fc["is_misinfo"].sum())
        sample_mis = fc[fc["is_misinfo"]]["text"].dropna().sample(min(5, n_avail), random_state=42)
        for t in sample_mis:
            st.markdown(f'<div class="comment-card-pink">{clean_text(t)}</div>', unsafe_allow_html=True)

    with cs2:
        st.markdown(
            '<span style="color:#06b6d4; font-weight:600; font-size:0.9rem">Legitimate</span>',
            unsafe_allow_html=True,
        )
        n_legit    = int((~fc["is_misinfo"]).sum())
        sample_leg = fc[~fc["is_misinfo"]]["text"].dropna().sample(min(5, n_legit), random_state=77)
        for t in sample_leg:
            st.markdown(f'<div class="comment-card-teal">{clean_text(t)}</div>', unsafe_allow_html=True)


def render_classifier_tab(selected_model, m, badge_cls):
    tokenizer, clf_model, load_error = load_classifier(selected_model)
    model_ready = tokenizer is not None

    st.markdown(f"#### Live Classifier — {selected_model}")
    st.markdown(
        f'<span class="badge {badge_cls}">{selected_model}</span> &nbsp; '
        f'<code>{m["base"]}</code> &nbsp;·&nbsp; '
        f'<small><code>{MODEL_HUB_IDS[selected_model]}</code></small>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if not model_ready:
        if load_error:
            st.markdown(
                f'<div class="warn-box">⚠️ <b>Could not load model from Hub.</b><br>'
                f'<small style="color:#94a3b8">{load_error}</small><br><br>'
                f'If the repository is private, set the <code>HF_TOKEN</code> environment variable '
                f'to a HuggingFace token with read access, then restart the app.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warn-box">⚠️ <b>Model not loaded.</b> '
                'Set the <code>HF_TOKEN</code> environment variable and restart.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter a social media comment to classify:",
        placeholder=(
            "e.g. 'This medication is dangerous and doctors don't want you to know the truth…'\n"
            "or 'I've been in therapy for 6 months and it's genuinely helped my anxiety.'"
        ),
        height=130,
    )

    btn_col, info_col = st.columns([1, 4])
    with btn_col:
        classify = st.button(
            "Classify →",
            type="primary",
            disabled=not model_ready or not user_text.strip(),
        )
    with info_col:
        st.markdown(
            f'<small>{"Model loaded ✓" if model_ready else "Model not loaded"} · '
            'Max token length: 128 · Classes: Misinformation / Legitimate</small>',
            unsafe_allow_html=True,
        )

    if classify and user_text.strip() and model_ready:
        with st.spinner("Running inference…"):
            inputs = tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            with torch.no_grad():
                logits = clf_model(**inputs).logits
            probs     = torch.softmax(logits, dim=1)[0]
            pred      = torch.argmax(logits, dim=1).item()
            label     = "Misinformation" if pred == 0 else "Legitimate"
            confidence = probs[pred].item() * 100

        if pred == 0:
            st.markdown(
                f'<div class="warn-box" style="font-size:1rem">'
                f'<b>⚠️ {label}</b> &nbsp;·&nbsp; '
                f'<span style="color:#94a3b8">{selected_model} confidence: '
                f'<b style="color:#ec4899">{confidence:.1f}%</b></span><br>'
                f'<small style="color:#94a3b8">'
                f'Misinfo: {probs[0].item()*100:.1f}% &nbsp;·&nbsp; '
                f'Legitimate: {probs[1].item()*100:.1f}%</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-box" style="font-size:1rem">'
                f'<b>✓ {label}</b> &nbsp;·&nbsp; '
                f'<span style="color:#94a3b8">{selected_model} confidence: '
                f'<b style="color:#06b6d4">{confidence:.1f}%</b></span><br>'
                f'<small style="color:#94a3b8">'
                f'Misinfo: {probs[0].item()*100:.1f}% &nbsp;·&nbsp; '
                f'Legitimate: {probs[1].item()*100:.1f}%</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        f'<div class="info-box"><b>{selected_model} quick reference</b><br>'
        f'Macro F1: <b>{m["macro_f1"]:.2f}</b> &nbsp;·&nbsp; '
        f'Accuracy: <b>{m["accuracy"]:.2f}</b><br>'
        f'Catches <b>{int(m["misinfo"]["recall"]*100)}%</b> of real misinformation '
        f'(recall) with <b>{int(m["misinfo"]["precision"]*100)}%</b> precision</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════
comments_df, videos_df, DATA_MODE = load_data()

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 MHMisinfo")
    st.markdown("Mental Health Misinformation  \nVisual Analytics Dashboard")
    st.markdown("---")

    st.markdown("### 🤖 Model")
    selected_model = st.selectbox(
        "Select classifier",
        options=list(MODEL_RESULTS.keys()),
        index=0,
        help="All model-performance sections update to reflect the selected model.",
    )
    m = MODEL_RESULTS[selected_model]
    badge_cls = "badge-purple" if selected_model == "RoBERTa" else "badge-teal"
    st.markdown(
        f'<span class="badge {badge_cls}">{selected_model}</span> '
        f'<small style="color:#94a3b8">Macro F1 = {m["macro_f1"]:.2f}</small>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🔎 Filters")
    platform_filter = st.multiselect(
        "Platform",
        options=["Youtube", "Bitchute"],
        default=["Youtube", "Bitchute"],
    )

    st.markdown("---")
    st.markdown("### 📦 Dataset")
    if DATA_MODE == "full":
        c_rows, c_mis, c_leg = "135,445", "8,025", "127,420"
        v_rows, v_mis, v_leg = "739", "120", "619"
        c_label, v_label = "Gold Comments", "Gold Videos"
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
        f'<div class="info-box"><b>{c_label}</b><br>{c_rows} records'
        f'<br><span style="color:#94a3b8;font-size:0.82rem">{c_mis} misinfo · {c_leg} legit</span></div>'
        f'<div class="info-box" style="border-color:#06b6d4;background:rgba(6,182,212,0.08);margin-top:6px">'
        f'<b>{v_label}</b><br>{v_rows} records'
        f'<br><span style="color:#94a3b8;font-size:0.82rem">{v_mis} misinfo · {v_leg} legit</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("PRT840 IT Thesis · CDU · 2026\nPaul S. Flores Sinche (S386377)")

# ═══════════════════════════════════════════════════════════════════
# FILTERS + HEADER
# ═══════════════════════════════════════════════════════════════════
fc = comments_df[comments_df["platform"].isin(platform_filter)] if platform_filter else comments_df
fv = videos_df[videos_df["platform"].isin(platform_filter)]     if platform_filter else videos_df

st.markdown("## AI-Driven Visual Analytics")
st.markdown(
    "**Mental Health Misinformation on Social Media** &nbsp;·&nbsp; "
    "MHMisinfo Dataset &nbsp;·&nbsp; RoBERTa & MentalBERT",
    unsafe_allow_html=True,
)

if DATA_MODE == "sample":
    st.info(
        "**Stratified sample** — showing a representative 1,000-comment / 200-video sample "
        "drawn from the real MHMisinfo Gold dataset. Class proportions reflect the full dataset.",
        icon="📊",
    )
elif DATA_MODE == "synthetic":
    st.warning(
        "**Demo mode** — sample CSV files were not found. "
        "Displaying synthetic data for illustration purposes only. "
        "Run locally with the CSV files present for real statistics.",
        icon="⚠️",
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🤖  Model Performance",
    "🔍  Pattern Analysis",
    "⚡  Live Classifier",
])

with tab1:
    render_overview(fc, fv)

with tab2:
    render_model_tab(selected_model, m, badge_cls)

with tab3:
    render_pattern_tab(fc)

with tab4:
    render_classifier_tab(selected_model, m, badge_cls)
