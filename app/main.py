"""
Bank Marketing — Stakeholder Intelligence Platform v5
======================================================
Lightning-fast, single-model dashboard with role-based views.
Loads ONLY the best model (composite-selected) for instant predictions.

Stakeholder Tabs:
  1. Executive Summary     — C-suite KPIs, ROI, banking economics
  2. Call Centre Ops       — Who/when/how to call, segment guide
  3. Model & Data Science  — Technical deep-dive, SHAP, calibration
  4. Predict Client        — Single client YES/NO with factors
  5. Batch Predict         — File upload / URL bulk scoring

Usage: streamlit run app/main.py
"""

from __future__ import annotations
import sys, io, json, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
MOD_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MET_DIR = ROOT / "reports" / "metrics"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bank Marketing Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# FAANG-LEVEL CSS — Glassmorphism, Gradients, Micro-animations
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Reset & Global ──────────────────────────────────────────────── */
.block-container { padding: 1rem 2rem 2rem 2rem; max-width: 1400px; }
h1, h2, h3 { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* ── Metric Cards ────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 800; color: #1e293b; }

/* ── Hero ────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%);
    color: white;
    padding: 2.2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 1.8rem;
    box-shadow: 0 12px 40px rgba(15,23,42,0.25);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(14,165,233,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; position: relative; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.8; font-size: 0.95rem; position: relative; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 8px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* ── Glass Cards ─────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}
.glass-card:hover { transform: translateY(-1px); }
.glass-card h4 { margin: 0 0 0.6rem 0; font-weight: 700; color: #1e293b; font-size: 1rem; }
.glass-card .big { font-size: 2.8rem; font-weight: 900; line-height: 1; }
.glass-card .sub { color: #64748b; font-size: 0.82rem; margin-top: 4px; }

/* ── Verdict Banners ─────────────────────────────────────────────── */
.verdict-yes {
    background: linear-gradient(135deg, #059669, #34d399);
    color: white; padding: 2rem; border-radius: 20px; text-align: center;
    box-shadow: 0 8px 32px rgba(5,150,105,0.3);
}
.verdict-no {
    background: linear-gradient(135deg, #dc2626, #f87171);
    color: white; padding: 2rem; border-radius: 20px; text-align: center;
    box-shadow: 0 8px 32px rgba(220,38,38,0.3);
}
.verdict-label { font-size: 2.2rem; font-weight: 900; margin: 0; }
.verdict-sub { font-size: 1rem; opacity: 0.9; margin: 0.3rem 0 0 0; }
.verdict-prob { font-size: 3.5rem; font-weight: 900; margin: 0.5rem 0; }

/* ── Factor Pills ────────────────────────────────────────────────── */
.factor-pos {
    display: inline-block; background: #dcfce7; color: #166534;
    padding: 6px 14px; border-radius: 20px; font-size: 0.85rem;
    font-weight: 600; margin: 3px; border: 1px solid #bbf7d0;
}
.factor-neg {
    display: inline-block; background: #fee2e2; color: #991b1b;
    padding: 6px 14px; border-radius: 20px; font-size: 0.85rem;
    font-weight: 600; margin: 3px; border: 1px solid #fecaca;
}
.factor-neutral {
    display: inline-block; background: #f1f5f9; color: #475569;
    padding: 6px 14px; border-radius: 20px; font-size: 0.85rem;
    font-weight: 600; margin: 3px; border: 1px solid #e2e8f0;
}

/* ── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 2px; background: #f1f5f9; border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 20px; font-weight: 600;
    font-size: 0.88rem;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}

/* ── Section Headers ─────────────────────────────────────────────── */
.section-header {
    font-size: 1.2rem; font-weight: 800; color: #1e293b;
    padding-bottom: 0.5rem; margin: 1.5rem 0 0.8rem 0;
    border-bottom: 3px solid #0ea5e9;
    display: inline-block;
}

/* ── Hide streamlit defaults ─────────────────────────────────────── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
DROP_COLS = ["duration", "y", "deposit"]


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD BEST MODEL ONLY (lightning fast)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_best_model():
    """Load ONLY the composite-selected best model. No wasted memory."""
    meta = {}
    if (MOD_DIR / "threshold.json").exists():
        with open(MOD_DIR / "threshold.json") as f:
            meta = json.load(f)

    name = meta.get("model", "")
    threshold = meta.get("threshold", 0.30)
    path = MOD_DIR / f"{name}.joblib"

    if not path.exists():
        # Fallback: find any available model
        for fallback in [
            "xgboost",
            "lightgbm",
            "random_forest",
            "logistic_regression",
            "voting_ensemble",
        ]:
            p = MOD_DIR / f"{fallback}.joblib"
            if p.exists():
                name, path = fallback, p
                break

    if not path.exists():
        return None, None, 0.30, {}

    model = joblib.load(path)
    return model, name, threshold, meta


@st.cache_data
def load_comparison():
    p = MET_DIR / "comparison.json"
    return pd.read_json(p) if p.exists() else None


@st.cache_data
def load_recall_analysis():
    p = MET_DIR / "recall_analysis.csv"
    return pd.read_csv(p) if p.exists() else None


model, model_name, threshold, meta = load_best_model()
comp_df = load_comparison()
recall_df = load_recall_analysis()
is_ensemble = isinstance(model, dict)

if model is None:
    st.error("No models found. Run `python scripts/run_pipeline.py` first.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def predict_one(df):
    """Predict probability for a single row DataFrame."""
    if is_ensemble:
        return float(
            model["voter"].predict_proba(model["preprocess"].transform(df))[0, 1]
        )
    return float(model.predict_proba(df)[0, 1])


def predict_batch(df):
    """Predict probabilities for a batch DataFrame."""
    if is_ensemble:
        return model["voter"].predict_proba(model["preprocess"].transform(df))[:, 1]
    return model.predict_proba(df)[:, 1]


def get_verdict(prob):
    if prob >= threshold:
        return (
            True,
            "YES — Subscribe",
            f"This client has a {prob:.0%} probability of subscribing to a term deposit.",
        )
    return (
        False,
        "NO — Will Not Subscribe",
        f"This client has only a {prob:.0%} probability. Below our {threshold:.0%} threshold.",
    )


def get_factors(row, prob):
    """Generate human-readable factor pills for the prediction."""
    pos, neg = [], []
    if row.get("poutcome") == "success":
        pos.append("Previous campaign: Success")
    elif row.get("poutcome") == "failure":
        neg.append("Previous campaign: Failed")
    if row.get("contact") == "cellular":
        pos.append("Mobile contact")
    else:
        neg.append("Landline contact")
    if row.get("euribor3m", 5) < 2:
        pos.append("Low interest rates")
    elif row.get("euribor3m", 0) > 4:
        neg.append("High interest rates")
    if row.get("job") in ("retired", "student"):
        pos.append(f"{row['job'].title()} — high-value segment")
    if row.get("month") in ("mar", "sep", "oct", "dec"):
        pos.append(f"Strong month ({row['month'].title()})")
    if row.get("campaign", 0) > 5:
        neg.append(f"{row['campaign']} calls — diminishing returns")
    elif row.get("campaign", 99) <= 2:
        pos.append("Low call count — fresh lead")
    if row.get("age", 0) > 60:
        pos.append("Senior client (60+)")
    elif row.get("age", 0) < 30:
        pos.append("Young client (<30)")
    p = row.get("pdays", 999)
    if p < 30 and p != 999:
        pos.append("Recently contacted")
    if row.get("housing") == "no" and row.get("loan") == "no":
        pos.append("No existing loans")
    nr = row.get("nr.employed", 5200)
    if nr < 5100:
        pos.append("Low employment → receptive market")
    return pos, neg


def validate_df(df):
    df.columns = df.columns.str.strip().str.lower()
    renames = {
        "dayofweek": "day_of_week",
        "emp_var_rate": "emp.var.rate",
        "cons_price_idx": "cons.price.idx",
        "cons_conf_idx": "cons.conf.idx",
        "nr_employed": "nr.employed",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    warns = []
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
            warns.append(f"Auto-dropped `{c}`")
    extra = [c for c in df.columns if c not in FEATURES]
    if extra:
        df = df.drop(columns=extra)
        warns.append(f"Dropped extra: {extra}")
    missing = [c for c in FEATURES if c not in df.columns]
    return df, warns, missing


# ═══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════════════════════
criterion = meta.get("selection_criterion", "composite_score")
composite = ""
if comp_df is not None:
    row = comp_df[comp_df["model"] == model_name]
    if not row.empty and "composite_score" in row.columns:
        composite = f" · Score: {row.iloc[0]['composite_score']:.3f}"

st.markdown(
    f"""
<div class="hero">
    <h1>🏦 Bank Marketing Intelligence</h1>
    <p>
        <span class="badge">Model: {model_name.replace('_', ' ').title()}</span>
        <span class="badge">Threshold: {threshold:.1%}</span>
        <span class="badge">Selection: {criterion.replace('_', ' ').title()}{composite}</span>
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_exec, tab_cc, tab_ds, tab_pred, tab_batch = st.tabs(
    [
        "📊 Executive Summary",
        "📞 Call Centre Operations",
        "🔬 Model & Data Science",
        "🎯 Predict Client",
        "📁 Batch Predict",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_exec:
    if comp_df is not None:
        best_row = comp_df[comp_df["model"] == model_name]
        if not best_row.empty:
            br = best_row.iloc[0]
            baseline = 6179 * 0.113 * 195 - 6179 * 0.887 * 5
            profit = br.get("net_profit", 0)
            roi = (profit - baseline) / abs(baseline) * 100 if baseline != 0 else 0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "ML Net Profit",
                f"${profit:,.0f}",
                delta=f"+${profit - baseline:,.0f} vs baseline",
            )
            k2.metric("ROI vs No-ML", f"{roi:+.0f}%")
            k3.metric("Subscribers Caught", f"{br.get('test_recall', 0):.0%}")
            k4.metric("Model AUC", f"{br.get('test_roc_auc', 0):.1%}")

    # ── Banking Economics ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">💰 Banking Economics — Why This Matters</div>',
        unsafe_allow_html=True,
    )

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown(
            """
<div class="glass-card">
    <h4>How Banks Earn From Term Deposits</h4>
    <table style="width:100%; font-size:0.9rem; border-collapse: collapse;">
        <tr><td style="padding:6px 0;">Avg term deposit</td><td style="text-align:right;font-weight:700;">$10,000</td></tr>
        <tr><td style="padding:6px 0;">Bank pays depositor</td><td style="text-align:right;font-weight:700;">2.0%</td></tr>
        <tr><td style="padding:6px 0;">Bank lends funds at</td><td style="text-align:right;font-weight:700;">8.4%</td></tr>
        <tr style="border-top:2px solid #0ea5e9;"><td style="padding:6px 0;"><b>Net Interest Margin</b></td><td style="text-align:right;font-weight:800;color:#0ea5e9;">6.4%</td></tr>
        <tr><td style="padding:6px 0;">Annual NIM revenue</td><td style="text-align:right;font-weight:700;">$640 / subscriber</td></tr>
        <tr><td style="padding:6px 0;">Over 2-year term</td><td style="text-align:right;font-weight:700;">$1,280 lifetime</td></tr>
    </table>
</div>""",
            unsafe_allow_html=True,
        )

    with ec2:
        st.markdown(
            """
<div class="glass-card">
    <h4>Cost of Prediction Errors</h4>
    <table style="width:100%; font-size:0.9rem; border-collapse: collapse;">
        <tr><td style="padding:6px 0;">❌ False Positive (wasted call)</td><td style="text-align:right;font-weight:700;">$5</td></tr>
        <tr><td style="padding:6px 0;">🚨 False Negative (missed subscriber)</td><td style="text-align:right;font-weight:700;color:#dc2626;">$200</td></tr>
        <tr style="border-top:2px solid #dc2626;"><td style="padding:6px 0;"><b>Cost ratio: FN/FP</b></td><td style="text-align:right;font-weight:800;color:#dc2626;">40:1</td></tr>
    </table>
    <div class="sub" style="margin-top:10px;">Missing a subscriber is <b>40× costlier</b> than a wasted call.<br>Our model is tuned to minimize missed opportunities.</div>
</div>""",
            unsafe_allow_html=True,
        )

    # ── Strategy Comparison ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📊 Strategy Comparison — Before vs After ML</div>',
        unsafe_allow_html=True,
    )

    if comp_df is not None:
        strategies = {
            "Call Nobody\n(Zero Cost)": 0,
            "Call Everyone\n(Current)": int(baseline),
        }
        for _, r in comp_df.iterrows():
            strategies[f"ML:\n{r['model']}"] = int(r["net_profit"])

        fig = go.Figure()
        names = list(strategies.keys())
        vals = list(strategies.values())
        best_ml = max(vals[2:]) if len(vals) > 2 else 0
        colors = ["#94a3b8", "#ef4444"] + [
            "#0ea5e9" if v == best_ml else "#64748b" for v in vals[2:]
        ]
        fig.add_trace(
            go.Bar(
                x=names,
                y=vals,
                marker_color=colors,
                text=[f"${v:,.0f}" for v in vals],
                textposition="outside",
                textfont=dict(size=12, color="#1e293b"),
            )
        )
        fig.update_layout(
            title=dict(
                text="Net Profit by Strategy (Test: 6,179 clients)", font=dict(size=16)
            ),
            yaxis_title="Net Profit ($)",
            template="plotly_white",
            height=420,
            margin=dict(t=60, b=60, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Model Selection Transparency ─────────────────────────────────────
    st.markdown(
        '<div class="section-header">🧠 Why This Model Was Selected</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="glass-card">
    <h4>Composite Weighted Scoring</h4>
    <div class="sub">Single metrics are fragile. We score across 4 dimensions:</div>
    <table style="width:100%; font-size:0.9rem; margin-top:10px; border-collapse: collapse;">
        <tr style="background:#f1f5f9;"><th style="padding:8px;text-align:left;">Criterion</th><th style="text-align:center;">Weight</th><th style="text-align:left;">Rationale</th></tr>
        <tr><td style="padding:8px;">💰 Net Profit</td><td style="text-align:center;font-weight:700;">40%</td><td>Business outcome — the whole point</td></tr>
        <tr><td style="padding:8px;">🎯 Recall</td><td style="text-align:center;font-weight:700;">25%</td><td>FN costs 40× more — don't miss subscribers</td></tr>
        <tr><td style="padding:8px;">📈 AUC-ROC</td><td style="text-align:center;font-weight:700;">20%</td><td>Overall discrimination ability</td></tr>
        <tr><td style="padding:8px;">🎯 Calibration</td><td style="text-align:center;font-weight:700;">15%</td><td>Can we trust the probabilities?</td></tr>
    </table>
</div>""",
        unsafe_allow_html=True,
    )

    img = FIG_DIR / "13d_composite_selection.png"
    if img.exists():
        st.image(
            str(img),
            caption="Normalized scores across all models — blue highlight = winner",
        )

    # ── Sensitivity check ────────────────────────────────────────────────
    sensitivity = meta.get("sensitivity", {})
    if sensitivity:
        unique = set(sensitivity.values())
        if len(unique) == 1:
            st.success(
                f"✅ **Selection is ROBUST** — `{model_name}` wins under ALL {len(sensitivity)} criteria tested"
            )
        else:
            st.warning(
                f"Selection varies across criteria ({len(unique)} different winners). Composite balances trade-offs."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CALL CENTRE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_cc:
    st.markdown(
        '<div class="section-header">📞 Campaign Transformation — Before vs After ML</div>',
        unsafe_allow_html=True,
    )

    if comp_df is not None:
        best_row = (
            comp_df[comp_df["model"] == model_name].iloc[0]
            if not comp_df[comp_df["model"] == model_name].empty
            else comp_df.iloc[0]
        )
        n_test = 6179
        tp_rate = best_row.get("test_recall", 0.99)
        prec = best_row.get("test_precision", 0.12)
        n_calls = int(tp_rate * 113 / prec) if prec > 0 else 1000
        n_tp = int(tp_rate * 113)
        n_fp = n_calls - n_tp
        calls_pct = n_calls / 10

        ba1, ba2 = st.columns(2)
        with ba1:
            st.markdown(
                f"""
<div class="glass-card" style="border-left: 4px solid #ef4444;">
    <h4>❌ Before — Call Everyone</h4>
    <div class="big" style="color:#ef4444;">1,000</div>
    <div class="sub">calls per 1,000 clients</div>
    <table style="width:100%; font-size:0.88rem; margin-top:12px;">
        <tr><td>Conversions</td><td style="text-align:right;font-weight:700;">113 (11.3%)</td></tr>
        <tr><td>Wasted calls</td><td style="text-align:right;font-weight:700;color:#ef4444;">887</td></tr>
        <tr style="border-top:2px solid #1e293b;"><td><b>Net Profit</b></td><td style="text-align:right;font-weight:700;">$17,600</td></tr>
    </table>
</div>""",
                unsafe_allow_html=True,
            )
        with ba2:
            scaled_profit = best_row.get("net_profit", 0) / n_test * 1000
            st.markdown(
                f"""
<div class="glass-card" style="border-left: 4px solid #059669;">
    <h4>✅ After — ML-Targeted</h4>
    <div class="big" style="color:#059669;">{n_calls}</div>
    <div class="sub">calls per 1,000 clients ({calls_pct:.0f}% of list)</div>
    <table style="width:100%; font-size:0.88rem; margin-top:12px;">
        <tr><td>Conversions caught</td><td style="text-align:right;font-weight:700;">{n_tp} ({tp_rate:.0%} recall)</td></tr>
        <tr><td>Wasted calls</td><td style="text-align:right;font-weight:700;color:#059669;">{n_fp}</td></tr>
        <tr style="border-top:2px solid #1e293b;"><td><b>Net Profit</b></td><td style="text-align:right;font-weight:700;color:#059669;">${scaled_profit:,.0f}</td></tr>
    </table>
</div>""",
                unsafe_allow_html=True,
            )

    # ── Priority Rules ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🎯 Who Should We Call?</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    cats = [
        "Prev Success",
        "Students",
        "Retirees",
        "Mobile",
        "Mar/Sep/Oct",
        "≤3 Calls",
        "Low Euribor",
        "Others",
    ]
    rates = [65, 31, 25, 15, 18, 14, 20, 8]
    colors = [
        "#059669" if c > 15 else "#f59e0b" if c > 10 else "#ef4444" for c in rates
    ]
    fig.add_trace(
        go.Bar(
            x=cats,
            y=rates,
            marker_color=colors,
            text=[f"{c}%" for c in rates],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=11.3,
        line_dash="dash",
        line_color="red",
        annotation_text="Avg: 11.3%",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Conversion Rate by Segment",
        yaxis_title="Conversion %",
        template="plotly_white",
        height=380,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-header">📋 Call Priority Rules</div>',
        unsafe_allow_html=True,
    )
    rules = pd.DataFrame(
        [
            {
                "Priority": "🔴 1 — MUST CALL",
                "Segment": "Previous campaign subscribers",
                "Conv Rate": "~65%",
                "Action": "Call first, always",
            },
            {
                "Priority": "🟠 2 — HIGH",
                "Segment": "Students + Retirees via mobile",
                "Conv Rate": "25–31%",
                "Action": "Call during Mar/Sep/Oct",
            },
            {
                "Priority": "🟡 3 — MEDIUM",
                "Segment": "ML score > threshold",
                "Conv Rate": "15–20%",
                "Action": "Call if capacity allows",
            },
            {
                "Priority": "🟢 4 — LOW",
                "Segment": "ML score < threshold",
                "Conv Rate": "<10%",
                "Action": "Email instead",
            },
            {
                "Priority": "⛔ 5 — STOP",
                "Segment": "5+ calls already made",
                "Conv Rate": "<5%",
                "Action": "Diminishing returns",
            },
        ]
    )
    st.dataframe(rules, use_container_width=True, hide_index=True)

    # ── Recall Trade-off Table ───────────────────────────────────────────
    if recall_df is not None:
        st.markdown(
            '<div class="section-header">📈 Threshold vs Catch Rate — What Happens If We Adjust?</div>',
            unsafe_allow_html=True,
        )
        display = recall_df[
            recall_df["threshold"].isin([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
        ].copy()
        display["recall"] = display["recall"].apply(lambda x: f"{x:.0%}")
        display["precision"] = display["precision"].apply(lambda x: f"{x:.0%}")
        display["net_profit"] = display["net_profit"].apply(lambda x: f"${x:,.0f}")
        display = display.rename(
            columns={
                "threshold": "Threshold",
                "recall": "Subscribers Caught",
                "precision": "Call Precision",
                "calls_made": "Calls Made",
                "subscribers_missed": "Missed",
                "net_profit": "Net Profit",
            }
        )
        st.dataframe(
            display[
                [
                    "Threshold",
                    "Subscribers Caught",
                    "Call Precision",
                    "Calls Made",
                    "Missed",
                    "Net Profit",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ── Monthly Strategy ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📅 When to Call</div>', unsafe_allow_html=True
    )
    for img_name in ["04_monthly_patterns.png", "05_contact_method.png"]:
        p = FIG_DIR / img_name
        if p.exists():
            st.image(str(p))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL & DATA SCIENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_ds:
    st.markdown(
        '<div class="section-header">📋 Model Comparison</div>', unsafe_allow_html=True
    )

    if comp_df is not None:
        display_cols = [
            "model",
            "test_roc_auc",
            "test_pr_auc",
            "test_recall",
            "test_precision",
            "test_f1",
            "brier_score",
            "net_profit",
        ]
        if "composite_score" in comp_df.columns:
            display_cols.append("composite_score")
        if "overall_rank" in comp_df.columns:
            display_cols.append("overall_rank")

        avail = [c for c in display_cols if c in comp_df.columns]
        disp = comp_df[avail].copy()
        rename = {
            "model": "Model",
            "test_roc_auc": "AUC-ROC",
            "test_pr_auc": "PR-AUC",
            "test_recall": "Recall",
            "test_precision": "Precision",
            "test_f1": "F1",
            "brier_score": "Brier ↓",
            "net_profit": "Profit ($)",
            "composite_score": "Composite",
            "overall_rank": "Rank",
        }
        disp = disp.rename(columns=rename)
        num_cols = [c for c in disp.columns if c not in ("Model", "Rank")]

        styled = disp.style
        for c in num_cols:
            if c == "Brier ↓":
                styled = styled.highlight_min(axis=0, subset=[c], color="#dcfce7")
            elif c != "Profit ($)":
                styled = styled.highlight_max(axis=0, subset=[c], color="#dcfce7")

        styled = styled.format(
            {c: "{:.4f}" for c in num_cols if c not in ("Profit ($)",)}
        )
        if "Profit ($)" in disp.columns:
            styled = styled.format({"Profit ($)": "${:,.0f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── SHAP ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🔬 Feature Impact — SHAP Analysis</div>',
        unsafe_allow_html=True,
    )
    for img_name, cap in [
        ("17_business_feature_importance.png", "Business-friendly feature importance"),
        ("14_shap_summary.png", "SHAP beeswarm — red = high value → right = subscribe"),
    ]:
        p = FIG_DIR / img_name
        if p.exists():
            st.image(str(p), caption=cap)

    # ── Data Overview ────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📊 Data Quality & Distribution</div>',
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    with d1:
        p = FIG_DIR / "01_target_distribution.png"
        if p.exists():
            st.image(str(p), caption="89% vs 11% class imbalance")
    with d2:
        p = FIG_DIR / "08_unknown_values.png"
        if p.exists():
            st.image(str(p), caption="Missing data profile")

    # Non-linear relationships (NEW)
    p = FIG_DIR / "08b_nonlinear_relationships.png"
    if p.exists():
        st.image(
            str(p),
            caption="Non-linear patterns justify our binning strategy for age, campaign & euribor3m",
        )

    # ── Model Diagnostics ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📈 Model Diagnostics</div>', unsafe_allow_html=True
    )
    d1, d2 = st.columns(2)
    with d1:
        p = FIG_DIR / "11_roc_pr_curves.png"
        if p.exists():
            st.image(str(p))
    with d2:
        p = FIG_DIR / "10_confusion_matrices.png"
        if p.exists():
            st.image(str(p))

    d1, d2 = st.columns(2)
    with d1:
        p = FIG_DIR / "13_threshold_sensitivity.png"
        if p.exists():
            st.image(str(p), caption="Threshold optimized for profit")
    with d2:
        p = FIG_DIR / "13b_calibration_curves.png"
        if p.exists():
            st.image(str(p), caption="Probability calibration quality")

    p = FIG_DIR / "09b_feature_count_sweep.png"
    if p.exists():
        st.image(str(p), caption="Most signal from top 15-20 features")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PREDICT CLIENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown(
        f"""
<div class="glass-card" style="padding: 1rem 1.5rem;">
    <b>Model:</b> {model_name.replace('_',' ').title()} &nbsp;·&nbsp;
    <b>Threshold:</b> {threshold:.1%} &nbsp;·&nbsp;
    <b>Above threshold → YES (Call)</b> &nbsp;·&nbsp; <b>Below → NO (Skip)</b>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Smart Form: 3 grouped sections ───────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**👤 Client Profile**")
        age = st.slider("Age", 17, 98, 40, key="p_age")
        job = st.selectbox(
            "Job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
            ],
            key="p_job",
        )
        marital = st.selectbox(
            "Marital", ["married", "single", "divorced"], key="p_mar"
        )
        education = st.selectbox(
            "Education",
            [
                "basic.4y",
                "basic.6y",
                "basic.9y",
                "high.school",
                "professional.course",
                "university.degree",
                "illiterate",
                "unknown",
            ],
            key="p_edu",
        )
        default = st.selectbox("Credit Default", ["no", "unknown", "yes"], key="p_def")
        housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"], key="p_hou")
        loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"], key="p_loan")

    with c2:
        st.markdown("**📞 Campaign Details**")
        contact = st.selectbox("Contact Method", ["cellular", "telephone"], key="p_con")
        month = st.selectbox(
            "Month",
            [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ],
            index=4,
            key="p_mon",
        )
        day = st.selectbox("Day", ["mon", "tue", "wed", "thu", "fri"], key="p_day")
        campaign = st.number_input("Calls This Campaign", 1, 50, 2, key="p_camp")
        pdays = st.number_input(
            "Days Since Last Contact",
            0,
            999,
            999,
            help="999 = never contacted",
            key="p_pdays",
        )
        previous = st.number_input("Previous Contacts", 0, 50, 0, key="p_prev")
        poutcome = st.selectbox(
            "Previous Outcome", ["nonexistent", "failure", "success"], key="p_pout"
        )

    with c3:
        st.markdown("**📈 Economic Indicators**")
        emp_var = st.number_input(
            "Emp Variation Rate", -4.0, 2.0, 1.1, step=0.1, key="p_emp"
        )
        cpi = st.number_input(
            "Consumer Price Idx", 90.0, 96.0, 93.994, step=0.1, key="p_cpi"
        )
        cci = st.number_input(
            "Consumer Confidence", -55.0, -20.0, -36.4, step=0.1, key="p_cci"
        )
        euribor = st.number_input("Euribor 3m", 0.0, 6.0, 4.857, step=0.1, key="p_eur")
        nr_emp = st.number_input(
            "Nr Employed", 4800.0, 5400.0, 5191.0, step=10.0, key="p_nr"
        )

    # ── Predict Button ───────────────────────────────────────────────────
    st.markdown("")
    if st.button(
        "🔮  Run Prediction", type="primary", use_container_width=True, key="run_pred"
    ):
        inp_dict = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp.var.rate": emp_var,
            "cons.price.idx": cpi,
            "cons.conf.idx": cci,
            "euribor3m": euribor,
            "nr.employed": nr_emp,
        }
        inp_df = pd.DataFrame([inp_dict])

        try:
            prob = predict_one(inp_df)
            is_yes, verdict_text, explanation = get_verdict(prob)
            pos_factors, neg_factors = get_factors(inp_dict, prob)

            # ── YES / NO Verdict ─────────────────────────────────────────
            css = "verdict-yes" if is_yes else "verdict-no"
            icon = "✅" if is_yes else "❌"
            st.markdown(
                f"""
<div class="{css}">
    <p class="verdict-label">{icon} {verdict_text}</p>
    <p class="verdict-prob">{prob:.1%}</p>
    <p class="verdict-sub">{explanation}</p>
</div>""",
                unsafe_allow_html=True,
            )

            st.markdown("")

            # ── Metrics Row ──────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability", f"{prob:.1%}")
            m2.metric("Threshold", f"{threshold:.1%}")
            margin = prob - threshold
            m3.metric("Margin", f"{margin:+.1%}", delta=f"{margin:+.1%}")

            # ── Factor Pills ─────────────────────────────────────────────
            st.markdown(
                '<div class="section-header">🔍 Key Factors</div>',
                unsafe_allow_html=True,
            )

            pills_html = ""
            for f in pos_factors:
                pills_html += f'<span class="factor-pos">✅ {f}</span>'
            for f in neg_factors:
                pills_html += f'<span class="factor-neg">⚠️ {f}</span>'
            if not pos_factors and not neg_factors:
                pills_html = (
                    '<span class="factor-neutral">No strong signals detected</span>'
                )

            st.markdown(
                f'<div style="margin-bottom:1rem;">{pills_html}</div>',
                unsafe_allow_html=True,
            )

            # ── Gauge ────────────────────────────────────────────────────
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {
                            "color": "#059669" if is_yes else "#dc2626",
                            "thickness": 0.3,
                        },
                        "steps": [
                            {"range": [0, threshold * 100], "color": "#fee2e2"},
                            {"range": [threshold * 100, 100], "color": "#dcfce7"},
                        ],
                        "threshold": {
                            "line": {"color": "#1e293b", "width": 3},
                            "value": threshold * 100,
                        },
                    },
                    number={"suffix": "%", "font": {"size": 36}},
                )
            )
            fig.update_layout(height=250, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(
        f"""
<div class="glass-card" style="padding:1rem 1.5rem;">
    Upload a CSV/Excel file or provide a URL. The system will score each client
    and return <b>YES</b> (call) or <b>NO</b> (skip) with probabilities.
</div>""",
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Data Source", ["📁 File Upload", "🌐 URL"], horizontal=True, key="batch_src"
    )

    df_raw = None

    if source == "📁 File Upload":
        uploaded = st.file_uploader(
            "Upload client data", type=["csv", "xlsx", "xls"], key="fu2"
        )
        if uploaded:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df_raw = pd.read_excel(uploaded)
                else:
                    content = uploaded.read().decode("utf-8")
                    uploaded.seek(0)
                    sep = (
                        ";"
                        if ";" in content.split("\n")[0]
                        and "," not in content.split("\n")[0]
                        else ","
                    )
                    df_raw = pd.read_csv(io.StringIO(content), sep=sep)
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    else:
        url = st.text_input(
            "Dataset URL",
            placeholder="https://archive.ics.uci.edu/ml/...",
            key="url_in2",
        )
        if url and st.button("⬇️ Download", key="dl_btn"):
            try:
                with st.spinner("Downloading..."):
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    raw = resp.content
                    if url.endswith(".zip") or "zip" in resp.headers.get(
                        "Content-Type", ""
                    ):
                        zf = zipfile.ZipFile(io.BytesIO(raw))
                        csvs = sorted(
                            [
                                f
                                for f in zf.namelist()
                                if f.endswith(".csv") and "__MACOSX" not in f
                            ],
                            key=lambda f: zf.getinfo(f).file_size,
                            reverse=True,
                        )
                        if csvs:
                            chosen = (
                                st.selectbox("Select CSV:", csvs)
                                if len(csvs) > 1
                                else csvs[0]
                            )
                            content = zf.open(chosen).read().decode("utf-8")
                        else:
                            st.error("No CSV found in ZIP")
                            content = None
                    else:
                        content = raw.decode("utf-8")
                    if content:
                        sep = (
                            ";"
                            if ";" in content.split("\n")[0]
                            and "," not in content.split("\n")[0]
                            else ","
                        )
                        df_raw = pd.read_csv(io.StringIO(content), sep=sep)
            except Exception as e:
                st.error(f"Download failed: {e}")

    if df_raw is not None:
        st.success(
            f"Loaded **{len(df_raw):,}** rows × **{len(df_raw.columns)}** columns"
        )
        with st.expander("Preview data"):
            st.dataframe(df_raw.head(5), use_container_width=True, hide_index=True)

        df_clean, warns, missing = validate_df(df_raw)
        for w in warns:
            st.warning(w)

        if missing:
            st.error(f"Missing required columns: {missing}")
        elif st.button(
            "🔮  Score All Clients",
            type="primary",
            use_container_width=True,
            key="batch_pred",
        ):
            with st.spinner(f"Scoring {len(df_clean):,} clients..."):
                probs = predict_batch(df_clean)

            result = df_clean.copy()
            result.insert(0, "probability", np.round(probs, 4))
            result.insert(1, "verdict", np.where(probs >= threshold, "✅ YES", "❌ NO"))
            result = result.sort_values("probability", ascending=False).reset_index(
                drop=True
            )

            total = len(result)
            n_yes = (probs >= threshold).sum()
            n_no = total - n_yes
            avg_p = probs.mean()

            # ── KPIs ─────────────────────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Clients", f"{total:,}")
            k2.metric("✅ YES (Call)", f"{n_yes:,}", delta=f"{n_yes/total:.1%}")
            k3.metric("❌ NO (Skip)", f"{n_no:,}")
            k4.metric("Avg Probability", f"{avg_p:.1%}")

            # ── Distribution ─────────────────────────────────────────────
            fig = px.histogram(
                result,
                x="probability",
                nbins=50,
                color_discrete_sequence=["#0ea5e9"],
                title="Probability Distribution",
            )
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:.1%}",
            )
            fig.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig, use_container_width=True)

            # ── Profit Estimate ──────────────────────────────────────────
            est_tp = n_yes * avg_p
            est_fp = n_yes * (1 - avg_p)
            est_fn = n_no * 0.113
            ml_profit = est_tp * 195 - est_fp * 5 - est_fn * 200
            all_profit = total * 0.113 * 195 - total * 0.887 * 5

            p1, p2, p3 = st.columns(3)
            p1.metric("ML Profit (est)", f"${ml_profit:,.0f}")
            p2.metric("Call-Everyone Profit", f"${all_profit:,.0f}")
            p3.metric(
                "ML Advantage",
                f"${ml_profit - all_profit:,.0f}",
                delta=f"+{(ml_profit/all_profit-1)*100:.0f}%" if all_profit > 0 else "",
            )

            # ── Top Leads ────────────────────────────────────────────────
            st.markdown(
                '<div class="section-header">🏆 Top 20 Leads</div>',
                unsafe_allow_html=True,
            )
            top = result.head(20).copy()
            top["probability"] = top["probability"].apply(lambda x: f"{x:.1%}")
            st.dataframe(top, use_container_width=True, hide_index=True)

            with st.expander(f"View all {total:,} predictions"):
                full = result.copy()
                full["probability"] = full["probability"].apply(lambda x: f"{x:.1%}")
                st.dataframe(full, use_container_width=True, hide_index=True)

            # ── Downloads ────────────────────────────────────────────────
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️  Download CSV",
                    result.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with d2:
                buf = io.BytesIO()
                result.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "⬇️  Download Excel",
                    buf.getvalue(),
                    "predictions.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
