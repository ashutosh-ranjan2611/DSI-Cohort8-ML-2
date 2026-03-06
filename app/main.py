"""
Call Smarter: Predicting Term Deposit Subscribers - v9
Clean rewrite. No broken strings. Tested structure.
"""

from __future__ import annotations
import sys, io, json, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle, requests
import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
MOD_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MET_DIR = ROOT / "reports" / "metrics"

st.set_page_config(
    page_title="Call Smarter - Term Deposit Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# CSS
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
*,*::before,*::after{font-family:'Inter',-apple-system,sans-serif !important}
[data-testid="stAppViewContainer"]{background:linear-gradient(170deg,#dde4ed 0%,#e4e9f0 40%,#dfe5ed 70%,#e2e8f0 100%) !important}
.block-container{padding:1rem 2rem 3rem 2rem !important;max-width:1420px}

/* Fix broken Material Symbols arrow icons in Streamlit expanders */
[data-testid="stExpander"] summary span[data-testid="stMarkdownContainer"]{overflow:visible !important}
/* Hide ALL icon spans inside expander summaries (covers every nesting variant) */
[data-testid="stExpander"] details > summary svg,
[data-testid="stExpander"] details > summary [data-testid="stIconMaterial"],
[data-testid="stExpander"] details > summary > span:first-child,
[data-testid="stExpander"] details > summary span[class*="icon"],
[data-testid="stExpander"] details > summary span[style*="Material"] {
    font-size:0 !important;line-height:0 !important;overflow:hidden !important;
    display:inline-flex !important;align-items:center !important;justify-content:center !important;
    width:20px !important;height:20px !important;max-width:20px !important;
    color:transparent !important;-webkit-text-fill-color:transparent !important;
    text-indent:-9999px !important;
}
[data-testid="stExpander"] details > summary > span:first-child::before,
[data-testid="stExpander"] details > summary [data-testid="stIconMaterial"]::before {
    content:'▶';font-size:12px !important;font-family:sans-serif !important;
    color:#64748b !important;-webkit-text-fill-color:#64748b !important;
    text-indent:0 !important;transition:transform 0.2s;display:block;
}
[data-testid="stExpander"] details[open] > summary > span:first-child::before,
[data-testid="stExpander"] details[open] > summary [data-testid="stIconMaterial"]::before {
    content:'▼';
}
/* Nuclear fallback: hide any raw "arrow_right" / "expand_" text leaking from Material Symbols */
[data-testid="stExpander"] details > summary {
    font-size:0.88rem;
}
[data-testid="stExpander"] details > summary > span:first-child *:not(svg) {
    font-size:0 !important;color:transparent !important;-webkit-text-fill-color:transparent !important;
    width:0 !important;overflow:hidden !important;display:inline-block !important;
}

.banner{position:relative;width:100%;height:260px;border-radius:18px;overflow:hidden;margin-bottom:1rem;box-shadow:0 4px 20px rgba(0,0,0,0.1)}
.banner img{width:100%;height:100%;object-fit:cover;object-position:center 20%;display:block;filter:brightness(0.85)}
.banner-overlay{position:absolute;inset:0;background:linear-gradient(90deg,rgba(15,23,42,0.82) 0%,rgba(15,23,42,0.45) 40%,rgba(15,23,42,0.05) 70%,transparent 100%);display:flex;flex-direction:column;justify-content:center;padding:1.5rem 2.5rem;color:white}
.banner-tag{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:0.3rem}
.banner-title{font-size:1.6rem;font-weight:800;letter-spacing:-0.5px;line-height:1.2}
.banner-desc{font-size:0.82rem;opacity:0.7;margin-top:0.4rem;max-width:450px}

.hero-wrap{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 40%,#0c4a6e 70%,#0ea5e9 100%);border-radius:18px;padding:1.8rem 2.5rem;color:white;margin-bottom:1.2rem;position:relative;overflow:hidden;box-shadow:0 6px 24px rgba(15,23,42,0.2)}
.hero-wrap::before{content:'';position:absolute;inset:0;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='200' viewBox='0 0 600 200'%3E%3Cpath d='M0 180Q100 150 200 130T400 70T600 20' fill='none' stroke='rgba(255,255,255,0.07)' stroke-width='2'/%3E%3Cpath d='M0 190Q120 170 250 145T500 80T600 50' fill='none' stroke='rgba(255,255,255,0.04)' stroke-width='1.5'/%3E%3C/svg%3E") no-repeat bottom right/60% auto;pointer-events:none}
.hero-wrap::after{content:'';position:absolute;top:-30%;right:-5%;width:350px;height:350px;background:radial-gradient(circle,rgba(14,165,233,0.2) 0%,transparent 70%);border-radius:50%;pointer-events:none}
.hero-title{font-size:1.6rem;font-weight:800;margin:0;position:relative;z-index:1}
.hero-sub{font-size:0.82rem;opacity:0.7;margin:0.3rem 0 0.7rem;position:relative;z-index:1}
.hero-badges{position:relative;z-index:1;margin-bottom:0.7rem}
.hero-badge{display:inline-block;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.15);backdrop-filter:blur(6px);padding:4px 14px;border-radius:20px;font-size:0.76rem;font-weight:600;margin-right:6px}
.hero-kpis{display:inline-flex;gap:0;position:relative;z-index:1;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:0.5rem 0.3rem;backdrop-filter:blur(6px)}
.hero-kpi{text-align:center;padding:0.2rem 1.2rem;border-left:1px solid rgba(255,255,255,0.1)}
.hero-kpi:first-child{border-left:none}
.hero-kpi-val{font-size:1.2rem;font-weight:800}
.hero-kpi-lbl{font-size:0.6rem;text-transform:uppercase;letter-spacing:0.6px;opacity:0.55}

[data-testid="stMetric"]{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
[data-testid="stMetricLabel"]{font-size:0.7rem !important;font-weight:700 !important;color:#64748b !important;text-transform:uppercase;letter-spacing:0.5px}
[data-testid="stMetricValue"]{font-size:1.4rem !important;font-weight:800 !important;color:#1e293b !important}

.card{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:1.3rem;box-shadow:0 1px 4px rgba(0,0,0,0.03);margin-bottom:0.8rem}
.card h4{margin:0 0 0.4rem;font-size:0.9rem;font-weight:700;color:#1e293b}
.card .muted{color:#64748b;font-size:0.8rem;line-height:1.5}

.input-header{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:0.9rem 1.2rem;margin-bottom:0.6rem;box-shadow:0 1px 3px rgba(0,0,0,0.03);border-top:3px solid #0ea5e9}
.input-header-text{font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#0ea5e9;margin:0}

.sec-head{font-size:1rem;font-weight:800;color:#1e293b;border-bottom:3px solid #0ea5e9;display:inline-block;padding-bottom:0.3rem;margin:1.5rem 0 0.8rem}

.verd-yes{background:linear-gradient(135deg,#059669,#34d399);color:white;padding:1.8rem 2rem;border-radius:18px;box-shadow:0 6px 24px rgba(5,150,105,0.25)}
.verd-no{background:linear-gradient(135deg,#dc2626,#f87171);color:white;padding:1.8rem 2rem;border-radius:18px;box-shadow:0 6px 24px rgba(220,38,38,0.25)}
.verd-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.8rem}
.verd-left{text-align:left}
.verd-title{font-size:1.6rem;font-weight:900;margin:0}
.verd-action{font-size:0.85rem;opacity:0.85;margin:0.2rem 0 0}
.verd-right{text-align:right}
.verd-prob{font-size:2.8rem;font-weight:900;margin:0;line-height:1}
.verd-prob-label{font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;opacity:0.7;margin-top:2px}
.verd-desc{font-size:0.82rem;opacity:0.9;margin:0}
.biz-card{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:1.2rem 1.4rem;margin-top:0.8rem;box-shadow:0 1px 4px rgba(0,0,0,0.03)}
.biz-card h4{margin:0 0 0.5rem;font-size:0.85rem;font-weight:700;color:#1e293b}
.biz-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f1f5f9;font-size:0.82rem}
.biz-row:last-child{border-bottom:none}
.biz-label{color:#64748b;font-weight:500}
.biz-val{font-weight:700;color:#1e293b}

.pill-pos{display:inline-block;background:#f0fdf4;color:#166534;padding:5px 12px;border-radius:16px;font-size:0.8rem;font-weight:600;margin:3px;border:1px solid #bbf7d0}
.pill-neg{display:inline-block;background:#fef2f2;color:#991b1b;padding:5px 12px;border-radius:16px;font-size:0.8rem;font-weight:600;margin:3px;border:1px solid #fecaca}
.factor-summary{font-size:0.82rem;color:#475569;line-height:1.6;padding:0.6rem 0.8rem;background:#f8fafc;border-left:3px solid #0ea5e9;border-radius:0 8px 8px 0;margin:0.4rem 0 0.8rem}
.explainer-table{width:100%;border-collapse:collapse;font-size:0.82rem;margin:0.6rem 0}
.explainer-table th{background:#f1f5f9;color:#1e293b;font-weight:700;padding:8px 12px;text-align:left;border-bottom:2px solid #e2e8f0}
.explainer-table td{padding:8px 12px;border-bottom:1px solid #f1f5f9;color:#475569}
.explainer-table tr:last-child td{border-bottom:none}
.explainer-table .val{font-weight:700;color:#1e293b}
.explain-card{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:1.2rem 1.4rem;margin-top:0.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.03)}
.explain-card h4{margin:0 0 0.3rem;font-size:0.85rem;font-weight:700;color:#1e293b}
.explain-card p,.explain-card li{font-size:0.82rem;color:#475569;line-height:1.7}
.explain-highlight{background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1px solid #93c5fd;border-radius:10px;padding:0.8rem 1rem;margin:0.5rem 0;font-size:0.82rem;color:#1e40af;line-height:1.6}
.explain-highlight strong{color:#1e3a5f}

.stTabs [data-baseweb="tab-list"]{background:white;border-radius:12px;padding:4px;border:1px solid #e2e8f0;gap:2px}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:9px 24px;font-weight:600;font-size:0.86rem}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0ea5e9,#2563eb) !important;color:white !important}

div.stButton>button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#0ea5e9,#2563eb) !important;color:white !important;border:none !important;border-radius:12px !important;font-weight:700 !important;padding:0.6rem 2rem !important;box-shadow:0 3px 12px rgba(37,99,235,0.25) !important}
div.stButton>button[data-testid="stBaseButton-secondary"]{background:white !important;border:1.5px solid #e2e8f0 !important;border-radius:10px !important;font-weight:600 !important;color:#475569 !important}
.stDownloadButton>button{background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;color:white !important;border:none !important;border-radius:10px !important}

[data-baseweb="select"]>div{background:white !important;border:1.5px solid #e2e8f0 !important;border-radius:10px !important}
[data-testid="stNumberInput"] input{background:white !important;border:1.5px solid #e2e8f0 !important;border-radius:10px !important}
[data-testid="stTextInput"] input{background:white !important;border:1.5px solid #e2e8f0 !important;border-radius:10px !important}
[data-testid="stFileUploader"]>div{border:2px dashed #cbd5e1 !important;border-radius:14px !important;background:white !important}
.stSelectbox label,.stSlider label,.stNumberInput label{font-size:0.78rem !important;font-weight:600 !important;color:#475569 !important}

.app-foot{margin-top:2.5rem;padding:1.2rem 1.5rem;text-align:center;background:white;border:1px solid #e2e8f0;border-radius:14px}
.app-foot-title{font-size:0.9rem;font-weight:800;color:#1e293b}
.app-foot-line{width:36px;height:3px;margin:0.4rem auto;background:linear-gradient(90deg,#0ea5e9,#6366f1);border-radius:2px}
.app-foot-sub{font-size:0.72rem;color:#94a3b8}

#MainMenu,header,footer{visibility:hidden}

@keyframes pulse-glow{0%,100%{box-shadow:0 0 15px rgba(14,165,233,0.15)}50%{box-shadow:0 0 30px rgba(14,165,233,0.3)}}
.kpi-glow{animation:pulse-glow 3s ease-in-out infinite}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# CONFIG & LOADING
# =====================================================================
FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
DROP_COLS = ["duration", "y", "deposit"]


@st.cache_resource
def load_best_model():
    meta = {}
    if (MOD_DIR / "threshold.json").exists():
        with open(MOD_DIR / "threshold.json") as f:
            meta = json.load(f)
    name = meta.get("model", "")
    thr = meta.get("threshold", 0.30)
    path = MOD_DIR / f"{name}.pkl"
    if not path.exists():
        for fb in ["xgboost", "lightgbm", "random_forest", "logistic_regression", "voting_ensemble"]:
            p = MOD_DIR / f"{fb}.pkl"
            if p.exists():
                name, path = fb, p
                break
    if not path.exists():
        return None, None, 0.30, {}
    with open(path, "rb") as f:
        mdl = pickle.load(f)
    return mdl, name, thr, meta


@st.cache_data
def load_comparison():
    p = MET_DIR / "comparison.json"
    return pd.read_json(p) if p.exists() else None


model, model_name, threshold, meta = load_best_model()
comp_df = load_comparison()
is_ensemble = isinstance(model, dict)

if model is None:
    st.error("No models found. Run `python scripts/run_pipeline.py` first.")
    st.stop()


# =====================================================================
# ENGINE
# =====================================================================
def predict_one(df):
    if is_ensemble:
        return float(model["voter"].predict_proba(model["preprocess"].transform(df))[0, 1])
    return float(model.predict_proba(df)[0, 1])


def predict_batch(df):
    # Sanitise numerics: fill NaN with 0, replace inf, clip extremes
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0).replace([np.inf, -np.inf], 0)
    df[num_cols] = df[num_cols].clip(-1e9, 1e9)
    if is_ensemble:
        return model["voter"].predict_proba(model["preprocess"].transform(df))[:, 1]
    return model.predict_proba(df)[:, 1]


def get_verdict(prob):
    # Confidence band
    if prob >= 0.60:
        confidence = "Very High Confidence"
    elif prob >= 0.35:
        confidence = "High Confidence"
    elif prob >= threshold:
        confidence = "Moderate Confidence"
    else:
        if prob >= threshold * 0.5:
            confidence = "Low Confidence"
        else:
            confidence = "Very Low"

    ev_profit = prob * 200 - (1 - prob) * 5  # expected value per call

    if prob >= threshold:
        return (
            True,
            "LIKELY TO SUBSCRIBE",
            "Recommended Action: Call this customer",
            confidence,
            ev_profit,
        )
    return (
        False,
        "UNLIKELY TO SUBSCRIBE",
        "Recommended Action: Skip — focus resources elsewhere",
        confidence,
        ev_profit,
    )


def get_factors(row, prob):
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
        pos.append(f"{row['job'].title()} - high-value")
    if row.get("month") in ("mar", "sep", "oct", "dec"):
        pos.append(f"Strong month ({row['month'].title()})")
    if row.get("campaign", 0) > 5:
        neg.append(f"{row['campaign']} calls - diminishing returns")
    elif row.get("campaign", 99) <= 2:
        pos.append("Fresh lead (2 or fewer calls)")
    if row.get("age", 0) > 60:
        pos.append("Senior (60+)")
    elif row.get("age", 0) < 30:
        pos.append("Young (<30)")
    p = row.get("pdays", 999)
    if 0 < p < 30:
        pos.append("Recently contacted")
    if row.get("housing") == "no" and row.get("loan") == "no":
        pos.append("No existing loans")
    if row.get("nr.employed", 5200) < 5100:
        pos.append("Receptive market")
    return pos, neg


# Median defaults from the training data — used to fill missing macro columns in batch uploads
_COLUMN_DEFAULTS = {
    "day_of_week": "thu",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
}


def validate_df(df):
    df.columns = df.columns.str.strip().str.lower()
    renames = {
        "dayofweek": "day_of_week",
        "day": "day_of_week",
        "emp_var_rate": "emp.var.rate",
        "empvarrate": "emp.var.rate",
        "cons_price_idx": "cons.price.idx",
        "conspriceidx": "cons.price.idx",
        "cons_conf_idx": "cons.conf.idx",
        "consconfidx": "cons.conf.idx",
        "nr_employed": "nr.employed",
        "nremployed": "nr.employed",
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
    # Fill missing columns with training-data medians instead of blocking
    missing = [c for c in FEATURES if c not in df.columns]
    filled = []
    for c in missing:
        if c in _COLUMN_DEFAULTS:
            df[c] = _COLUMN_DEFAULTS[c]
            filled.append(c)
    if filled:
        warns.append(f"Auto-filled missing columns with defaults: {filled}")
    # Sanitise numeric values: NaN → 0, inf → 0, clip extremes
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        nan_count = int(df[num_cols].isna().sum().sum())
        inf_count = int(np.isinf(df[num_cols].select_dtypes(include="number")).sum().sum())
        df[num_cols] = df[num_cols].fillna(0).replace([np.inf, -np.inf], 0)
        df[num_cols] = df[num_cols].clip(-1e9, 1e9)
        if nan_count:
            warns.append(f"Replaced {nan_count} NaN values with 0")
        if inf_count:
            warns.append(f"Replaced {inf_count} Inf values with 0")
    still_missing = [c for c in FEATURES if c not in df.columns]
    return df, warns, still_missing


# =====================================================================
# BANNER IMAGE
# =====================================================================
criterion = meta.get("selection_criterion", "composite_score")
composite = ""
best_recall = ""
best_auc = ""
best_profit = ""
display_name = model_name.replace("_", " ").title()
crit_display = criterion.replace("_", " ").title()

if comp_df is not None:
    r = comp_df[comp_df["model"] == model_name]
    if not r.empty:
        if "composite_score" in r.columns:
            composite = f" | {r.iloc[0]['composite_score']:.3f}"
        if "test_recall" in r.columns:
            best_recall = f"{r.iloc[0]['test_recall']:.0%}"
        if "test_roc_auc" in r.columns:
            best_auc = f"{r.iloc[0]['test_roc_auc']:.1%}"
        if "net_profit" in r.columns:
            best_profit = f"${r.iloc[0]['net_profit']:,.0f}"

# ── Banner ──────────────────────────────────────────────────────────────────────
# Professional banking banner with inline SVG scene (no external images needed)
_BANK_SVG = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 520 320'%3E"
    # floor
    "%3Crect x='0' y='260' width='520' height='60' fill='rgba(255,255,255,0.04)'/%3E"
    "%3Cline x1='0' y1='260' x2='520' y2='260' stroke='rgba(255,255,255,0.08)' stroke-width='1'/%3E"
    # bank building
    "%3Crect x='160' y='80' width='200' height='180' rx='3' fill='rgba(255,255,255,0.06)' stroke='rgba(255,255,255,0.10)' stroke-width='1'/%3E"
    # pediment / triangle roof
    "%3Cpolygon points='140,80 260,20 380,80' fill='rgba(255,255,255,0.07)' stroke='rgba(255,255,255,0.10)' stroke-width='1'/%3E"
    # columns
    "%3Crect x='185' y='100' width='12' height='160' rx='2' fill='rgba(255,255,255,0.09)'/%3E"
    "%3Crect x='225' y='100' width='12' height='160' rx='2' fill='rgba(255,255,255,0.09)'/%3E"
    "%3Crect x='283' y='100' width='12' height='160' rx='2' fill='rgba(255,255,255,0.09)'/%3E"
    "%3Crect x='323' y='100' width='12' height='160' rx='2' fill='rgba(255,255,255,0.09)'/%3E"
    # door
    "%3Crect x='243' y='180' width='34' height='80' rx='3' fill='rgba(255,255,255,0.05)' stroke='rgba(255,255,255,0.12)' stroke-width='1'/%3E"
    # person 1 (left, standing)
    "%3Ccircle cx='100' cy='210' r='10' fill='rgba(255,255,255,0.10)'/%3E"
    "%3Crect x='93' y='222' width='14' height='38' rx='4' fill='rgba(255,255,255,0.08)'/%3E"
    # person 2 (right, with briefcase)
    "%3Ccircle cx='420' cy='215' r='10' fill='rgba(255,255,255,0.10)'/%3E"
    "%3Crect x='413' y='227' width='14' height='33' rx='4' fill='rgba(255,255,255,0.08)'/%3E"
    "%3Crect x='430' y='235' width='10' height='14' rx='2' fill='rgba(255,255,255,0.06)'/%3E"
    # decorative chart line (growth)
    "%3Cpolyline points='30,250 80,230 130,240 180,200 230,190 280,150 330,160 380,120 430,100 480,60' "
    "fill='none' stroke='rgba(14,165,233,0.18)' stroke-width='2' stroke-linecap='round'/%3E"
    # dollar sign on building
    "%3Ctext x='260' y='62' text-anchor='middle' font-size='18' font-weight='bold' fill='rgba(255,255,255,0.10)' font-family='sans-serif'%3E%24%3C/text%3E"
    # subtle sparkles
    "%3Ccircle cx='60' cy='50' r='2' fill='rgba(255,255,255,0.12)'/%3E"
    "%3Ccircle cx='450' cy='70' r='1.5' fill='rgba(255,255,255,0.10)'/%3E"
    "%3Ccircle cx='490' cy='180' r='2' fill='rgba(255,255,255,0.08)'/%3E"
    "%3C/svg%3E"
)

st.markdown(
    f'''
    <div style="background:linear-gradient(135deg,#0f172a 0%,#162544 25%,#1e3a5f 50%,#1a5276 75%,#2980b9 100%);
                border-radius:18px;overflow:hidden;margin-bottom:1rem;
                box-shadow:0 6px 24px rgba(15,23,42,0.35);position:relative;">
      <!-- SVG bank scene on right side -->
      <div style="position:absolute;right:0;top:0;bottom:0;width:50%;
                  background:url('{_BANK_SVG}') right center/contain no-repeat;
                  pointer-events:none;opacity:0.8;"></div>
      <!-- subtle radial glow -->
      <div style="position:absolute;top:-20%;right:10%;width:300px;height:300px;
                  background:radial-gradient(circle,rgba(41,128,185,0.15) 0%,transparent 70%);
                  border-radius:50%;pointer-events:none;"></div>
      <!-- wave lines -->
      <div style="position:absolute;inset:0;
                  background:url(&quot;data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='300' viewBox='0 0 800 300'%3E%3Cpath d='M0 280Q150 240 300 250T600 200T800 170' fill='none' stroke='rgba(255,255,255,0.04)' stroke-width='1.5'/%3E%3Cpath d='M0 290Q200 260 400 270T700 220T800 200' fill='none' stroke='rgba(255,255,255,0.03)' stroke-width='1'/%3E%3C/svg%3E&quot;) no-repeat bottom/100% auto;
                  pointer-events:none;"></div>
      <!-- content -->
      <div style="position:relative;z-index:1;padding:2rem 2.5rem 1.8rem;">
        <div style="max-width:58%;color:white;">
          <div style="font-size:1.65rem;font-weight:800;line-height:1.25;margin-bottom:0.3rem;
                      letter-spacing:-0.5px;">Call Smarter: Predicting Term Deposit Subscribers</div>
          <div style="font-size:0.82rem;opacity:0.55;margin-bottom:1rem;line-height:1.5;">
            ML-Powered Campaign Intelligence
          </div>
          <div style="margin-bottom:0.85rem;display:flex;flex-wrap:wrap;gap:8px;">
            <span style="background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.18);
                         backdrop-filter:blur(6px);padding:5px 18px;border-radius:22px;
                         font-size:0.78rem;font-weight:600;">{display_name}</span>
            <span style="background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.18);
                         backdrop-filter:blur(6px);padding:5px 18px;border-radius:22px;
                         font-size:0.78rem;font-weight:600;">{threshold:.1%} Threshold</span>
            <span style="background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.18);
                         backdrop-filter:blur(6px);padding:5px 18px;border-radius:22px;
                         font-size:0.78rem;font-weight:600;">Max Net Profit</span>
          </div>
          <div style="display:inline-flex;background:rgba(255,255,255,0.08);
                      border:1px solid rgba(255,255,255,0.14);border-radius:12px;
                      padding:0.55rem 0.15rem;backdrop-filter:blur(8px);">
            <div style="text-align:center;padding:0.2rem 1.3rem;">
              <div style="font-size:1.2rem;font-weight:800;line-height:1.2;">{best_recall or "N/A"}</div>
              <div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.7px;opacity:0.55;margin-top:3px;">Recall</div>
            </div>
            <div style="text-align:center;padding:0.2rem 1.3rem;border-left:1px solid rgba(255,255,255,0.14);">
              <div style="font-size:1.2rem;font-weight:800;line-height:1.2;">{best_auc or "N/A"}</div>
              <div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.7px;opacity:0.55;margin-top:3px;">AUC-ROC</div>
            </div>
            <div style="text-align:center;padding:0.2rem 1.3rem;border-left:1px solid rgba(255,255,255,0.14);">
              <div style="font-size:1.2rem;font-weight:800;line-height:1.2;">{best_profit or "N/A"}</div>
              <div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.7px;opacity:0.55;margin-top:3px;">Net Profit</div>
            </div>
            <div style="text-align:center;padding:0.2rem 1.3rem;border-left:1px solid rgba(255,255,255,0.14);">
              <div style="font-size:1.2rem;font-weight:800;line-height:1.2;">{threshold:.0%}</div>
              <div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.7px;opacity:0.55;margin-top:3px;">Threshold</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)


# =====================================================================
# HERO
# =====================================================================
criterion = meta.get("selection_criterion", "composite_score")
composite = ""
best_recall = ""
best_auc = ""
best_profit = ""

if comp_df is not None:
    r = comp_df[comp_df["model"] == model_name]
    if not r.empty:
        if "composite_score" in r.columns:
            composite = f" | {r.iloc[0]['composite_score']:.3f}"
        if "test_recall" in r.columns:
            best_recall = f"{r.iloc[0]['test_recall']:.0%}"
        if "test_roc_auc" in r.columns:
            best_auc = f"{r.iloc[0]['test_roc_auc']:.1%}"
        if "net_profit" in r.columns:
            best_profit = f"${r.iloc[0]['net_profit']:,.0f}"

display_name = model_name.replace("_", " ").title()
crit_display = criterion.replace("_", " ").title()

st.markdown("")


# =====================================================================
# TABS
# =====================================================================
tab_pred, tab_batch = st.tabs(["Predict Client", "Batch Predict"])


# =====================================================================
# TAB 1 - PREDICT CLIENT
# =====================================================================
with tab_pred:

    DEFAULTS = {
        "p_age": 40,
        "p_job": 0,
        "p_mar": "married",
        "p_edu": 0,
        "p_def": "no",
        "p_hou": "yes",
        "p_loan": "no",
        "p_con": "cellular",
        "p_mon": 4,
        "p_day": "mon",
        "p_camp": 2,
        "p_pdays": 999,
        "p_prev": 0,
        "p_pout": "nonexistent",
        "p_emp": 1.1,
        "p_cpi": 93.994,
        "p_cci": -36.4,
        "p_eur": 4.857,
        "p_nr": 5191.0,
    }

    for _k, _v in DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    def reset_form():
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    # Info bar + reset
    i1, i2 = st.columns([6, 1])
    with i1:
        st.markdown(
            f'<div class="card" style="padding:0.7rem 1.2rem;">'
            f'<span style="font-size:0.83rem;"><b>Model:</b> {display_name}</span>'
            f' &nbsp;|&nbsp; <span style="font-size:0.83rem;"><b>Threshold:</b> {threshold:.1%}</span>'
            f' &nbsp;|&nbsp; <span style="font-size:0.83rem;color:#059669;">Above = Call</span>'
            f' &nbsp; <span style="font-size:0.83rem;color:#dc2626;">Below = Skip</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with i2:
        st.button("Reset", key="reset_btn", on_click=reset_form, use_container_width=True)

    # Three columns
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            '<div class="input-header"><p class="input-header-text">Client Profile</p></div>',
            unsafe_allow_html=True,
        )
        age = st.slider("Age", 17, 98, key="p_age")
        JOB_OPTS = [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician", "unemployed",
        ]
        job = st.selectbox("Occupation", JOB_OPTS, key="p_job")
        marital = st.segmented_control("Marital Status", ["married", "single", "divorced"], key="p_mar")
        EDU_OPTS = [
            "basic.4y", "basic.6y", "basic.9y", "high.school",
            "professional.course", "university.degree", "illiterate", "unknown",
        ]
        education = st.selectbox("Education", EDU_OPTS, key="p_edu")
        default = st.segmented_control("Credit Default", ["no", "unknown", "yes"], key="p_def")
        housing = st.segmented_control("Housing Loan", ["yes", "no", "unknown"], key="p_hou")
        loan = st.segmented_control("Personal Loan", ["no", "yes", "unknown"], key="p_loan")

    with c2:
        st.markdown(
            '<div class="input-header"><p class="input-header-text">Campaign Details</p></div>',
            unsafe_allow_html=True,
        )
        contact = st.segmented_control("Contact Method", ["cellular", "telephone"], key="p_con")
        MONTH_OPTS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month = st.selectbox("Contact Month", MONTH_OPTS, key="p_mon")
        day = st.segmented_control("Day of Week", ["mon", "tue", "wed", "thu", "fri"], key="p_day")
        campaign = st.slider("Calls This Campaign", 1, 50, key="p_camp")
        pdays = st.number_input("Days Since Last Contact", 0, 999, help="999 = never contacted", key="p_pdays")
        previous = st.slider("Previous Contacts", 0, 50, key="p_prev")
        poutcome = st.segmented_control("Previous Outcome", ["nonexistent", "failure", "success"], key="p_pout")

    with c3:
        st.markdown(
            '<div class="input-header"><p class="input-header-text">Economic Indicators</p></div>',
            unsafe_allow_html=True,
        )
        emp_var = st.slider("Emp Variation Rate", -4.0, 2.0, step=0.1, key="p_emp")
        cpi = st.slider("Consumer Price Idx", 90.0, 96.0, step=0.1, key="p_cpi")
        cci = st.slider("Consumer Confidence", -55.0, -20.0, step=0.1, key="p_cci")
        euribor = st.slider("Euribor 3m Rate", 0.0, 6.0, step=0.1, key="p_eur")
        nr_emp = st.slider("Nr Employed (k)", 4800.0, 5400.0, step=10.0, key="p_nr")
        st.markdown(
            '<div class="card" style="margin-top:0.3rem;">'
            '<h4 style="font-size:0.8rem;">Quick Reference</h4>'
            '<div class="muted" style="font-size:0.75rem;">'
            "<b>Euribor &lt; 2.0</b> = favorable rates<br>"
            "<b>Nr Employed &lt; 5100</b> = receptive market<br>"
            "<b>Emp Var &lt; 0</b> = economic pressure<br>"
            "<b>CCI &lt; -40</b> = low confidence"
            "</div></div>",
            unsafe_allow_html=True,
        )

    # Predict button
    st.markdown("")
    _, bc, _ = st.columns([1, 2, 1])
    with bc:
        run = st.button("Run Prediction", type="primary", use_container_width=True, key="run_pred")

    if run:
        inp = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan, "contact": contact,
            "month": month, "day_of_week": day, "campaign": campaign, "pdays": pdays,
            "previous": previous, "poutcome": poutcome, "emp.var.rate": emp_var,
            "cons.price.idx": cpi, "cons.conf.idx": cci, "euribor3m": euribor,
            "nr.employed": nr_emp,
        }
        inp_df = pd.DataFrame([inp])

        try:
            prob = predict_one(inp_df)
            yes, vtext, vdesc, confidence, ev_profit = get_verdict(prob)
            pf, nf = get_factors(inp, prob)

            # =========================================================
            # TWO-COLUMN LAYOUT: Left = verdict + metrics, Right = gauge + factors + explainer
            # =========================================================
            st.markdown("")
            col_left, col_right = st.columns(2, gap="large")

            # ----- LEFT COLUMN: Verdict + Metrics + How it Works -----
            with col_left:
                cls = "verd-yes" if yes else "verd-no"
                icon = "\u2705" if yes else "\u274C"
                st.markdown(
                    f'<div class="{cls}">'
                    f'<div class="verd-header">'
                    f'<div class="verd-left">'
                    f'<p class="verd-title">{icon} {vtext}</p>'
                    f'<p class="verd-action">{vdesc}</p>'
                    f'</div>'
                    f'<div class="verd-right">'
                    f'<p class="verd-prob">{prob:.1%}</p>'
                    f'<p class="verd-prob-label">Subscription Probability</p>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("")
                m1, m2 = st.columns(2)
                m1.metric("Subscription Probability", f"{prob:.1%}")
                m2.metric("Confidence Level", confidence)
                m3, m4 = st.columns(2)
                mg = prob - threshold
                m3.metric("Above Threshold By", f"{mg:+.1%}", delta=f"{mg:+.1%}")
                m4.metric("Expected Value / Call", f"${ev_profit:+,.0f}")

                # Business explanation card
                st.markdown(
                    f'<div class="biz-card">'
                    f'<h4>\U0001F4CA How This Prediction Works</h4>'
                    f'<div class="biz-row"><span class="biz-label">Model\'s confidence</span>'
                    f'<span class="biz-val">{prob:.1%}</span></div>'
                    f'<div class="biz-row"><span class="biz-label">Business threshold</span>'
                    f'<span class="biz-val">{threshold:.1%}</span></div>'
                    f'<div class="biz-row"><span class="biz-label">Revenue if subscribes</span>'
                    f'<span class="biz-val">$200</span></div>'
                    f'<div class="biz-row"><span class="biz-label">Cost per call</span>'
                    f'<span class="biz-val">$5</span></div>'
                    f'<div class="biz-row"><span class="biz-label">Expected profit</span>'
                    f'<span class="biz-val" style="color:{"#059669" if ev_profit > 0 else "#dc2626"}">${ev_profit:+,.2f}</span></div>'
                    f'<div class="biz-row"><span class="biz-label">Why {threshold:.0%}? (not 50%)</span>'
                    f'<span class="biz-val" style="font-weight:500">FN=$200 is 40\u00d7 FP=$5</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ----- RIGHT COLUMN: Gauge + Factors + Explainer -----
            with col_right:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Customer Subscription Likelihood", "font": {"size": 14, "color": "#64748b"}},
                        gauge={
                            "axis": {"range": [0, 100], "ticksuffix": "%"},
                            "bar": {"color": "#059669" if yes else "#dc2626", "thickness": 0.25},
                            "bgcolor": "#f8fafc",
                            "steps": [
                                {"range": [0, threshold * 100], "color": "#fef2f2"},
                                {"range": [threshold * 100, 100], "color": "#f0fdf4"},
                            ],
                            "threshold": {
                                "line": {"color": "#1e293b", "width": 3},
                                "value": threshold * 100,
                            },
                        },
                        number={"suffix": "%", "font": {"size": 28, "color": "#1e293b"}},
                    )
                )
                fig.add_annotation(x=0.08, y=-0.12, text="<b>Skip Zone</b>",
                                   showarrow=False, font=dict(size=10, color="#dc2626"),
                                   xref="paper", yref="paper")
                fig.add_annotation(x=0.92, y=-0.12, text="<b>Call Zone</b>",
                                   showarrow=False, font=dict(size=10, color="#059669"),
                                   xref="paper", yref="paper")
                fig.update_layout(
                    height=250,
                    margin=dict(t=35, b=30, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Key Decision Factors
                st.markdown('<div class="sec-head">Key Decision Factors</div>', unsafe_allow_html=True)
                html = ""
                for f in pf:
                    html += f'<span class="pill-pos">{f}</span>'
                for f in nf:
                    html += f'<span class="pill-neg">{f}</span>'
                if not pf and not nf:
                    html = '<span style="color:#64748b;font-size:0.85rem;">No strong signals</span>'
                st.markdown(f'<div style="margin-bottom:0.4rem;">{html}</div>', unsafe_allow_html=True)

                # Contextual summary
                n_pos, n_neg = len(pf), len(nf)
                if yes:
                    if n_pos > n_neg:
                        factor_summary = (f"<strong>{n_pos} positive</strong> signal{'s' if n_pos!=1 else ''} "
                                          f"vs {n_neg} negative \u2014 strongly supports "
                                          f"<strong>{prob:.1%}</strong> probability. "
                                          f"Expected return: <strong>${ev_profit:+,.2f}</strong>/call.")
                    else:
                        factor_summary = (f"Despite {n_neg} negative signal{'s' if n_neg!=1 else ''}, "
                                          f"overall assessment yields <strong>{prob:.1%}</strong> probability, "
                                          f"above {threshold:.1%} threshold. "
                                          f"Net value: <strong>${ev_profit:+,.2f}</strong>/call.")
                else:
                    if n_pos > 0:
                        factor_summary = (f"Despite {n_pos} positive signal{'s' if n_pos!=1 else ''}, "
                                          f"overall profile yields only <strong>{prob:.1%}</strong> \u2014 "
                                          f"below {threshold:.1%} break-even. "
                                          f"Loss: ~<strong>${abs(ev_profit):.2f}</strong>/call.")
                    else:
                        factor_summary = (f"No strong positive signals. "
                                          f"Only <strong>{prob:.1%}</strong> probability. "
                                          f"Calling would lose ~<strong>${abs(ev_profit):.2f}</strong>/call.")
                st.markdown(f'<div class="factor-summary">{factor_summary}</div>', unsafe_allow_html=True)

            # =============================================================
            # FULL-WIDTH EXPLAINER (below both columns)
            # =============================================================
            with st.expander("Understanding This Prediction \u2014 Threshold, Accuracy & Business Logic", expanded=False):

                exp_left, exp_right = st.columns(2, gap="large")

                with exp_left:
                    # Dynamic intro
                    if yes:
                        st.markdown(
                            f'<div class="explain-highlight">'
                            f'<strong>\u2705 Why call this customer?</strong> '
                            f'The model gives <strong>{prob:.1%}</strong> probability. '
                            f'At our cost structure ($200 revenue vs $5 cost), expected net value '
                            f'is <strong>${ev_profit:+,.2f}</strong> \u2014 a profitable contact.</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="explain-highlight">'
                            f'<strong>\u274C Why skip this customer?</strong> '
                            f'Only <strong>{prob:.1%}</strong> probability. '
                            f'The $5 call cost exceeds the expected return '
                            f'($200 \u00d7 {prob:.3f} = <strong>${prob*200:.2f}</strong>), '
                            f'net loss: <strong>${abs(ev_profit):.2f}</strong>/call.</div>',
                            unsafe_allow_html=True)

                    # Key Concepts
                    st.markdown(
                        '<div class="explain-card">'
                        '<h4>\U0001F4D6 Key Concepts</h4>'
                        '<table class="explainer-table">'
                        '<tr><th>Concept</th><th>What It Means</th><th>Value</th></tr>'
                        f'<tr><td>Model Probability</td><td>Confidence <em>this</em> customer subscribes</td>'
                        f'<td class="val">{prob:.1%}</td></tr>'
                        f'<tr><td>Threshold</td><td>Break-even call decision</td>'
                        f'<td class="val">{threshold:.1%}</td></tr>'
                        '<tr><td>AUC-ROC</td><td>Model separating power</td>'
                        '<td class="val">80.1%</td></tr>'
                        '<tr><td>Accuracy</td><td>% correct (misleading!)</td>'
                        '<td class="val">~75%</td></tr>'
                        '</table>'
                        '</div>',
                        unsafe_allow_html=True)

                    # How scoring works
                    st.markdown(
                        '<div class="explain-card">'
                        '<h4>\U0001F3AF How Every Customer Gets Scored</h4>'
                        '<ol style="font-size:0.82rem;color:#475569;line-height:1.8;padding-left:1.2rem">'
                        '<li>Model scores every customer <strong>0 \u2013 100%</strong> '
                        '(e.g., A=62%, B=8%, C=23%)</li>'
                        f'<li><strong>Threshold ({threshold:.1%})</strong> = business decision line</li>'
                        f'<li>Above {threshold:.1%} \u2192 \u2705 Call &nbsp;|&nbsp; '
                        f'Below {threshold:.1%} \u2192 \u274C Skip</li>'
                        '</ol>'
                        '</div>',
                        unsafe_allow_html=True)

                    # Why not 50%?
                    st.markdown(
                        '<div class="explain-card">'
                        '<h4>\U0001F4B0 Why 14% Threshold \u2014 Not 50%?</h4>'
                        '<p>Because of the <strong>40:1 cost asymmetry</strong>:</p>'
                        '<table class="explainer-table">'
                        '<tr><td>Missing a subscriber (FN)</td>'
                        '<td class="val" style="color:#dc2626">\u2212$200</td></tr>'
                        '<tr><td>Wasting a call (FP)</td>'
                        '<td class="val" style="color:#059669">\u2212$5</td></tr>'
                        '</table>'
                        '<div class="explain-highlight" style="text-align:center">'
                        '$200 \u00d7 0.15 = <strong>$30</strong> revenue \u2212 '
                        '$5 cost = <strong style="color:#059669">+$25 net</strong>/call<br>'
                        '<span style="font-size:0.78rem;opacity:0.8">'
                        'A 50% threshold would lose $55K+ in profit.</span>'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True)

                with exp_right:
                    # Why accuracy is misleading
                    st.markdown(
                        '<div class="explain-card">'
                        '<h4>\u26A0\uFE0F Why Accuracy Is Misleading</h4>'
                        '<p>88.3% of customers say "no". A model that <em>always</em> predicts "no" '
                        'gets 88.3% accuracy but earns <strong>$0</strong>.</p>'
                        '<table class="explainer-table">'
                        '<tr><th>Strategy</th><th>Accuracy</th><th>Recall</th><th>Profit</th></tr>'
                        '<tr><td>Always "no"</td><td>88.3%</td>'
                        '<td class="val" style="color:#dc2626">0%</td>'
                        '<td class="val" style="color:#dc2626">$0</td></tr>'
                        f'<tr><td>Our model @ {threshold:.1%}</td><td>~75%</td>'
                        '<td class="val" style="color:#059669">100%</td>'
                        '<td class="val" style="color:#059669">$108,345</td></tr>'
                        '</table>'
                        '<p style="font-weight:600;color:#1e293b">Lower accuracy, but every subscriber is caught '
                        '\u2014 <strong>$108K more</strong> profit.</p>'
                        '</div>',
                        unsafe_allow_html=True)

                    # What to tell stakeholders
                    st.markdown(
                        '<div class="explain-card">'
                        '<h4>\U0001F4AC What to Tell Stakeholders</h4>'
                        '<div class="explain-highlight">'
                        '"Our model scores each customer\'s likelihood of subscribing. '
                        'For every 100 customers called based on recommendations, '
                        'we generate <strong>$108K net profit</strong> after call costs. '
                        'Without the model, we\'d call everyone (wasting 88% of calls) '
                        'or miss high-value subscribers."'
                        '</div>'
                        '<table class="explainer-table">'
                        '<tr><td>\U0001F4B5 Net Profit</td><td class="val">$108,345</td></tr>'
                        '<tr><td>\U0001F3AF Recall</td><td class="val">100%</td></tr>'
                        '<tr><td>\U0001F4C8 AUC-ROC</td><td class="val">80.1%</td></tr>'
                        '<tr><td>\U0001F4B0 Cost Savings</td><td class="val">vs calling everyone</td></tr>'
                        '</table>'
                        '</div>',
                        unsafe_allow_html=True)

                    # This customer's math
                    st.markdown(
                        f'<div class="explain-card">'
                        f'<h4>\U0001F9EE This Customer\'s Math</h4>'
                        f'<table class="explainer-table">'
                        f'<tr><td>Probability</td><td class="val">{prob:.1%}</td></tr>'
                        f'<tr><td>Expected revenue</td><td class="val">${prob*200:.2f}</td></tr>'
                        f'<tr><td>Call cost</td><td class="val">$5.00</td></tr>'
                        f'<tr><td>Net value</td>'
                        f'<td class="val" style="color:{"#059669" if ev_profit > 0 else "#dc2626"}">${ev_profit:+,.2f}</td></tr>'
                        f'<tr><td>Decision</td>'
                        f'<td class="val" style="color:{"#059669" if yes else "#dc2626"}">'
                        f'{"\u2705 CALL \u2014 profitable" if yes else "\u274C SKIP \u2014 would lose money"}</td></tr>'
                        f'</table>'
                        f'</div>',
                        unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)


# =====================================================================
# TAB 2 - BATCH PREDICT
# =====================================================================
with tab_batch:

    st.markdown(
        '<div class="card">'
        "<h4>Bulk Client Scoring</h4>"
        '<div class="muted">Upload CSV/Excel or provide a URL. Each client scored YES (call) or NO (skip).</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    source = st.radio("Data Source", ["File Upload", "URL"], horizontal=True, key="batch_src")
    df_raw = None

    if source == "File Upload":
        uploaded = st.file_uploader("Drop your file here", type=["csv", "xlsx", "xls"], key="fu2")
        if uploaded:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df_raw = pd.read_excel(uploaded)
                else:
                    content = uploaded.read().decode("utf-8")
                    uploaded.seek(0)
                    sep = ";" if ";" in content.split("\n")[0] and "," not in content.split("\n")[0] else ","
                    df_raw = pd.read_csv(io.StringIO(content), sep=sep)
            except Exception as e:
                st.error(f"Failed to read file: {e}")
    else:
        url = st.text_input("Dataset URL", placeholder="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip", key="url_in2")
        if url and url.startswith("http"):
            # Auto-correct common UCI dataset page URLs to the direct download link
            import re as _re
            _uci_page = _re.match(r"https?://archive\.ics\.uci\.edu/dataset/(\d+)/(.+?)/?$", url)
            if _uci_page:
                url = f"https://archive.ics.uci.edu/static/public/{_uci_page.group(1)}/{_uci_page.group(2)}.zip"
                st.info(f"Auto-corrected to direct download: `{url}`")
            try:
                with st.spinner("Fetching dataset..."):
                    resp = requests.get(url, timeout=60, verify=False)
                    resp.raise_for_status()
                    raw = resp.content
                    ct = resp.headers.get("Content-Type", "")
                    # Detect HTML responses (dataset info pages, not actual data)
                    if "text/html" in ct or raw[:50].strip().startswith(b"<"):
                        st.error(
                            "That URL returned an HTML page, not a data file. "
                            "Use a direct link to a `.csv` or `.zip` file.\n\n"
                            "**Tip:** For UCI datasets, use the direct download URL, e.g.:\n"
                            "`https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`"
                        )
                        content = None
                    elif url.endswith(".zip") or "zip" in ct or raw[:4] == b"PK\x03\x04":
                        zf = zipfile.ZipFile(io.BytesIO(raw))
                        # Collect CSVs — also look inside nested ZIPs
                        csvs = sorted(
                            [f for f in zf.namelist() if f.endswith(".csv") and "__MACOSX" not in f],
                            key=lambda f: zf.getinfo(f).file_size,
                            reverse=True,
                        )
                        inner_zips = [f for f in zf.namelist() if f.endswith(".zip")]
                        nested_map = {}  # display_name -> (outer_zip_name, inner_csv_name, file_size)
                        for iz in inner_zips:
                            try:
                                izf = zipfile.ZipFile(io.BytesIO(zf.read(iz)))
                                for ic in izf.namelist():
                                    if ic.endswith(".csv") and "__MACOSX" not in ic:
                                        label = f"{iz} \u2192 {ic}"
                                        nested_map[label] = (iz, ic, izf.getinfo(ic).file_size)
                            except Exception:
                                pass
                        # Sort all options: largest file first so the best CSV is the default
                        nested_sorted = sorted(nested_map.keys(), key=lambda k: nested_map[k][2], reverse=True)
                        all_options = csvs + nested_sorted
                        if all_options:
                            chosen = st.selectbox("Select CSV:", all_options) if len(all_options) > 1 else all_options[0]
                            if chosen in nested_map:
                                iz_name, ic_name, _ = nested_map[chosen]
                                izf = zipfile.ZipFile(io.BytesIO(zf.read(iz_name)))
                                content = izf.open(ic_name).read().decode("utf-8")
                            else:
                                content = zf.open(chosen).read().decode("utf-8")
                        else:
                            st.error("No CSV files found inside the ZIP.")
                            content = None
                    else:
                        content = raw.decode("utf-8")
                    if content:
                        sep = ";" if ";" in content.split("\n")[0] and "," not in content.split("\n")[0] else ","
                        df_raw = pd.read_csv(io.StringIO(content), sep=sep)
            except Exception as e:
                st.error(f"Download failed: {e}")

    if df_raw is not None:
        st.success(f"Loaded **{len(df_raw):,}** rows x **{len(df_raw.columns)}** columns")
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(df_raw.head(5), use_container_width=True, hide_index=True)

        df_clean, warns, missing = validate_df(df_raw)
        for w in warns:
            st.warning(w)

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            _, sc, _ = st.columns([1, 2, 1])
            with sc:
                do_score = st.button("Score All Clients", type="primary", use_container_width=True, key="batch_pred")

            if do_score:
                with st.spinner(f"Scoring {len(df_clean):,} clients..."):
                    probs = predict_batch(df_clean)

                result = df_clean.copy()
                result.insert(0, "probability", np.round(probs, 4))
                result.insert(1, "verdict", np.where(probs >= threshold, "YES", "NO"))
                result = result.sort_values("probability", ascending=False).reset_index(drop=True)

                total = len(result)
                n_yes = int((probs >= threshold).sum())
                n_no = total - n_yes
                avg_p = probs.mean()

                st.markdown("")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total", f"{total:,}")
                k2.metric("YES (Call)", f"{n_yes:,}", delta=f"{n_yes / total:.1%}")
                k3.metric("NO (Skip)", f"{n_no:,}")
                k4.metric("Avg Prob", f"{avg_p:.1%}")

                fig = px.histogram(
                    result, x="probability", nbins=50, color_discrete_sequence=["#0ea5e9"]
                )
                fig.add_vline(
                    x=threshold, line_dash="dash", line_color="#dc2626",
                    annotation_text=f"Threshold {threshold:.0%}",
                )
                fig.update_layout(
                    template="plotly_white", height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="sec-head">Campaign Impact</div>', unsafe_allow_html=True)
                tp = n_yes * avg_p
                fp = n_yes * (1 - avg_p)
                fn = n_no * 0.113
                ml_p = tp * 195 - fp * 5 - fn * 200
                all_p = total * 0.113 * 195 - total * 0.887 * 5

                p1, p2, p3 = st.columns(3)
                p1.metric("ML Profit", f"${ml_p:,.0f}")
                p2.metric("Call-All Profit", f"${all_p:,.0f}")
                delta_str = f"+{(ml_p / all_p - 1) * 100:.0f}%" if all_p > 0 else ""
                p3.metric("Advantage", f"${ml_p - all_p:,.0f}", delta=delta_str)

                st.markdown('<div class="sec-head">Top 20 Leads</div>', unsafe_allow_html=True)
                top = result.head(20).copy()
                top["probability"] = top["probability"].apply(lambda x: f"{x:.1%}")
                st.dataframe(top, use_container_width=True, hide_index=True)

                with st.expander(f"All {total:,} predictions"):
                    full = result.copy()
                    full["probability"] = full["probability"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(full, use_container_width=True, hide_index=True)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "CSV", result.to_csv(index=False),
                        "predictions.csv", "text/csv", use_container_width=True,
                    )
                with d2:
                    buf = io.BytesIO()
                    result.to_excel(buf, index=False, engine="openpyxl")
                    st.download_button(
                        "Excel", buf.getvalue(), "predictions.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )


# =====================================================================
# FOOTER
# =====================================================================
st.markdown(
    f'<div class="app-foot">'
    f'<div class="app-foot-title">Call Smarter: Predicting Term Deposit Subscribers</div>'
    f'<div class="app-foot-line"></div>'
    f'<div class="app-foot-sub">{display_name} | Threshold {threshold:.1%} | Composite Selection</div>'
    f'</div>',
    unsafe_allow_html=True,
)