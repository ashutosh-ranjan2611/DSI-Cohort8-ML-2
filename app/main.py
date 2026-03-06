"""
Call Smarter: Predicting Term Deposit Subscribers - v9
Clean rewrite. No broken strings. Tested structure.
"""

from __future__ import annotations
import sys, io, json, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle, requests
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

.banner{position:relative;width:100%;height:200px;border-radius:18px;overflow:hidden;margin-bottom:1rem;box-shadow:0 4px 20px rgba(0,0,0,0.1)}
.banner img{width:100%;height:100%;object-fit:cover;display:block;filter:brightness(0.5)}
.banner-overlay{position:absolute;inset:0;background:linear-gradient(90deg,rgba(15,23,42,0.88) 0%,rgba(15,23,42,0.5) 50%,rgba(15,23,42,0.15) 100%);display:flex;flex-direction:column;justify-content:center;padding:1.5rem 2.5rem;color:white}
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

.verd-yes{background:linear-gradient(135deg,#059669,#34d399);color:white;padding:1.8rem;border-radius:18px;text-align:center;box-shadow:0 6px 24px rgba(5,150,105,0.25)}
.verd-no{background:linear-gradient(135deg,#dc2626,#f87171);color:white;padding:1.8rem;border-radius:18px;text-align:center;box-shadow:0 6px 24px rgba(220,38,38,0.25)}
.verd-title{font-size:1.8rem;font-weight:900;margin:0}
.verd-prob{font-size:3rem;font-weight:900;margin:0.3rem 0}
.verd-desc{font-size:0.9rem;opacity:0.9;margin:0}

.pill-pos{display:inline-block;background:#f0fdf4;color:#166534;padding:5px 12px;border-radius:16px;font-size:0.8rem;font-weight:600;margin:3px;border:1px solid #bbf7d0}
.pill-neg{display:inline-block;background:#fef2f2;color:#991b1b;padding:5px 12px;border-radius:16px;font-size:0.8rem;font-weight:600;margin:3px;border:1px solid #fecaca}

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
    if is_ensemble:
        return model["voter"].predict_proba(model["preprocess"].transform(df))[:, 1]
    return model.predict_proba(df)[:, 1]


def get_verdict(prob):
    if prob >= threshold:
        return (
            True,
            "YES - Subscribe",
            f"{prob:.0%} probability of subscribing. Above the {threshold:.0%} threshold.",
        )
    return (
        False,
        "NO - Will Not Subscribe",
        f"Only {prob:.0%} probability. Below the {threshold:.0%} threshold.",
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


# =====================================================================
# BANNER IMAGE
# =====================================================================
BANNER_URL = "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=1400&h=400&fit=crop&crop=faces"

st.markdown(
    f'<div class="banner">'
    f'<img src="{BANNER_URL}" alt="Term Deposit Banner"/>'
    f'<div class="banner-overlay">'
    f'<div class="banner-tag">Term Deposit Subscription Predictor</div>'
    f'<div class="banner-title">Grow Your Savings with<br>Guaranteed Returns</div>'
    f'<div class="banner-desc">Identify high-potential clients for term deposit campaigns '
    f'using machine learning - call smarter, convert more, waste less.</div>'
    f'</div></div>',
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

st.markdown(
    f'<div class="hero-wrap">'
    f'<div class="hero-title">Call Smarter: Predicting Term Deposit Subscribers</div>'
    f'<div class="hero-sub">ML-Powered Campaign Intelligence</div>'
    f'<div class="hero-badges">'
    f'<span class="hero-badge">{display_name}</span>'
    f'<span class="hero-badge">{threshold:.1%} Threshold</span>'
    f'<span class="hero-badge">{crit_display}{composite}</span>'
    f'</div>'
    f'<div class="hero-kpis">'
    f'<div class="hero-kpi"><div class="hero-kpi-val">{best_recall or "N/A"}</div><div class="hero-kpi-lbl">Recall</div></div>'
    f'<div class="hero-kpi"><div class="hero-kpi-val">{best_auc or "N/A"}</div><div class="hero-kpi-lbl">AUC-ROC</div></div>'
    f'<div class="hero-kpi"><div class="hero-kpi-val">{best_profit or "N/A"}</div><div class="hero-kpi-lbl">Net Profit</div></div>'
    f'<div class="hero-kpi"><div class="hero-kpi-val">{threshold:.0%}</div><div class="hero-kpi-lbl">Threshold</div></div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


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
            yes, vtext, vdesc = get_verdict(prob)
            pf, nf = get_factors(inp, prob)

            st.markdown("")
            cls = "verd-yes" if yes else "verd-no"
            st.markdown(
                f'<div class="{cls}">'
                f'<p class="verd-title">{vtext}</p>'
                f'<p class="verd-prob">{prob:.1%}</p>'
                f'<p class="verd-desc">{vdesc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability", f"{prob:.1%}")
            m2.metric("Threshold", f"{threshold:.1%}")
            mg = prob - threshold
            m3.metric("Margin", f"{mg:+.1%}", delta=f"{mg:+.1%}")

            st.markdown('<div class="sec-head">Key Decision Factors</div>', unsafe_allow_html=True)
            html = ""
            for f in pf:
                html += f'<span class="pill-pos">{f}</span>'
            for f in nf:
                html += f'<span class="pill-neg">{f}</span>'
            if not pf and not nf:
                html = '<span style="color:#64748b;font-size:0.85rem;">No strong signals</span>'
            st.markdown(f'<div style="margin-bottom:1rem;">{html}</div>', unsafe_allow_html=True)

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
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
                    number={"suffix": "%", "font": {"size": 30, "color": "#1e293b"}},
                )
            )
            fig.update_layout(
                height=220,
                margin=dict(t=20, b=5, l=25, r=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

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
        url = st.text_input("Dataset URL", placeholder="https://archive.ics.uci.edu/ml/...", key="url_in2")
        if url and url.startswith("http"):
            try:
                with st.spinner("Fetching dataset..."):
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    raw = resp.content
                    ct = resp.headers.get("Content-Type", "")
                    if url.endswith(".zip") or "zip" in ct:
                        zf = zipfile.ZipFile(io.BytesIO(raw))
                        csvs = sorted(
                            [f for f in zf.namelist() if f.endswith(".csv") and "__MACOSX" not in f],
                            key=lambda f: zf.getinfo(f).file_size,
                            reverse=True,
                        )
                        if csvs:
                            chosen = st.selectbox("Select CSV:", csvs) if len(csvs) > 1 else csvs[0]
                            content = zf.open(chosen).read().decode("utf-8")
                        else:
                            st.error("No CSV in ZIP")
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