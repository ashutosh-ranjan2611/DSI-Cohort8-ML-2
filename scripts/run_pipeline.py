#!/usr/bin/env python3
"""
PIPELINE v5.0 — End-to-end bank marketing ML pipeline
=======================================================
What this script does (in order):
  Step 1-3  : Download data, clean it, split into train/val/test
  Step 5    : Rank features by importance (Random Forest on 5-fold CV)
  Step 5c   : Train 4 models with default settings (before tuning baseline)
  Step 6-7  : Tune each model with Optuna, evaluate on test set, pick the best
  Step 7b   : Show before vs after tuning comparison

The 4 models we compare:
  LR  = Logistic Regression  (fast, explainable baseline)
  RF  = Random Forest         (many trees voting together)
  XGB = XGBoost               (sequential trees, each corrects the last)
  KNN = K-Nearest Neighbours  ("what did similar customers do?")

Model selection: pick the model with the highest net business profit.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --n-trials 30
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse, json, logging, pickle, time, sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from xgboost import XGBClassifier

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import download_and_extract, load_raw_data
from src.clean import clean_data
from src.split import stratified_split, save_splits
from src.features import (
    TARGET,
    build_pipeline,
    PdaysTransformer,
)
from src.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    business_cost_analysis,
    select_best_model,
    recall_analysis,
    get_cost_derivation_text,
    BANKING_ECONOMICS,
)
from src.train import tune_model, train_final_model

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MET_DIR = ROOT / "reports" / "metrics"
MOD_DIR = ROOT / "models"
for d in [MET_DIR, MOD_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
class PrettyFormatter(logging.Formatter):
    GREY = "\033[90m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED + BOLD,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.GREY)
        ts = self.formatTime(record, "%H:%M:%S")
        return f"{self.GREY}{ts}{self.RESET} {color}│{self.RESET} {record.getMessage()}"


handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
log = logging.getLogger("pipeline")


def banner(title, emoji="═"):
    w = 65
    log.info("")
    log.info(f"\033[1m\033[96m{'═' * w}\033[0m")
    log.info(f"\033[1m\033[96m  {emoji}  {title}\033[0m")
    log.info(f"\033[1m\033[96m{'═' * w}\033[0m")


def step_done(msg):
    log.info(f"  \033[92m[OK] {msg}\033[0m")


def metric_log(label, value):
    log.info(f"     \033[93m▸ {label}: {value}\033[0m")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1–3 : DATA
# ═══════════════════════════════════════════════════════════════════════════════

# ANSI helpers reused across tables
_C  = "\033[96m"   # cyan  — headers / borders
_Y  = "\033[93m"   # yellow — values
_G  = "\033[92m"   # green  — OK / pass
_W  = "\033[93m"   # warning
_B  = "\033[1m"    # bold
_R  = "\033[0m"    # reset


def _tbl_row(*cells, widths, sep="│"):
    """Render one table row with right-padded cells."""
    body = f" {sep} ".join(
        f"{_Y}{str(c):<{w}}{_R}" for c, w in zip(cells, widths)
    )
    return f"  {_C}{sep}{_R} {body} {_C}{sep}{_R}"


def _tbl_div(widths, left="├", mid="┼", right="┤", fill="─"):
    segments = (fill * (w + 2) for w in widths)
    return f"  {_C}{left}" + f"{mid}".join(segments) + f"{right}{_R}"


def _tbl_head(headers, widths):
    body = f" {_C}│{_R} ".join(
        f"{_B}{_C}{h:<{w}}{_R}" for h, w in zip(headers, widths)
    )
    return f"  {_C}│{_R} {body} {_C}│{_R}"


def _tbl_top(widths):
    segments = ("─" * (w + 2) for w in widths)
    return f"  {_C}┌" + "┬".join(segments) + f"┐{_R}"


def _tbl_bot(widths):
    segments = ("─" * (w + 2) for w in widths)
    return f"  {_C}└" + "┴".join(segments) + f"┘{_R}"


def _section(title: str):
    log.info(f"  {_B}{_C}▸ {title}{_R}")


def _print_data_report(df_raw, df, tr, va, te):
    """Print a structured tabular summary of the full ingest → clean → split pipeline."""

    IMPUTE_COLS = ["job", "marital", "housing", "loan"]
    KEEP_COLS   = ["education", "default"]
    OUTLIER_COLS_CHK = [
        "age", "campaign", "pdays", "previous",
        "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
    ]
    SKEW_COLS = ["age", "campaign", "previous", "emp.var.rate",
                 "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    log.info("")

    # ── 1. Dataset Overview ──────────────────────────────────────────────────
    _section("Dataset Overview")
    duplicates = len(df_raw) - len(df)
    w = [18, 12]
    log.info(_tbl_top(w))
    log.info(_tbl_head(["Metric", "Value"], w))
    log.info(_tbl_div(w))
    for label, val in [
        ("Raw records",      f"{len(df_raw):,}"),
        ("After de-dup",     f"{len(df):,}"),
        ("Duplicates removed", f"{duplicates:,}"),
        ("Features (prod)",  str(len(df.columns))),
        ("Positive rate",    f"{df[TARGET].mean():.1%}"),
    ]:
        log.info(_tbl_row(label, val, widths=w))
    log.info(_tbl_bot(w))
    log.info("")

    # ── 2. Data Splits ───────────────────────────────────────────────────────
    _section("Stratified Split  (70 / 15 / 15)")
    w = [8, 10, 14]
    log.info(_tbl_top(w))
    log.info(_tbl_head(["Split", "Rows", "Positive Rate"], w))
    log.info(_tbl_div(w))
    for name, split in [("Train", tr), ("Val", va), ("Test", te)]:
        log.info(_tbl_row(name, f"{len(split):,}", f"{split[TARGET].mean():.1%}", widths=w))
    log.info(_tbl_bot(w))
    log.info("")

    # ── 3. Cardinality ───────────────────────────────────────────────────────
    _section("Categorical Cardinality")
    cat_cols = df_raw.select_dtypes(include="object").columns.tolist()
    w = [18, 8, 10, 24]
    log.info(_tbl_top(w))
    log.info(_tbl_head(["Column", "Levels", "Min Count", "Flag"], w))
    log.info(_tbl_div(w))
    for col in cat_cols:
        if col == "duration":
            continue
        vc = df_raw[col].value_counts()
        flag = f"rare: '{vc.idxmin()}' ({vc.min()})" if vc.min() < 50 else "OK"
        log.info(_tbl_row(col, vc.nunique(), f"{vc.min():,}", flag, widths=w))
    log.info(_tbl_bot(w))
    log.info("")

    # ── 4. Outlier Clipping ──────────────────────────────────────────────────
    _section("Outlier Clipping  (1st / 99th percentile)")
    available = [c for c in OUTLIER_COLS_CHK if c in df_raw.columns]
    w = [18, 10, 7, 10, 10]
    log.info(_tbl_top(w))
    log.info(_tbl_head(["Column", "Outliers", "Pct", "Low Bound", "High Bound"], w))
    log.info(_tbl_div(w))
    for col in available:
        q01, q99 = df_raw[col].quantile(0.01), df_raw[col].quantile(0.99)
        n = int(((df_raw[col] < q01) | (df_raw[col] > q99)).sum())
        pct = n / len(df_raw) * 100
        log.info(_tbl_row(col, f"{n:,}", f"{pct:.1f}%", f"{q01:.2f}", f"{q99:.2f}", widths=w))
    log.info(_tbl_bot(w))
    log.info("")

    # ── 5. Skewness ──────────────────────────────────────────────────────────
    _section("Skewness & Kurtosis")
    available_sk = [c for c in SKEW_COLS if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    w = [18, 9, 11, 10]
    log.info(_tbl_top(w))
    log.info(_tbl_head(["Column", "Skewness", "Kurtosis", "Status"], w))
    log.info(_tbl_div(w))
    for col in available_sk:
        sk, ku = df[col].skew(), df[col].kurtosis()
        status = "HIGH" if abs(sk) > 2 else "OK"
        log.info(_tbl_row(col, f"{sk:+.2f}", f"{ku:.2f}", status, widths=w))
    log.info(_tbl_bot(w))
    log.info("")

    # ── 6. Multicollinearity ─────────────────────────────────────────────────
    _section("Multicollinearity  (|r| > 0.80)")
    num_cols = [c for c in SKEW_COLS if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    corr = df[num_cols].corr().abs()
    pairs = [
        (num_cols[i], num_cols[j], corr.iloc[i, j])
        for i in range(len(num_cols))
        for j in range(i + 1, len(num_cols))
        if corr.iloc[i, j] > 0.8
    ]
    w = [18, 18, 8, 30]
    log.info(_tbl_top(w))
    log.info(_tbl_head(["Feature A", "Feature B", "|r|", "Mitigation"], w))
    log.info(_tbl_div(w))
    mitigation = "L1 (LogReg) + tree invariance"
    for f1, f2, r in pairs:
        log.info(_tbl_row(f1, f2, f"{r:.3f}", mitigation, widths=w))
    if not pairs:
        log.info(_tbl_row("—", "No high-correlation pairs found", "", "", widths=w))
    log.info(_tbl_bot(w))
    log.info("")

    # ── 7. Unknown Handling ──────────────────────────────────────────────────
    _section("Unknown Value Handling")
    w = [18, 10, 12, 24]
    log.info(_tbl_top(w))
    log.info(_tbl_head(["Column", "Unknowns", "Strategy", "Mode / Category"], w))
    log.info(_tbl_div(w))
    for col in IMPUTE_COLS:
        if col in df_raw.columns:
            n = int((df_raw[col] == "unknown").sum())
            mode_val = df_raw.loc[df_raw[col] != "unknown", col].mode()[0] if n else "—"
            log.info(_tbl_row(col, f"{n:,}", "Impute (mode)", f"→ '{mode_val}'", widths=w))
    for col in KEEP_COLS:
        if col in df_raw.columns:
            n = int((df_raw[col] == "unknown").sum())
            log.info(_tbl_row(col, f"{n:,}", "Keep as cat.", "informative signal", widths=w))
    log.info(_tbl_bot(w))
    log.info("")


def step_data():
    banner("STEP 1–3  ·  INGEST -> CLEAN -> SPLIT", "")

    # Suppress per-line chatter from src modules — tables replace it below
    for mod in ("src.ingest", "src.clean", "src.split"):
        logging.getLogger(mod).setLevel(logging.WARNING)

    download_and_extract()
    df_raw = load_raw_data()
    df = clean_data(df_raw, production=True)
    tr, va, te = stratified_split(df)
    save_splits(tr, va, te)

    _print_data_report(df_raw, df, tr, va, te)
    return df_raw, df, tr, va, te


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 : FEATURE IMPORTANCE + SWEEP (FIXED — no data leakage)
# ═══════════════════════════════════════════════════════════════════════════════
def step_feature_importance(X_train, y_train):
    banner("STEP 5  ·  FEATURE IMPORTANCE (leakage-free)", "")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    importances_all = []
    feat_names = None

    for fold_idx, (train_idx, _) in enumerate(cv.split(X_train, y_train)):
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]

        rf_pipe = build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )
        rf_pipe.fit(X_fold, y_fold)

        if feat_names is None:
            preprocessor = rf_pipe.named_steps["preprocessor"]
            try:
                feat_names = [str(f) for f in preprocessor.get_feature_names_out()]
            except Exception:
                steps_before_clf = [s for s in rf_pipe.steps if s[0] != "classifier"]
                pre = SkPipeline(steps_before_clf)
                n = pre.transform(X_fold[:1]).shape[1]
                feat_names = [f"feature_{i}" for i in range(n)]

        importances_all.append(rf_pipe.named_steps["classifier"].feature_importances_)

    avg_importances = np.mean(importances_all, axis=0)
    imp_df = pd.DataFrame(
        {"feature": feat_names, "importance": avg_importances}
    ).sort_values("importance", ascending=False)


    sorted_features = imp_df["feature"].tolist()
    log.info(f"  Total features: {len(sorted_features)}")
    log.info(f"  Top-3: {', '.join(sorted_features[:3])}")
    return feat_names, sorted_features, imp_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5c : BASELINE (Before Tuning) — Default Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════
def step_baseline_models(X_tr, y_tr, X_te, y_te):
    """
    Train all 4 models with DEFAULT hyperparameters to establish a baseline.

    This runs BEFORE Optuna tuning, so we can show a clear before/after comparison
    that demonstrates the value of hyperparameter search.
    """
    banner("STEP 5c  ·  BASELINE MODELS (Before Tuning)", "")

    baseline_results = {}
    default_configs = {
        "logistic_regression": LogisticRegression(
            solver="saga",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            scale_pos_weight=round((y_tr == 0).sum() / (y_tr == 1).sum(), 2),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="aucpr",
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=11,
            weights="uniform",
            n_jobs=-1,
        ),
    }

    for name, clf in default_configs.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_tr, y_tr)

        y_te_prob = pipe.predict_proba(X_te)[:, 1]
        y_te_pred = (y_te_prob >= 0.5).astype(int)  # Default threshold = 0.5
        auc = roc_auc_score(y_te, y_te_prob)
        logloss = log_loss(y_te, y_te_prob)
        brier = brier_score_loss(y_te, y_te_prob)
        tn, fp, fn, tp = confusion_matrix(y_te, y_te_pred).ravel()
        profit = tp * 195 - fp * 5 - fn * 200

        baseline_results[name] = {
            "auc": auc,
            "log_loss": logloss,
            "brier": brier,
            "profit": profit,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        }

    # ── Baseline summary table ────────────────────────────────────────────
    _section("Baseline Results  (default hyperparameters, threshold = 0.5)")
    w_bl = [22, 8, 9, 8, 13, 9, 10]
    log.info(_tbl_top(w_bl))
    log.info(_tbl_head(
        ["Model", "AUC", "LogLoss", "Brier", "Net Profit", "Recall", "Precision"],
        w_bl,
    ))
    log.info(_tbl_div(w_bl))
    for _nm, _br in baseline_results.items():
        log.info(_tbl_row(
            _nm,
            f"{_br['auc']:.4f}",
            f"{_br['log_loss']:.4f}",
            f"{_br['brier']:.4f}",
            f"${_br['profit']:,.0f}",
            f"{_br['recall']:.3f}",
            f"{_br['precision']:.3f}",
            widths=w_bl,
        ))
    log.info(_tbl_bot(w_bl))

    return baseline_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6–7 : TRAIN + EVALUATE + COMPOSITE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def step_train_and_evaluate(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials):
    banner("STEP 6–7  ·  TRAINING + MODEL SELECTION", "")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    all_metrics = []
    pipelines = {}

    # ── All 4 models ──────────────────────────────────────────────────────
    for name in ["logistic_regression", "random_forest", "xgboost", "knn"]:
        log.info("")
        log.info(f"  \033[1mTraining: {name} ({n_trials} Optuna trials)\033[0m")

        tune_result = tune_model(name, X_tr, y_tr, n_trials=n_trials)
        pipeline = train_final_model(name, tune_result["best_params"], X_tr, y_tr)
        pipelines[name] = pipeline

        y_va_prob = pipeline.predict_proba(X_va)[:, 1]
        opt_t, _ = find_optimal_threshold(y_va, y_va_prob)
        y_te_prob = pipeline.predict_proba(X_te)[:, 1]
        y_te_pred = (y_te_prob >= opt_t).astype(int)
        metrics = compute_metrics(y_te, y_te_pred, y_te_prob)
        cost = business_cost_analysis(y_te, y_te_pred)

        results[name] = {
            "params": tune_result["best_params"],
            "cv_auc": tune_result["best_auc"],
            "threshold": opt_t,
            "test_metrics": metrics,
            "cost": cost,
            "y_te_prob": y_te_prob,
            "y_te_pred": y_te_pred,
        }
        row = {
            "model": name,
            "cv_auc": round(tune_result["best_auc"], 4),
            "threshold": round(opt_t, 3),
        }
        row.update({f"test_{k}": round(v, 4) for k, v in metrics.items()})
        row["net_profit"] = round(cost["net_profit"], 0)
        row["brier_score"] = round(metrics["brier_score"], 4)
        all_metrics.append(row)

        with open(MOD_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(pipeline, f)

    # ── Training Results Summary table ─────────────────────────────────────
    _section("Training Results Summary")
    w_tr = [22, 7, 9, 8, 8, 8, 8, 12]
    log.info(_tbl_top(w_tr))
    log.info(_tbl_head(
        ["Model", "CV AUC", "Thresh", "Recall", "F1", "ROC-AUC", "Brier", "Net Profit"],
        w_tr,
    ))
    log.info(_tbl_div(w_tr))
    for _rm in all_metrics:
        _cv = f"{_rm['cv_auc']:.4f}" if not (
            isinstance(_rm.get('cv_auc'), float)
            and np.isnan(_rm.get('cv_auc', float('nan')))
        ) else "  —  "
        log.info(_tbl_row(
            _rm["model"], _cv,
            f"{_rm['threshold']:.3f}",
            f"{_rm['test_recall']:.3f}",
            f"{_rm['test_f1']:.4f}",
            f"{_rm['test_roc_auc']:.4f}",
            f"{_rm['brier_score']:.4f}",
            f"${_rm['net_profit']:,.0f}",
            widths=w_tr,
        ))
    log.info(_tbl_bot(w_tr))

    # ── MODEL SELECTION: pick the model with the highest net profit ──────────
    comp_df = pd.DataFrame(all_metrics)

    banner("MODEL SELECTION", "")
    # Simple rule: whichever model made the most net profit on the test set wins.
    # Net profit already reflects recall (FN = $200 loss) and precision (FP = $5 cost),
    # so it is the best single criterion for this business problem.
    best_name, sorted_df = select_best_model(comp_df)

    # Save results
    sorted_df.to_json(MET_DIR / "comparison.json", orient="records", indent=2)
    sorted_df.to_csv(MET_DIR / "comparison.csv", index=False)

    best_t = results[best_name]["threshold"]
    with open(MOD_DIR / "threshold.json", "w") as f:
        json.dump(
            {
                "model": best_name,
                "threshold": round(best_t, 4),
                "selection_criterion": "max_net_profit",
                "banking_economics": BANKING_ECONOMICS,
            },
            f,
            indent=2,
        )
    log.info(f"\n  Selected: \033[1m{best_name}\033[0m (highest net profit on test set)")

    # ── Recall analysis table ────────────────────────────────────────────────
    banner("RECALL ANALYSIS — Stakeholder View", "")
    best_probs = results[best_name]["y_te_prob"]
    recall_df = recall_analysis(y_te, best_probs)
    recall_df.to_csv(MET_DIR / "recall_analysis.csv", index=False)
    _section("Recall Trade-off  (key thresholds)")
    w_rc = [10, 8, 12, 8, 8, 12]
    log.info(_tbl_top(w_rc))
    log.info(_tbl_head(
        ["Threshold", "Recall", "Calls Made", "Caught", "Missed", "Net Profit"],
        w_rc,
    ))
    log.info(_tbl_div(w_rc))
    for _, r in recall_df.iterrows():
        if r["threshold"] in [0.10, 0.20, 0.30, 0.40, 0.50]:
            log.info(_tbl_row(
                f"{r['threshold']:.2f}",
                f"{r['recall']:.3f}",
                f"{r['calls_made']:,}",
                str(int(r['subscribers_caught'])),
                str(int(r['subscribers_missed'])),
                f"${r['net_profit']:,.0f}",
                widths=w_rc,
            ))
    log.info(_tbl_bot(w_rc))
    log.info("  Full table saved -> recall_analysis.csv")

    return results, pipelines, sorted_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7b : BEFORE vs AFTER TUNING COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def step_tuning_comparison(baseline_results, tuned_results, y_te):
    """Compare baseline (default params) vs tuned (Optuna) performance."""
    banner("STEP 7b  ·  BEFORE vs AFTER HYPERPARAMETER TUNING", "")

    comparison_rows = []
    for name in baseline_results:
        if name not in tuned_results:
            continue
        base = baseline_results[name]
        tuned = tuned_results[name]
        tuned_metrics = tuned["test_metrics"]
        tuned_cost = tuned["cost"]

        # Compute tuned log-loss
        tuned_logloss = log_loss(y_te, tuned["y_te_prob"])

        row = {
            "model": name,
            "baseline_auc": round(base["auc"], 4),
            "tuned_auc": round(tuned_metrics["roc_auc"], 4),
            "auc_gain": round(tuned_metrics["roc_auc"] - base["auc"], 4),
            "baseline_logloss": round(base["log_loss"], 4),
            "tuned_logloss": round(tuned_logloss, 4),
            "logloss_reduction": round(base["log_loss"] - tuned_logloss, 4),
            "baseline_brier": round(base["brier"], 4),
            "tuned_brier": round(tuned_metrics["brier_score"], 4),
            "brier_reduction": round(base["brier"] - tuned_metrics["brier_score"], 4),
            "baseline_profit": round(base["profit"], 0),
            "tuned_profit": round(tuned_cost["net_profit"], 0),
            "profit_gain": round(tuned_cost["net_profit"] - base["profit"], 0),
        }
        comparison_rows.append(row)

    # ── Tuning comparison table ───────────────────────────────────────────
    _section("Before vs After Hyperparameter Tuning  (Optuna)")
    w_tc = [22, 9, 9, 9, 10, 10, 12]
    log.info(_tbl_top(w_tc))
    log.info(_tbl_head(
        ["Model", "Base AUC", "Tuned AUC", "AUC Gain", "Base Loss", "Tuned Loss", "Profit Gain"],
        w_tc,
    ))
    log.info(_tbl_div(w_tc))
    for _tr in comparison_rows:
        log.info(_tbl_row(
            _tr["model"],
            f"{_tr['baseline_auc']:.4f}",
            f"{_tr['tuned_auc']:.4f}",
            f"{_tr['auc_gain']:+.4f}",
            f"{_tr['baseline_logloss']:.4f}",
            f"{_tr['tuned_logloss']:.4f}",
            f"${_tr['profit_gain']:+,.0f}",
            widths=w_tc,
        ))
    log.info(_tbl_bot(w_tc))

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(MET_DIR / "tuning_comparison.csv", index=False)


    return comp_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Bank Marketing ML Pipeline v5.0")
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    start = time.time()
    n_models = 4  # LR, RF, XGB, KNN

    log.info("")
    log.info(
        "\033[1m\033[95m╔═══════════════════════════════════════════════════════════════╗\033[0m"
    )
    log.info(
        "\033[1m\033[95m║   BANK MARKETING ML PIPELINE v5.0 (Production)               ║\033[0m"
    )
    log.info(
        "\033[1m\033[95m║   Models: LR · RF · XGB · KNN                                    ║\033[0m"
    )
    log.info(
        "\033[1m\033[95m║   Metrics: Acc · Prec · Rec · F1 · AUC · PR-AUC · LL · MCC  ║\033[0m"
    )
    log.info(
        "\033[1m\033[95m╚═══════════════════════════════════════════════════════════════╝\033[0m"
    )
    log.info(f"  Trials: {args.n_trials}  |  Models: {n_models}")

    # 1–3: Data (now includes duplicate removal + outlier clipping)
    df_raw, df, df_train, df_val, df_test = step_data()
    X_tr, y_tr = df_train.drop(columns=[TARGET]), df_train[TARGET]
    X_va, y_va = df_val.drop(columns=[TARGET]), df_val[TARGET]
    X_te, y_te = df_test.drop(columns=[TARGET]), df_test[TARGET]

    # 5: Feature importance
    feat_names, sorted_features, imp_df = step_feature_importance(X_tr, y_tr)

    # 5c: Baseline models (BEFORE tuning) — for comparison
    baseline_results = step_baseline_models(X_tr, y_tr, X_te, y_te)

    # 6–7: Train with Optuna + model selection
    results, pipelines, comp_df = step_train_and_evaluate(
        X_tr, y_tr, X_va, y_va, X_te, y_te, args.n_trials
    )

    # 7b: Before vs After tuning comparison
    step_tuning_comparison(baseline_results, results, y_te)

    # Summary
    elapsed = time.time() - start

    log.info("")
    log.info(
        "\033[1m\033[92m╔═══════════════════════════════════════════════════════════════╗\033[0m"
    )
    log.info(
        f"\033[1m\033[92m║   PIPELINE v5.0 COMPLETE in {elapsed / 60:.1f} minutes{' ' * max(0, 29 - len(f'{elapsed / 60:.1f}'))}║\033[0m"
    )
    log.info(f"\033[1m\033[92m║   Models  -> {str(MOD_DIR)[:45]:<45s} ║\033[0m")
    log.info(f"\033[1m\033[92m║   Metrics -> {str(MET_DIR)[:45]:<45s} ║\033[0m")
    log.info(
        "\033[1m\033[92m╚═══════════════════════════════════════════════════════════════╝\033[0m"
    )

    # ── Results Summary Table ────────────────────────────────────────────────
    log.info("")
    log.info(f"  {_B}{_C}{'═' * 73}{_R}")
    log.info(f"  {_B}{_C}  RESULTS SUMMARY{_R}")
    log.info(f"  {_B}{_C}{'═' * 73}{_R}")
    log.info("")

    # ── Per-model metrics table ──────────────────────────────────────────────
    _section("Model Comparison")
    w_m = [22, 7, 9, 9, 10, 7, 9, 8, 14]
    log.info(_tbl_top(w_m))
    log.info(_tbl_head(
        ["Model", "CV AUC", "Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        w_m,
    ))
    log.info(_tbl_div(w_m))
    for _, row in comp_df.iterrows():
        cv = f"{row['cv_auc']:.4f}" if not (isinstance(row.get('cv_auc'), float) and np.isnan(row.get('cv_auc', float('nan')))) else "—"
        log.info(_tbl_row(
            row["model"], cv,
            f"{row['threshold']:.3f}",
            f"{row['test_accuracy']:.4f}",
            f"{row['test_precision']:.4f}",
            f"{row['test_recall']:.4f}",
            f"{row['test_f1']:.4f}",
            f"{row['test_roc_auc']:.4f}",
            f"{row['test_pr_auc']:.4f}",
            widths=w_m,
        ))
    log.info(_tbl_bot(w_m))
    log.info("")

    # ── Full All-Models × All-Metrics table ──────────────────────────────────
    _section("Full Metrics Comparison  (all models × all metrics)")
    w_full = [22, 9, 10, 7, 9, 8, 10, 9, 8, 14]
    log.info(_tbl_top(w_full))
    log.info(_tbl_head(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Log Loss", "MCC", "Net Profit"],
        w_full,
    ))
    log.info(_tbl_div(w_full))
    for _, row in comp_df.iterrows():
        log.info(_tbl_row(
            row["model"],
            f"{row['test_accuracy']:.4f}",
            f"{row['test_precision']:.4f}",
            f"{row['test_recall']:.4f}",
            f"{row['test_f1']:.4f}",
            f"{row['test_roc_auc']:.4f}",
            f"{row['test_pr_auc']:.4f}",
            f"{row.get('test_log_loss', float('nan')):.4f}" if 'test_log_loss' in row else "   —  ",
            f"{row.get('test_mcc', float('nan')):.4f}" if 'test_mcc' in row else "   —  ",
            f"${row['net_profit']:,.0f}",
            widths=w_full,
        ))
    log.info(_tbl_bot(w_full))
    log.info("")

    # ── Best Model per Criterion ──────────────────────────────────────────────
    _section("Best Model per Criterion  (separate selection view)")
    criteria = {
        "ROC-AUC (highest)":   comp_df.loc[comp_df["test_roc_auc"].idxmax()],
        "PR-AUC (highest)":    comp_df.loc[comp_df["test_pr_auc"].idxmax()],
        "Recall (highest)":    comp_df.loc[comp_df["test_recall"].idxmax()],
        "Precision (highest)": comp_df.loc[comp_df["test_precision"].idxmax()],
        "F1 (highest)":        comp_df.loc[comp_df["test_f1"].idxmax()],
        "Accuracy (highest)":  comp_df.loc[comp_df["test_accuracy"].idxmax()],
        "Net Profit (highest)": comp_df.loc[comp_df["net_profit"].idxmax()],
        "Brier Score (lowest)": comp_df.loc[comp_df["brier_score"].idxmin()],
    }
    if "test_log_loss" in comp_df.columns:
        criteria["Log Loss (lowest)"] = comp_df.loc[comp_df["test_log_loss"].idxmin()]
    if "test_mcc" in comp_df.columns:
        criteria["MCC (highest)"] = comp_df.loc[comp_df["test_mcc"].idxmax()]

    w_bc = [26, 22, 8, 8, 8, 8, 12]
    log.info(_tbl_top(w_bc))
    log.info(_tbl_head(
        ["Criterion", "Best Model", "Recall", "F1", "AUC", "MCC", "Net Profit"],
        w_bc,
    ))
    log.info(_tbl_div(w_bc))
    for criterion, best_row in criteria.items():
        mcc_val = f"{best_row.get('test_mcc', float('nan')):.3f}" if 'test_mcc' in best_row else "—"
        log.info(_tbl_row(
            criterion,
            best_row["model"],
            f"{best_row['test_recall']:.3f}",
            f"{best_row['test_f1']:.4f}",
            f"{best_row['test_roc_auc']:.4f}",
            mcc_val,
            f"${best_row['net_profit']:,.0f}",
            widths=w_bc,
        ))
    log.info(_tbl_bot(w_bc))
    log.info("")

    # ── Business Impact table ────────────────────────────────────────────────
    _section("Business Impact")
    w_b = [22, 12, 8]
    log.info(_tbl_top(w_b))
    log.info(_tbl_head(
        ["Model", "Net Profit", "Brier"],
        w_b,
    ))
    log.info(_tbl_div(w_b))
    for _, row in comp_df.iterrows():
        log.info(_tbl_row(
            row["model"],
            f"${row['net_profit']:,.0f}",
            f"{row['brier_score']:.4f}",
            widths=w_b,
        ))
    log.info(_tbl_bot(w_b))
    log.info("")

    # ── Winner box ───────────────────────────────────────────────────────────
    best = comp_df.iloc[0]
    log.info(f"  {_B}{_C}┌{'─' * 71}┐{_R}")
    log.info(f"  {_B}{_C}│  Winner  : {_Y}{best['model']:<20}{_C}  (highest net profit){' ':>18}│{_R}")
    log.info(f"  {_B}{_C}│  Profit  : {_Y}${best['net_profit']:>10,.0f}{_C}   Recall : {_Y}{best['test_recall']:.3f}{_C}   AUC : {_Y}{best['test_roc_auc']:.4f}{_C}   Brier : {_Y}{best['brier_score']:.4f}{_C}  │{_R}")
    log.info(f"  {_B}{_C}│  Selection criterion : max net profit on test set{' ':>22}│{_R}")
    log.info(f"  {_B}{_C}└{'─' * 71}┘{_R}")
    log.info("")

    # ── Run metadata ─────────────────────────────────────────────────────────
    _section("Run Metadata")
    w_r = [18, 60]
    log.info(_tbl_top(w_r))
    log.info(_tbl_head(["Item", "Value"], w_r))
    log.info(_tbl_div(w_r))
    for label, val in [
        ("Models dir",  str(MOD_DIR)),
        ("Metrics dir", str(MET_DIR)),
        ("Total time",  f"{elapsed / 60:.1f} minutes"),
    ]:
        log.info(_tbl_row(label, val, widths=w_r))
    log.info(_tbl_bot(w_r))


if __name__ == "__main__":
    main()
