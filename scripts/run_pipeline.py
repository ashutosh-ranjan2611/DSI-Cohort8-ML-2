#!/usr/bin/env python3
"""
PIPELINE v4.1 — Final Production
==================================
Changes from v4:
  🐛  Fixed logging format error ($%,.0f → pre-formatted string)
  🔍  Added duplicate detection + outlier clipping in cleaning
  📊  Added EDA figures: outlier boxplots, duplicates summary
  📈  Before/After hyperparameter tuning comparison (baseline vs tuned)
  📉  Loss calculation (log-loss) before and after tuning

Usage:  python scripts/run_pipeline.py
        python scripts/run_pipeline.py --n-trials 30
        python scripts/run_pipeline.py --skip-shap --skip-feature-sweep
        python scripts/run_pipeline.py --no-binning  (ablation study)
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse, json, logging, time, sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline as SkPipeline
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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
    NonLinearBinningTransformer,
)
from src.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    business_cost_analysis,
    select_best_model,
    selection_sensitivity_analysis,
    recall_analysis,
    get_cost_derivation_text,
    BANKING_ECONOMICS,
    SELECTION_WEIGHTS,
)
from src.train import tune_model, train_final_model

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
MET_DIR = ROOT / "reports" / "metrics"
MOD_DIR = ROOT / "models"
for d in [FIG_DIR, MET_DIR, MOD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130


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
    log.info(f"  \033[92m✅ {msg}\033[0m")


def metric_log(label, value):
    log.info(f"     \033[93m▸ {label}: {value}\033[0m")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1–3 : DATA
# ═══════════════════════════════════════════════════════════════════════════════
def step_data():
    banner("STEP 1–3  ·  INGEST → CLEAN → SPLIT", "📦")
    download_and_extract()
    df_raw = load_raw_data()
    df = clean_data(df_raw, production=True)
    tr, va, te = stratified_split(df)
    save_splits(tr, va, te)
    log.info(
        f"  📊 Raw: {len(df_raw):,}  Clean: {len(df):,}  |  Train: {len(tr):,}  Val: {len(va):,}  Test: {len(te):,}"
    )
    log.info(f"  📊 Positive rate: {df[TARGET].mean():.1%}")
    return df_raw, df, tr, va, te


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 : EDA
# ═══════════════════════════════════════════════════════════════════════════════
def step_eda(df_raw, df):
    banner("STEP 4  ·  EDA — Stakeholder-Friendly Figures", "📊")

    # 4a. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = df_raw[TARGET].value_counts()
    colors = ["#e74c3c", "#27ae60"]
    axes[0].bar(
        ["Did NOT Subscribe\n(89%)", "Subscribed\n(11%)"],
        counts.values,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )
    axes[0].set_title(
        "How many clients actually subscribed?", fontsize=14, fontweight="bold"
    )
    axes[0].set_ylabel("Number of Clients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 400, f"{v:,}", ha="center", fontsize=12, fontweight="bold")
    pct = counts / counts.sum() * 100
    axes[1].pie(
        pct.values,
        labels=[f"No\n({pct.iloc[0]:.0f}%)", f"Yes\n({pct.iloc[1]:.0f}%)"],
        colors=colors,
        startangle=90,
        explode=(0, 0.06),
        textprops={"fontsize": 13, "fontweight": "bold"},
        wedgeprops={"edgecolor": "black", "linewidth": 1.2},
    )
    axes[1].set_title("The class imbalance challenge", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close()
    step_done("01_target_distribution.png")

    # 4b. Subscription by job
    fig, ax = plt.subplots(figsize=(10, 7))
    conv = (
        df.groupby("job")[TARGET]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
    )
    conv["mean_pct"] = conv["mean"] * 100
    bars = ax.barh(conv.index, conv["mean_pct"], color="#3498db", edgecolor="black")
    ax.axvline(
        df[TARGET].mean() * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Overall avg",
    )
    ax.set_xlabel("Subscription Rate (%)")
    ax.set_title("Which job types subscribe most?", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    for bar, pct_val in zip(bars, conv["mean_pct"]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{pct_val:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_subscription_by_job.png", bbox_inches="tight")
    plt.close()
    step_done("02_subscription_by_job.png")

    # 4c. Previous campaign
    fig, ax = plt.subplots(figsize=(8, 5))
    pout_conv = df.groupby("poutcome")[TARGET].mean().sort_values() * 100
    colors_po = ["#e74c3c" if v < 20 else "#27ae60" for v in pout_conv.values]
    ax.barh(pout_conv.index, pout_conv.values, color=colors_po, edgecolor="black")
    ax.set_xlabel("Subscription Rate (%)")
    ax.set_title(
        "Previous campaign success is the #1 predictor", fontsize=14, fontweight="bold"
    )
    for i, v in enumerate(pout_conv.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_previous_campaign_effect.png", bbox_inches="tight")
    plt.close()
    step_done("03_previous_campaign_effect.png")

    # 4d. Monthly patterns
    month_order = [
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
    ]
    month_data = df.groupby("month")[TARGET].agg(["mean", "count"]).reindex(month_order)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(
        month_data.index,
        month_data["count"],
        alpha=0.35,
        color="#3498db",
        edgecolor="black",
        label="Call Volume",
    )
    ax2.plot(
        month_data.index,
        month_data["mean"] * 100,
        "o-",
        color="#e74c3c",
        linewidth=2.5,
        markersize=8,
        label="Conversion %",
    )
    ax1.set_ylabel("Calls", color="#3498db")
    ax2.set_ylabel("Conversion %", color="#e74c3c")
    ax1.set_title("Volume vs Conversion by Month", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_monthly_patterns.png", bbox_inches="tight")
    plt.close()
    step_done("04_monthly_patterns.png")

    # 4e. Contact method
    fig, ax = plt.subplots(figsize=(7, 4))
    contact_conv = df.groupby("contact")[TARGET].mean() * 100
    ax.bar(
        contact_conv.index,
        contact_conv.values,
        color=["#e67e22", "#27ae60"],
        edgecolor="black",
        width=0.5,
    )
    ax.set_ylabel("Subscription Rate (%)")
    ax.set_title("Mobile outperforms landline", fontsize=14, fontweight="bold")
    for i, v in enumerate(contact_conv.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_contact_method.png", bbox_inches="tight")
    plt.close()
    step_done("05_contact_method.png")

    # 4f. Correlation heatmap
    num_cols = df.select_dtypes(include="number").columns.tolist()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
    sns.heatmap(
        df[num_cols].corr(),
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_correlation_matrix.png", bbox_inches="tight")
    plt.close()
    step_done("06_correlation_matrix.png")

    # 4g. Age distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df[TARGET] == 0]["age"].hist(
        bins=40, ax=ax, alpha=0.5, color="#e74c3c", label="No sub", edgecolor="black"
    )
    df[df[TARGET] == 1]["age"].hist(
        bins=40,
        ax=ax,
        alpha=0.6,
        color="#27ae60",
        label="Subscribed",
        edgecolor="black",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age distribution — Who subscribes?", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_age_distribution.png", bbox_inches="tight")
    plt.close()
    step_done("07_age_distribution.png")

    # 4h. Unknown values
    unknowns = []
    for col in df_raw.select_dtypes(include="object").columns:
        n = (df_raw[col] == "unknown").sum()
        if n > 0:
            unknowns.append({"Column": col, "Percent": round(n / len(df_raw) * 100, 1)})
    if unknowns:
        udf = pd.DataFrame(unknowns).sort_values("Percent", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(udf["Column"], udf["Percent"], color="#e67e22", edgecolor="black")
        ax.set_xlabel("% Unknown")
        ax.set_title("Missing data as 'unknown'", fontsize=14, fontweight="bold")
        for i, row in udf.iterrows():
            ax.text(
                row["Percent"] + 0.3,
                list(udf["Column"]).index(row["Column"]),
                f'{row["Percent"]}%',
                va="center",
                fontweight="bold",
            )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "08_unknown_values.png", bbox_inches="tight")
        plt.close()
        step_done("08_unknown_values.png")

    # NEW 4h2. Outlier detection boxplots
    outlier_cols = ["age", "campaign", "previous", "cons.conf.idx", "euribor3m"]
    available_cols = [c for c in outlier_cols if c in df_raw.columns]
    if available_cols:
        fig, axes = plt.subplots(
            1, len(available_cols), figsize=(4 * len(available_cols), 5)
        )
        if len(available_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, available_cols):
            bp = ax.boxplot(
                [
                    df_raw.loc[df_raw[TARGET] == 0, col].dropna(),
                    df_raw.loc[df_raw[TARGET] == 1, col].dropna(),
                ],
                labels=["No Sub", "Subscribed"],
                patch_artist=True,
                boxprops=dict(facecolor="#AED6F1", edgecolor="black"),
                medianprops=dict(color="red", linewidth=2),
            )
            bp["boxes"][1].set_facecolor("#ABEBC6")
            n_outliers_low = (df_raw[col] < df_raw[col].quantile(0.01)).sum()
            n_outliers_high = (df_raw[col] > df_raw[col].quantile(0.99)).sum()
            ax.set_title(
                f"{col}\n({n_outliers_low + n_outliers_high} outliers)",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(alpha=0.3)
        plt.suptitle(
            "Outlier Detection — Boxplots by Target Class",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "08c_outlier_boxplots.png", bbox_inches="tight", dpi=150)
        plt.close()
        step_done("08c_outlier_boxplots.png — outlier detection visualization")

    # NEW 4h3. Feature distributions — all numeric features by target class
    dist_cols = [
        "age",
        "campaign",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    available_dist = [c for c in dist_cols if c in df.columns]
    if available_dist:
        ncols_d = 4
        nrows_d = (len(available_dist) + ncols_d - 1) // ncols_d
        fig, axes = plt.subplots(nrows_d, ncols_d, figsize=(5 * ncols_d, 4 * nrows_d))
        axes = np.array(axes).flatten()
        for i, col in enumerate(available_dist):
            ax = axes[i]
            df.loc[df[TARGET] == 0, col].hist(
                bins=30,
                ax=ax,
                alpha=0.5,
                color="#e74c3c",
                label="No",
                edgecolor="black",
                density=True,
            )
            df.loc[df[TARGET] == 1, col].hist(
                bins=30,
                ax=ax,
                alpha=0.6,
                color="#27ae60",
                label="Yes",
                edgecolor="black",
                density=True,
            )
            skew_val = df[col].skew()
            ax.set_title(f"{col} (skew={skew_val:.2f})", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        for j in range(len(available_dist), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(
            "Feature Distributions by Target Class (with Skewness)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / "08d_feature_distributions.png", bbox_inches="tight", dpi=150
        )
        plt.close()
        step_done(
            "08d_feature_distributions.png — all numeric feature distributions + skewness"
        )

    # NEW 4h4. Categorical cardinality + conversion rates
    cat_cols_eda = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "poutcome",
    ]
    available_cat = [c for c in cat_cols_eda if c in df.columns]
    if available_cat:
        ncols_c = 4
        nrows_c = (len(available_cat) + ncols_c - 1) // ncols_c
        fig, axes = plt.subplots(nrows_c, ncols_c, figsize=(5 * ncols_c, 4 * nrows_c))
        axes = np.array(axes).flatten()
        for i, col in enumerate(available_cat):
            ax = axes[i]
            conv_rates = (
                df.groupby(col)[TARGET]
                .agg(["mean", "count"])
                .sort_values("mean", ascending=True)
            )
            colors_bar = [
                "#27ae60" if m > df[TARGET].mean() else "#e74c3c"
                for m in conv_rates["mean"]
            ]
            ax.barh(
                conv_rates.index,
                conv_rates["mean"] * 100,
                color=colors_bar,
                edgecolor="black",
            )
            ax.axvline(df[TARGET].mean() * 100, color="gray", linestyle="--", alpha=0.6)
            ax.set_title(
                f"{col} ({conv_rates.shape[0]} levels)", fontsize=10, fontweight="bold"
            )
            ax.set_xlabel("Conv %", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        for j in range(len(available_cat), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(
            "Categorical Features — Conversion Rates + Cardinality",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / "08e_categorical_analysis.png", bbox_inches="tight", dpi=150
        )
        plt.close()
        step_done(
            "08e_categorical_analysis.png — categorical cardinality + conversion rates"
        )

    # 4i. Non-linear relationship visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(
        axes,
        ["age", "campaign", "euribor3m"],
        [
            "Age (U-shaped)",
            "Campaign (Diminishing Returns)",
            "Euribor3m (Economic Regimes)",
        ],
    ):
        bins = pd.qcut(df[col], q=10, duplicates="drop")
        conv_by_bin = df.groupby(bins)[TARGET].mean() * 100
        ax.plot(
            range(len(conv_by_bin)),
            conv_by_bin.values,
            "o-",
            color="#2980b9",
            linewidth=2,
            markersize=8,
        )
        ax.axhline(
            df[TARGET].mean() * 100,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Overall avg",
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Subscription Rate (%)")
        ax.set_title(f"{title}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(conv_by_bin)))
        ax.set_xticklabels(
            [f"{x.left:.0f}-{x.right:.0f}" for x in conv_by_bin.index],
            rotation=45,
            fontsize=8,
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle(
        "Why Binning Matters — Non-Linear Feature Relationships",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "08b_nonlinear_relationships.png", bbox_inches="tight", dpi=150
    )
    plt.close()
    step_done("08b_nonlinear_relationships.png — justifies binning strategy")

    log.info(f"  🎨 EDA complete — {len(list(FIG_DIR.glob('*.png')))} figures")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 : FEATURE IMPORTANCE + SWEEP (FIXED — no data leakage)
# ═══════════════════════════════════════════════════════════════════════════════
def step_feature_importance(X_train, y_train, use_binning):
    banner("STEP 5  ·  FEATURE IMPORTANCE (leakage-free)", "🔍")

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
            use_binning=use_binning,
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

    fig, ax = plt.subplots(figsize=(10, 9))
    top = imp_df.head(15)
    ax.barh(top["feature"], top["importance"], color="#2980b9", edgecolor="black")
    ax.set_xlabel("Importance (Gini, CV-averaged)")
    ax.invert_yaxis()
    ax.set_title("Top 15 Features (Cross-Validated RF)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_feature_importance_rf.png", bbox_inches="tight")
    plt.close()
    step_done("09_feature_importance_rf.png (CV-averaged, no data leakage)")

    sorted_features = imp_df["feature"].tolist()
    log.info(f"  📐 Total features: {len(sorted_features)}")
    log.info(f"  🏆 Top-3: {', '.join(sorted_features[:3])}")
    return feat_names, sorted_features, imp_df


def step_feature_count_sweep(
    X_train, y_train, feat_names, sorted_features, use_binning
):
    banner("STEP 5b  ·  FEATURE COUNT SWEEP", "📐")

    base_pipe = build_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        use_binning=use_binning,
    )
    preprocess_pipe = SkPipeline(base_pipe.steps[:-1])
    preprocess_pipe.fit(X_train, y_train)
    X_transformed = preprocess_pipe.transform(X_train)
    X_df = pd.DataFrame(X_transformed, columns=feat_names)

    feature_order = [f for f in sorted_features if f in feat_names]
    counts_to_try = sorted(set([2, 3, 5, 7, 10, 15, 20, 25, 30] + [len(feature_order)]))
    counts_to_try = [c for c in counts_to_try if c <= len(feature_order)]

    sweep_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    log.info(f"  🔄 Sweeping {len(counts_to_try)} feature counts: {counts_to_try}")

    for n_feat in counts_to_try:
        top_n = feature_order[:n_feat]
        X_sub = X_df[top_n]

        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        scores_rf = cross_val_predict(
            rf, X_sub, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        auc_rf = roc_auc_score(y_train, scores_rf)

        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
        )
        scores_xgb = cross_val_predict(
            xgb, X_sub, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        auc_xgb = roc_auc_score(y_train, scores_xgb)

        lr = LogisticRegression(
            C=0.1,
            solver="saga",
            penalty="l1",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        scores_lr = cross_val_predict(
            lr, X_sub, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        auc_lr = roc_auc_score(y_train, scores_lr)

        sweep_results.append(
            {
                "n_features": n_feat,
                "rf_auc": round(auc_rf, 4),
                "xgb_auc": round(auc_xgb, 4),
                "lr_auc": round(auc_lr, 4),
            }
        )
        bar = "█" * int(auc_xgb * 40)
        log.info(
            f"     {n_feat:3d} features │ RF: {auc_rf:.4f}  XGB: {auc_xgb:.4f}  LR: {auc_lr:.4f}  │ {bar}"
        )

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(MET_DIR / "feature_count_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        sweep_df["n_features"],
        sweep_df["rf_auc"],
        "o-",
        linewidth=2.5,
        markersize=8,
        label="Random Forest",
        color="#27ae60",
    )
    ax.plot(
        sweep_df["n_features"],
        sweep_df["xgb_auc"],
        "s-",
        linewidth=2.5,
        markersize=8,
        label="XGBoost",
        color="#2980b9",
    )
    ax.plot(
        sweep_df["n_features"],
        sweep_df["lr_auc"],
        "^-",
        linewidth=2.5,
        markersize=8,
        label="Logistic Reg",
        color="#e74c3c",
    )

    for col, color, name in [
        ("rf_auc", "#27ae60", "RF"),
        ("xgb_auc", "#2980b9", "XGB"),
        ("lr_auc", "#e74c3c", "LR"),
    ]:
        best_idx = sweep_df[col].idxmax()
        best_n = sweep_df.loc[best_idx, "n_features"]
        best_auc = sweep_df.loc[best_idx, col]
        ax.annotate(
            f"{name} best: {best_n}f → {best_auc:.4f}",
            xy=(best_n, best_auc),
            fontsize=9,
            fontweight="bold",
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color),
            color=color,
        )

    ax.set_xlabel("Number of Features (ranked by importance)", fontsize=12)
    ax.set_ylabel("Cross-Validated AUC-ROC", fontsize=12)
    ax.set_title(
        "How many features do we actually need?", fontsize=15, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(sweep_df["n_features"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09b_feature_count_sweep.png", bbox_inches="tight", dpi=150)
    plt.close()
    step_done("09b_feature_count_sweep.png")

    return sweep_df


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM TUNING
# ═══════════════════════════════════════════════════════════════════════════════
def _tune_lightgbm(X, y, n_trials, cv, use_binning):
    pos = y.sum()
    neg = len(y) - pos
    scale = neg / pos

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "scale_pos_weight": scale,
        }
        pipe = build_pipeline(
            LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1),
            use_binning=use_binning,
        )
        scores = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(y, scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {
        "best_params": {**study.best_params, "scale_pos_weight": scale},
        "best_auc": study.best_value,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5c : BASELINE (Before Tuning) — Default Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════
def step_baseline_models(X_tr, y_tr, X_te, y_te, use_binning):
    """
    Train models with DEFAULT hyperparameters (no tuning) to establish baseline.
    This enables a before/after comparison showing the value of Optuna tuning.
    """
    banner("STEP 5c  ·  BASELINE MODELS (Before Tuning)", "📏")

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
            scale_pos_weight=7.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="aucpr",
        ),
    }
    if HAS_LGBM:
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        default_configs["lightgbm"] = LGBMClassifier(
            n_estimators=100,
            max_depth=-1,
            learning_rate=0.1,
            scale_pos_weight=neg / pos,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

    for name, clf in default_configs.items():
        pipe = build_pipeline(clf, use_binning=use_binning)
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
        log.info(
            f"  📏 {name:25s} │ AUC={auc:.4f}  LogLoss={logloss:.4f}  "
            f"Brier={brier:.4f}  Profit=${profit:,.0f}"
        )

    return baseline_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6–7 : TRAIN + EVALUATE + COMPOSITE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def step_train_and_evaluate(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials, use_binning):
    banner("STEP 6–7  ·  TRAINING + COMPOSITE MODEL SELECTION", "🏋️")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    all_metrics = []
    pipelines = {}

    # ── Core 3 models ────────────────────────────────────────────────────────
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        log.info("")
        log.info(f"  \033[1m🔧 Training: {name} ({n_trials} Optuna trials)\033[0m")

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

        joblib.dump(pipeline, MOD_DIR / f"{name}.joblib")
        metric_log(
            name,
            f"AUC={metrics['roc_auc']:.4f}  Recall={metrics['recall']:.3f}  "
            f"Thresh={opt_t:.3f}  Profit=${cost['net_profit']:,.0f}  "
            f"Brier={metrics['brier_score']:.4f}",
        )

    # ── LightGBM ─────────────────────────────────────────────────────────────
    if HAS_LGBM:
        log.info("")
        log.info(f"  \033[1m🔧 Training: lightgbm ({n_trials} Optuna trials)\033[0m")
        lgb_result = _tune_lightgbm(X_tr, y_tr, n_trials, cv, use_binning)
        lgb_pipe = build_pipeline(
            LGBMClassifier(
                **lgb_result["best_params"], random_state=42, verbosity=-1, n_jobs=-1
            ),
            use_binning=use_binning,
        )
        lgb_pipe.fit(X_tr, y_tr)
        pipelines["lightgbm"] = lgb_pipe

        y_va_prob = lgb_pipe.predict_proba(X_va)[:, 1]
        opt_t, _ = find_optimal_threshold(y_va, y_va_prob)
        y_te_prob = lgb_pipe.predict_proba(X_te)[:, 1]
        y_te_pred = (y_te_prob >= opt_t).astype(int)
        metrics = compute_metrics(y_te, y_te_pred, y_te_prob)
        cost = business_cost_analysis(y_te, y_te_pred)

        results["lightgbm"] = {
            "params": lgb_result["best_params"],
            "cv_auc": lgb_result["best_auc"],
            "threshold": opt_t,
            "test_metrics": metrics,
            "cost": cost,
            "y_te_prob": y_te_prob,
            "y_te_pred": y_te_pred,
        }
        row = {
            "model": "lightgbm",
            "cv_auc": round(lgb_result["best_auc"], 4),
            "threshold": round(opt_t, 3),
        }
        row.update({f"test_{k}": round(v, 4) for k, v in metrics.items()})
        row["net_profit"] = round(cost["net_profit"], 0)
        row["brier_score"] = round(metrics["brier_score"], 4)
        all_metrics.append(row)
        joblib.dump(lgb_pipe, MOD_DIR / "lightgbm.joblib")
        metric_log(
            "lightgbm",
            f"AUC={metrics['roc_auc']:.4f}  Recall={metrics['recall']:.3f}  "
            f"Thresh={opt_t:.3f}  Profit=${cost['net_profit']:,.0f}  "
            f"Brier={metrics['brier_score']:.4f}",
        )
    else:
        log.info("  ⚠️  LightGBM not installed — skipping")

    # ── Diverse Voting Ensemble ──────────────────────────────────────────────
    log.info("")
    log.info(f"  \033[1m🔧 Building: voting_ensemble (diverse soft vote)\033[0m")

    tree_candidates = {
        n: r["test_metrics"]["roc_auc"]
        for n, r in results.items()
        if n != "logistic_regression"
    }
    best_tree = max(tree_candidates, key=tree_candidates.get)

    ensemble_members = ["logistic_regression", best_tree]
    if HAS_LGBM and "lightgbm" in pipelines and "lightgbm" != best_tree:
        ensemble_members.append("lightgbm")
    elif "xgboost" in pipelines and "xgboost" != best_tree:
        ensemble_members.append("xgboost")
    elif "random_forest" in pipelines and "random_forest" != best_tree:
        ensemble_members.append("random_forest")

    log.info(f"     Ensemble members: {ensemble_members}")

    ref_pipe = pipelines[ensemble_members[0]]
    preprocess_steps = ref_pipe.steps[:-1]
    preprocess_pipe = SkPipeline(preprocess_steps)
    preprocess_pipe.fit(X_tr, y_tr)
    X_tr_t = preprocess_pipe.transform(X_tr)
    X_va_t = preprocess_pipe.transform(X_va)
    X_te_t = preprocess_pipe.transform(X_te)

    estimators = [
        (nm, pipelines[nm].named_steps["classifier"]) for nm in ensemble_members
    ]
    voter = VotingClassifier(estimators=estimators, voting="soft")
    voter.fit(X_tr_t, y_tr)

    y_va_prob = voter.predict_proba(X_va_t)[:, 1]
    opt_t, _ = find_optimal_threshold(y_va, y_va_prob)
    y_te_prob = voter.predict_proba(X_te_t)[:, 1]
    y_te_pred = (y_te_prob >= opt_t).astype(int)
    metrics = compute_metrics(y_te, y_te_pred, y_te_prob)
    cost = business_cost_analysis(y_te, y_te_pred)

    results["voting_ensemble"] = {
        "params": {"members": ensemble_members},
        "cv_auc": np.nan,
        "threshold": opt_t,
        "test_metrics": metrics,
        "cost": cost,
        "y_te_prob": y_te_prob,
        "y_te_pred": y_te_pred,
    }
    row = {"model": "voting_ensemble", "cv_auc": np.nan, "threshold": round(opt_t, 3)}
    row.update({f"test_{k}": round(v, 4) for k, v in metrics.items()})
    row["net_profit"] = round(cost["net_profit"], 0)
    row["brier_score"] = round(metrics["brier_score"], 4)
    all_metrics.append(row)
    joblib.dump(
        {"preprocess": preprocess_pipe, "voter": voter},
        MOD_DIR / "voting_ensemble.joblib",
    )
    metric_log(
        "voting_ensemble",
        f"AUC={metrics['roc_auc']:.4f}  Recall={metrics['recall']:.3f}  "
        f"Thresh={opt_t:.3f}  Profit=${cost['net_profit']:,.0f}  "
        f"Brier={metrics['brier_score']:.4f}",
    )

    # ── COMPOSITE MODEL SELECTION ────────────────────────────────────────────
    comp_df = pd.DataFrame(all_metrics)

    banner("COMPOSITE MODEL SELECTION", "🧠")
    best_name, scored_df = select_best_model(comp_df)

    # Sensitivity analysis
    sensitivity = selection_sensitivity_analysis(comp_df)
    log.info("")
    log.info("  📊 Sensitivity Analysis (winner under each criterion):")
    for criterion, winner in sensitivity.items():
        marker = "✅" if winner == best_name else "⚠️ "
        log.info(f"     {marker} {criterion:>18s} → {winner}")

    unique_winners = set(sensitivity.values())
    if len(unique_winners) == 1:
        log.info(
            f"\n  \033[1m\033[92m✅ ROBUST: {best_name} wins under ALL criteria!\033[0m"
        )
    elif len(unique_winners) <= 2:
        log.info(
            f"\n  \033[1m\033[93m🟡 MOSTLY ROBUST: {best_name} wins composite; {len(unique_winners)} unique winners\033[0m"
        )
    else:
        log.info(
            f"\n  \033[1m\033[91m⚠️  FRAGILE: {len(unique_winners)} different winners across criteria\033[0m"
        )

    # Save everything
    scored_df.to_json(MET_DIR / "comparison.json", orient="records", indent=2)
    scored_df.to_csv(MET_DIR / "comparison.csv", index=False)

    best_t = results[best_name]["threshold"]
    with open(MOD_DIR / "threshold.json", "w") as f:
        json.dump(
            {
                "model": best_name,
                "threshold": round(best_t, 4),
                "selection_criterion": "composite_score",
                "composite_weights": SELECTION_WEIGHTS,
                "sensitivity": sensitivity,
                "banking_economics": BANKING_ECONOMICS,
            },
            f,
            indent=2,
        )
    log.info(f"\n  📌 Selected: \033[1m{best_name}\033[0m (composite-optimized)")

    # ── Recall analysis table ────────────────────────────────────────────────
    banner("RECALL ANALYSIS — Stakeholder View", "📋")
    best_probs = results[best_name]["y_te_prob"]
    recall_df = recall_analysis(y_te, best_probs)
    recall_df.to_csv(MET_DIR / "recall_analysis.csv", index=False)
    log.info("  📋 Recall trade-off table saved → recall_analysis.csv")

    for _, r in recall_df.iterrows():
        if r["threshold"] in [0.10, 0.20, 0.30, 0.40, 0.50]:
            profit_str = f"${r['net_profit']:,.0f}"
            log.info(
                f"     t={r['threshold']:.2f}  recall={r['recall']:.3f}  "
                f"calls={r['calls_made']:,}  caught={r['subscribers_caught']}  "
                f"missed={r['subscribers_missed']}  profit={profit_str}"
            )

    return results, pipelines, scored_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7b : BEFORE vs AFTER TUNING COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def step_tuning_comparison(baseline_results, tuned_results, y_te):
    """Compare baseline (default params) vs tuned (Optuna) performance."""
    banner("STEP 7b  ·  BEFORE vs AFTER HYPERPARAMETER TUNING", "📈")

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

        profit_gain_str = f"${row['profit_gain']:+,.0f}"
        log.info(
            f"  {name:25s} │ AUC: {base['auc']:.4f} → {tuned_metrics['roc_auc']:.4f} "
            f"({row['auc_gain']:+.4f})  │ LogLoss: {base['log_loss']:.4f} → {tuned_logloss:.4f} "
            f"({row['logloss_reduction']:+.4f})  │ Profit: {profit_gain_str}"
        )

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(MET_DIR / "tuning_comparison.csv", index=False)

    # ── Figure: Before vs After Tuning ───────────────────────────────────────
    if comparison_rows:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        models = [r["model"] for r in comparison_rows]
        x = np.arange(len(models))
        w = 0.35

        # AUC comparison
        axes[0].bar(
            x - w / 2,
            [r["baseline_auc"] for r in comparison_rows],
            w,
            label="Before Tuning",
            color="#e74c3c",
            edgecolor="black",
            alpha=0.8,
        )
        axes[0].bar(
            x + w / 2,
            [r["tuned_auc"] for r in comparison_rows],
            w,
            label="After Tuning",
            color="#27ae60",
            edgecolor="black",
            alpha=0.8,
        )
        for i, r in enumerate(comparison_rows):
            axes[0].annotate(
                f"{r['auc_gain']:+.4f}",
                xy=(i + w / 2, r["tuned_auc"]),
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#27ae60",
            )
        axes[0].set_ylabel("AUC-ROC")
        axes[0].set_title("AUC: Before vs After Tuning", fontsize=12, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3, axis="y")

        # Log-Loss comparison (lower is better)
        axes[1].bar(
            x - w / 2,
            [r["baseline_logloss"] for r in comparison_rows],
            w,
            label="Before Tuning",
            color="#e74c3c",
            edgecolor="black",
            alpha=0.8,
        )
        axes[1].bar(
            x + w / 2,
            [r["tuned_logloss"] for r in comparison_rows],
            w,
            label="After Tuning",
            color="#27ae60",
            edgecolor="black",
            alpha=0.8,
        )
        for i, r in enumerate(comparison_rows):
            axes[1].annotate(
                f"{r['logloss_reduction']:+.4f}",
                xy=(i + w / 2, r["tuned_logloss"]),
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#27ae60",
            )
        axes[1].set_ylabel("Log-Loss (↓ better)")
        axes[1].set_title(
            "Log-Loss: Before vs After Tuning", fontsize=12, fontweight="bold"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3, axis="y")

        # Profit comparison
        axes[2].bar(
            x - w / 2,
            [r["baseline_profit"] for r in comparison_rows],
            w,
            label="Before Tuning",
            color="#e74c3c",
            edgecolor="black",
            alpha=0.8,
        )
        axes[2].bar(
            x + w / 2,
            [r["tuned_profit"] for r in comparison_rows],
            w,
            label="After Tuning",
            color="#27ae60",
            edgecolor="black",
            alpha=0.8,
        )
        for i, r in enumerate(comparison_rows):
            gain_str = f"${r['profit_gain']:+,.0f}"
            axes[2].annotate(
                gain_str,
                xy=(i + w / 2, r["tuned_profit"]),
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#27ae60",
            )
        axes[2].set_ylabel("Net Profit ($)")
        axes[2].set_title(
            "Profit: Before vs After Tuning", fontsize=12, fontweight="bold"
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3, axis="y")

        plt.suptitle(
            "Hyperparameter Tuning Impact — Default vs Optuna-Optimized",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "13e_tuning_comparison.png", bbox_inches="tight", dpi=150)
        plt.close()
        step_done("13e_tuning_comparison.png — before/after hyperparameter tuning")

    return comp_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 : EVALUATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
def step_eval_figures(results, y_te):
    banner("STEP 8  ·  EVALUATION FIGURES", "📈")

    # ── Confusion matrices ───────────────────────────────────────────────────
    n_models = len(results)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if n_models > 1 else [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_te, r["y_te_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{name}\n(t={r['threshold']:.3f})", fontsize=10, fontweight="bold"
        )
    for ax in axes[len(results) :]:
        ax.set_visible(False)
    plt.suptitle("Confusion Matrices — Test Set", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_confusion_matrices.png", bbox_inches="tight")
    plt.close()
    step_done("10_confusion_matrices.png")

    # ── ROC + PR ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.tab10
    for i, (name, r) in enumerate(results.items()):
        color = cmap(i)
        fpr, tpr, _ = roc_curve(y_te, r["y_te_prob"])
        auc = roc_auc_score(y_te, r["y_te_prob"])
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} ({auc:.3f})", color=color)
        prec, rec, _ = precision_recall_curve(y_te, r["y_te_prob"])
        pr_auc = average_precision_score(y_te, r["y_te_prob"])
        axes[1].plot(
            rec, prec, linewidth=2, label=f"{name} ({pr_auc:.3f})", color=color
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(
        y_te.mean(),
        color="gray",
        linestyle="--",
        alpha=0.4,
        label=f"Baseline ({y_te.mean():.2f})",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    plt.suptitle("Model Discrimination — Test Set", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_roc_pr_curves.png", bbox_inches="tight")
    plt.close()
    step_done("11_roc_pr_curves.png")

    # ── Business impact ──────────────────────────────────────────────────────
    strategies = {
        "Call Nobody": 0,
        "Call Everybody": int(y_te.sum()) * 195 - int((~y_te.astype(bool)).sum()) * 5,
    }
    for name, r in results.items():
        strategies[f"ML: {name}"] = r["cost"]["net_profit"]
    fig, ax = plt.subplots(figsize=(14, 6))
    names = list(strategies.keys())
    profits = list(strategies.values())
    colors = ["#95a5a6", "#e74c3c"] + [plt.cm.tab10(i) for i in range(len(names) - 2)]
    bars = ax.bar(names, profits, color=colors, edgecolor="black")
    ax.set_ylabel("Net Profit ($)")
    ax.set_title("How much does ML save?", fontsize=15, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, profits):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"${val:,.0f}",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_business_impact.png", bbox_inches="tight")
    plt.close()
    step_done("12_business_impact.png")

    # ── Threshold sensitivity ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, r) in enumerate(results.items()):
        profits_curve = []
        threshs = np.linspace(0.05, 0.95, 100)
        for t in threshs:
            yp = (r["y_te_prob"] >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
            profits_curve.append(tp * 195 - fp * 5 - fn * 200)
        ax.plot(threshs, profits_curve, linewidth=2, label=name, color=plt.cm.tab10(i))
        best_idx = np.argmax(profits_curve)
        ax.scatter(
            threshs[best_idx],
            profits_curve[best_idx],
            s=80,
            zorder=5,
            color=plt.cm.tab10(i),
        )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default 0.5")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Net Profit ($)")
    ax.set_title("Why custom thresholds matter", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_threshold_sensitivity.png", bbox_inches="tight")
    plt.close()
    step_done("13_threshold_sensitivity.png")

    # ── Calibration curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    for i, (name, r) in enumerate(results.items()):
        prob_true, prob_pred = calibration_curve(
            y_te, r["y_te_prob"], n_bins=10, strategy="uniform"
        )
        brier = brier_score_loss(y_te, r["y_te_prob"])
        ax.plot(
            prob_pred,
            prob_true,
            "o-",
            linewidth=2,
            markersize=6,
            label=f"{name} (Brier={brier:.4f})",
            color=plt.cm.tab10(i),
        )
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(
        "Calibration Curves — Are probabilities trustworthy?",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13b_calibration_curves.png", bbox_inches="tight", dpi=150)
    plt.close()
    step_done("13b_calibration_curves.png")

    # ── Model comparison bars ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    model_names = list(results.keys())
    aucs = [results[n]["test_metrics"]["roc_auc"] for n in model_names]
    pr_aucs = [results[n]["test_metrics"]["pr_auc"] for n in model_names]
    recalls = [results[n]["test_metrics"]["recall"] for n in model_names]
    x = np.arange(len(model_names))
    w = 0.25
    ax.bar(x - w, aucs, w, label="ROC-AUC", color="#2980b9", edgecolor="black")
    ax.bar(x, pr_aucs, w, label="PR-AUC", color="#27ae60", edgecolor="black")
    ax.bar(x + w, recalls, w, label="Recall", color="#e67e22", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("All Models — AUC vs PR-AUC vs Recall", fontsize=15, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13c_model_comparison_bars.png", bbox_inches="tight")
    plt.close()
    step_done("13c_model_comparison_bars.png")

    # Composite score breakdown figure
    fig, ax = plt.subplots(figsize=(12, 6))
    comp_data = []
    for name, r in results.items():
        m = r["test_metrics"]
        comp_data.append(
            {
                "model": name,
                "Profit (40%)": r["cost"]["net_profit"],
                "Recall (25%)": m["recall"],
                "AUC (20%)": m["roc_auc"],
                "1-Brier (15%)": 1 - m["brier_score"],
            }
        )
    cdf = pd.DataFrame(comp_data)
    for col in ["Profit (40%)", "Recall (25%)", "AUC (20%)", "1-Brier (15%)"]:
        vmin, vmax = cdf[col].min(), cdf[col].max()
        cdf[col] = (cdf[col] - vmin) / (vmax - vmin) if vmax > vmin else 1.0
    cdf.set_index("model")[
        ["Profit (40%)", "Recall (25%)", "AUC (20%)", "1-Brier (15%)"]
    ].plot(
        kind="bar",
        ax=ax,
        width=0.8,
        edgecolor="black",
        color=["#27ae60", "#e67e22", "#2980b9", "#9b59b6"],
    )
    ax.set_ylabel("Normalized Score (0-1)")
    ax.set_title(
        "Composite Selection — Why the best model wins", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13d_composite_selection.png", bbox_inches="tight", dpi=150)
    plt.close()
    step_done("13d_composite_selection.png — shows WHY this model was selected")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 : SHAP
# ═══════════════════════════════════════════════════════════════════════════════
def step_shap(best_pipeline, X_test, y_test, feat_names, best_name):
    banner("STEP 9  ·  SHAP EXPLAINABILITY", "🔬")
    import shap

    classifier = best_pipeline.named_steps["classifier"]
    preprocess_pipe = SkPipeline(best_pipeline.steps[:-1])
    X_transformed = preprocess_pipe.transform(X_test)
    X_df = pd.DataFrame(X_transformed, columns=feat_names)
    sample = X_df.iloc[:800]

    tree_models = (RandomForestClassifier, XGBClassifier)
    if HAS_LGBM:
        tree_models = tree_models + (LGBMClassifier,)

    linear_models = (LogisticRegression,)

    if isinstance(classifier, tree_models):
        log.info(f"  Using TreeExplainer for {best_name}")
        explainer = shap.TreeExplainer(classifier)
        sv = explainer(sample)
    elif isinstance(classifier, linear_models):
        log.info(f"  Using LinearExplainer for {best_name}")
        explainer = shap.LinearExplainer(classifier, sample)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        base_val = classifier.predict_proba(sample.values)[:, 1].mean()
        sv = shap.Explanation(
            values=shap_vals,
            base_values=np.full(shap_vals.shape[0], base_val),
            data=sample.values,
            feature_names=feat_names,
        )
    else:
        inner = getattr(
            classifier, "estimator", getattr(classifier, "base_estimator", None)
        )
        if inner is not None and isinstance(inner, tree_models):
            log.info(f"  Using TreeExplainer on inner estimator of {best_name}")
            explainer = shap.TreeExplainer(inner)
            sv = explainer(sample)
        else:
            log.info("  Using KernelExplainer (slower — limited to 100 samples)")
            bg = shap.sample(X_df, 100)
            explainer = shap.KernelExplainer(classifier.predict_proba, bg)
            shap_vals = explainer.shap_values(sample.iloc[:100])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            sample = sample.iloc[:100]
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1] if len(base_val) > 1 else base_val[0]
            sv = shap.Explanation(
                values=shap_vals,
                base_values=np.full(shap_vals.shape[0], base_val),
                data=sample.values,
                feature_names=feat_names,
            )

    if len(sv.shape) == 3:
        sv = sv[:, :, 1]

    plt.figure(figsize=(12, 9))
    shap.summary_plot(sv, sample, feature_names=feat_names, show=False, max_display=15)
    plt.title(f"SHAP Summary — {best_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    step_done("14_shap_summary.png")

    plt.figure(figsize=(10, 8))
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Mean |SHAP Value|", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "15_shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close()
    step_done("15_shap_bar.png")

    probs = classifier.predict_proba(X_transformed[:800])[:, 1]
    edge_idx = int(np.argmin(np.abs(probs - 0.30)))
    plt.figure(figsize=(12, 5))
    shap.plots.waterfall(sv[edge_idx], max_display=12, show=False)
    plt.title(f"Why P={probs[edge_idx]:.2f}?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "16_shap_waterfall.png", bbox_inches="tight", dpi=150)
    plt.close()
    step_done("16_shap_waterfall.png")

    mean_abs = np.abs(sv.values).mean(axis=0)
    top_feats = sorted(zip(feat_names, mean_abs), key=lambda x: x[1], reverse=True)[:12]
    rename = {
        "num__euribor3m": "3-Month Interest Rate",
        "num__nr.employed": "Employment Level",
        "num__emp.var.rate": "Employment Change Rate",
        "num__age": "Client Age",
        "num__cons.conf.idx": "Consumer Confidence",
        "num__campaign": "# Calls This Campaign",
        "num__was_previously_contacted": "Was Contacted Before",
        "nom__poutcome_success": "Previous Campaign: Success",
        "nom__contact_telephone": "Contact: Telephone",
        "num__cons.price.idx": "Consumer Price Index",
        "num__pdays_log": "Recency of Contact",
        "num__previous": "# Prior Contacts",
        "nom__age_bin_young": "Age: Young (<30)",
        "nom__age_bin_senior": "Age: Senior (60+)",
        "nom__campaign_bin_high": "Campaign: High (6+ calls)",
        "nom__euribor3m_bin_low_rate": "Euribor: Low Rate Env",
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [rename.get(f, f) for f, _ in top_feats]
    vals = [v for _, v in top_feats]
    cmap_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))
    ax.barh(labels[::-1], vals[::-1], color=cmap_colors[::-1], edgecolor="black")
    ax.set_xlabel("Impact on Prediction")
    ax.set_title(
        "What drives term deposit subscriptions?", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "17_business_feature_importance.png", bbox_inches="tight", dpi=150
    )
    plt.close()
    step_done("17_business_feature_importance.png")

    shap_export = [
        {"feature": rename.get(f, f), "technical": f, "importance": round(float(v), 4)}
        for f, v in top_feats
    ]
    with open(MET_DIR / "shap_importance.json", "w") as fp:
        json.dump(shap_export, fp, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Bank Marketing ML Pipeline v4.1")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-feature-sweep", action="store_true")
    parser.add_argument(
        "--no-binning",
        action="store_true",
        help="Disable non-linear binning (ablation study)",
    )
    args = parser.parse_args()

    use_binning = not args.no_binning
    start = time.time()
    n_models = 3 + (1 if HAS_LGBM else 0) + 1

    log.info("")
    log.info(
        "\033[1m\033[95m╔═══════════════════════════════════════════════════════════════╗\033[0m"
    )
    log.info(
        "\033[1m\033[95m║   🚀  BANK MARKETING ML PIPELINE v4.1 (Production)           ║\033[0m"
    )
    log.info(
        "\033[1m\033[95m║   Models: LR · RF · XGB"
        + (" · LGBM" if HAS_LGBM else "")
        + " · Diverse Ensemble"
        + " " * (18 - (7 if HAS_LGBM else 0))
        + "║\033[0m"
    )
    log.info(
        "\033[1m\033[95m║   NEW: Outliers · Duplicates · Before/After Tuning Compare   ║\033[0m"
    )
    log.info(
        "\033[1m\033[95m╚═══════════════════════════════════════════════════════════════╝\033[0m"
    )
    log.info(
        f"  ⚙️  Trials: {args.n_trials}  |  Models: {n_models}  |  "
        f"Binning: {'ON' if use_binning else 'OFF'}  |  "
        f"SHAP: {'OFF' if args.skip_shap else 'ON'}  |  "
        f"Sweep: {'OFF' if args.skip_feature_sweep else 'ON'}"
    )

    # 1–3: Data (now includes duplicate removal + outlier clipping)
    df_raw, df, df_train, df_val, df_test = step_data()
    X_tr, y_tr = df_train.drop(columns=[TARGET]), df_train[TARGET]
    X_va, y_va = df_val.drop(columns=[TARGET]), df_val[TARGET]
    X_te, y_te = df_test.drop(columns=[TARGET]), df_test[TARGET]

    # 4: EDA (now includes outlier boxplots)
    step_eda(df_raw, df)

    # 5: Feature importance
    feat_names, sorted_features, imp_df = step_feature_importance(
        X_tr, y_tr, use_binning
    )

    # 5b: Feature count sweep
    if not args.skip_feature_sweep:
        step_feature_count_sweep(X_tr, y_tr, feat_names, sorted_features, use_binning)
    else:
        log.info("  ⏭️  Feature sweep skipped")

    # 5c: Baseline models (BEFORE tuning) — for comparison
    baseline_results = step_baseline_models(X_tr, y_tr, X_te, y_te, use_binning)

    # 6–7: Train with Optuna + composite selection
    results, pipelines, comp_df = step_train_and_evaluate(
        X_tr, y_tr, X_va, y_va, X_te, y_te, args.n_trials, use_binning
    )

    # 7b: Before vs After tuning comparison + figure
    step_tuning_comparison(baseline_results, results, y_te)

    # 8: Evaluation figures
    step_eval_figures(results, y_te)

    # 9: SHAP
    if not args.skip_shap:
        best_name = comp_df.iloc[0]["model"]
        if best_name == "voting_ensemble":
            individual = comp_df[comp_df["model"] != "voting_ensemble"]
            best_name = individual.loc[individual["test_roc_auc"].idxmax(), "model"]
            log.info(f"  ℹ️  Ensemble is composite-winner — SHAP on {best_name}")
        step_shap(pipelines[best_name], X_te, y_te, feat_names, best_name)
    else:
        log.info("  ⏭️  SHAP skipped")

    # Summary
    elapsed = time.time() - start
    n_figs = len(list(FIG_DIR.glob("*.png")))

    log.info("")
    log.info(
        "\033[1m\033[92m╔═══════════════════════════════════════════════════════════════╗\033[0m"
    )
    log.info(
        f"\033[1m\033[92m║   ✅  PIPELINE v4.1 COMPLETE in {elapsed / 60:.1f} minutes{' ' * max(0, 29 - len(f'{elapsed / 60:.1f}'))}║\033[0m"
    )
    log.info(f"\033[1m\033[92m║   📁  Models: {str(MOD_DIR)[:47]:<47s} ║\033[0m")
    log.info(
        f"\033[1m\033[92m║   📊  {n_figs} figures → {str(FIG_DIR)[:43]:<43s} ║\033[0m"
    )
    log.info(f"\033[1m\033[92m║   📋  Metrics → {str(MET_DIR)[:45]:<45s} ║\033[0m")
    log.info(
        "\033[1m\033[92m╚═══════════════════════════════════════════════════════════════╝\033[0m"
    )

    print("\n" + "=" * 65)
    print("📊 RESULTS SUMMARY")
    print("=" * 65)
    display_cols = [c for c in comp_df.columns if not c.startswith("norm_")]
    print(comp_df[display_cols].to_string(index=False))
    best = comp_df.iloc[0]
    print(f"\n🏆 Winner: {best['model']} (composite={best['composite_score']:.4f})")
    print(
        f"   Profit: ${best['net_profit']:,.0f} | Recall: {best['test_recall']:.3f} | "
        f"AUC: {best['test_roc_auc']:.4f} | Brier: {best['brier_score']:.4f}"
    )
    print(
        f"\n💡 Selection method: Weighted composite (Profit 40% + Recall 25% + AUC 20% + Calibration 15%)"
    )
    print(f"📁 Models: {MOD_DIR}")
    print(f"📊 Figures: {FIG_DIR} ({n_figs} files)")
    print(f"⏱️  Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
