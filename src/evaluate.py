"""
Evaluation — metrics, cost-optimal threshold, business impact, composite model selection.

═══════════════════════════════════════════════════════════════════════════════
BANKING ECONOMICS COST DERIVATION
═══════════════════════════════════════════════════════════════════════════════
When a customer places a term deposit, the bank earns via Net Interest Margin:

  - Avg term deposit:       $10,000
  - Bank pays depositor:     2.0% interest
  - Bank lends funds at:     8.4% interest
  - Net Interest Margin:     6.4% (8.4% − 2.0%)
  - Annual NIM revenue:      $640 ($10,000 × 6.4%)
  - Over 2-year avg term:    $1,280 lifetime revenue

  Conservative FN cost:      $200  (accounts for <100% conversion, churn, partial deposits)
  FP cost (wasted call):     $5    (agent time + telephony)
  TP net value:              $195  ($200 LTV − $5 call cost)
  TN:                        $0    (correctly skipped)

  Cost ratio:  FN/FP = 40:1 → Missing a subscriber is 40× costlier than a wasted call
  Implication: Model should lean toward higher recall, accepting some false positives

═══════════════════════════════════════════════════════════════════════════════
MODEL SELECTION STRATEGY — Composite Weighted Scoring
═══════════════════════════════════════════════════════════════════════════════
  40% Net Profit       — the business outcome (most important)
  25% Recall           — don't miss subscribers (FN is 40× costlier)
  20% AUC-ROC          — overall discrimination ability
  15% 1−Brier          — probability calibration quality
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BANKING ECONOMICS — Derived, Not Assumed
# ═══════════════════════════════════════════════════════════════════════════════
BANKING_ECONOMICS = {
    "avg_deposit_amount": 10_000,
    "deposit_interest_rate": 0.020,
    "loan_interest_rate": 0.084,
    "net_interest_margin": 0.064,
    "avg_deposit_term_years": 2,
    "annual_nim_revenue": 640,
    "lifetime_nim_revenue": 1_280,
    "conservative_fn_cost": 200,
    "call_cost_per_contact": 5,
    "fn_to_fp_ratio": 40,
}

COST_FP = BANKING_ECONOMICS["call_cost_per_contact"]  # $5
COST_FN = BANKING_ECONOMICS["conservative_fn_cost"]  # $200
VALUE_TP = BANKING_ECONOMICS["conservative_fn_cost"]  # $200 LTV


def get_cost_derivation_text() -> str:
    """Return human-readable cost derivation for stakeholder presentations."""
    e = BANKING_ECONOMICS
    return (
        f"**How We Derived These Costs:**\n\n"
        f"When a customer deposits ${e['avg_deposit_amount']:,} in a term deposit "
        f"at {e['deposit_interest_rate']:.1%} interest, the bank lends those funds "
        f"at ~{e['loan_interest_rate']:.1%}, earning a "
        f"**{e['net_interest_margin']:.1%} net interest margin**.\n\n"
        f"- Annual NIM revenue: **${e['annual_nim_revenue']:,}** per subscriber\n"
        f"- Over {e['avg_deposit_term_years']}-year avg term: "
        f"**${e['lifetime_nim_revenue']:,}** lifetime value\n"
        f"- Conservative FN cost: **${e['conservative_fn_cost']}** "
        f"(adjusted for conversion uncertainty & churn)\n"
        f"- Call cost per contact: **${e['call_cost_per_contact']}**\n\n"
        f"**Cost ratio: FN/FP = {e['fn_to_fp_ratio']}:1** — "
        f"missing a subscriber is {e['fn_to_fp_ratio']}× costlier than a wasted call."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Compute full classification metric suite including calibration."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def find_optimal_threshold(
    y_true,
    y_prob,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
    value_tp: float = VALUE_TP,
    n_steps: int = 200,
) -> tuple[float, float]:
    """Search for threshold that maximizes net business profit."""
    best_t, best_profit = 0.5, -np.inf

    for t in np.linspace(0.05, 0.95, n_steps):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        profit = tp * (value_tp - cost_fp) - fp * cost_fp - fn * cost_fn
        if profit > best_profit:
            best_profit, best_t = profit, float(t)

    logger.info("Optimal threshold: %.3f (profit: $%.0f)", best_t, best_profit)
    return best_t, best_profit


# ═══════════════════════════════════════════════════════════════════════════════
# BUSINESS COST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def business_cost_analysis(y_true, y_pred) -> dict[str, Any]:
    """Translate confusion matrix into dollar impact with banking economics context."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    revenue = tp * (VALUE_TP - COST_FP)
    wasted = fp * COST_FP
    missed = fn * COST_FN
    net = revenue - wasted - missed
    total = tp + fp + fn + tn

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "revenue": float(revenue),
        "wasted_cost": float(wasted),
        "missed_opportunity": float(missed),
        "net_profit": float(net),
        "profit_per_client": float(net / total) if total > 0 else 0.0,
        "calls_made": int(tp + fp),
        "total_clients": int(total),
        "subscribers_found": int(tp),
        "subscribers_missed": int(fn),
        "catch_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "call_efficiency": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "missed_annual_nim": float(fn * BANKING_ECONOMICS["annual_nim_revenue"]),
        "missed_lifetime_nim": float(fn * BANKING_ECONOMICS["lifetime_nim_revenue"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
SELECTION_WEIGHTS = {
    "net_profit": 0.40,
    "test_recall": 0.25,
    "test_roc_auc": 0.20,
    "calibration": 0.15,
}


def composite_model_score(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models using weighted composite score across multiple criteria.

    Scoring:
      1. Min-max normalize each metric to [0, 1]
      2. Apply weights: profit 40%, recall 25%, AUC 20%, calibration 15%
      3. Sum weighted scores → composite score
      4. Rank by composite score (higher = better)
    """
    df = comp_df.copy()

    metrics_config = {
        "net_profit": {"col": "net_profit", "invert": False},
        "test_recall": {"col": "test_recall", "invert": False},
        "test_roc_auc": {"col": "test_roc_auc", "invert": False},
        "calibration": {"col": "brier_score", "invert": True},
    }

    composite = np.zeros(len(df))

    for metric_name, config in metrics_config.items():
        col = config["col"]
        if col not in df.columns:
            logger.warning("Column %s not found, skipping %s", col, metric_name)
            continue

        values = df[col].astype(float)
        if config["invert"]:
            values = 1 - values

        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = pd.Series(1.0, index=values.index)

        weight = SELECTION_WEIGHTS[metric_name]
        df[f"norm_{metric_name}"] = normalized.round(4)
        composite += weight * normalized.values

    df["composite_score"] = np.round(composite, 4)
    df["overall_rank"] = df["composite_score"].rank(ascending=False).astype(int)

    reasons = []
    for _, row in df.iterrows():
        strengths = []
        if row.get("norm_net_profit", 0) >= 0.8:
            strengths.append("top profit")
        if row.get("norm_test_recall", 0) >= 0.8:
            strengths.append("high recall")
        if row.get("norm_test_roc_auc", 0) >= 0.8:
            strengths.append("strong AUC")
        if row.get("norm_calibration", 0) >= 0.8:
            strengths.append("well calibrated")
        reasons.append(", ".join(strengths) if strengths else "balanced")
    df["strengths"] = reasons

    return df.sort_values("composite_score", ascending=False)


def select_best_model(comp_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select best model using composite scoring."""
    scored = composite_model_score(comp_df)
    best = scored.iloc[0]["model"]

    logger.info("")
    logger.info("═══ COMPOSITE MODEL SELECTION ═══")
    logger.info(
        "  Weights: Profit=%.0f%% · Recall=%.0f%% · AUC=%.0f%% · Calibration=%.0f%%",
        SELECTION_WEIGHTS["net_profit"] * 100,
        SELECTION_WEIGHTS["test_recall"] * 100,
        SELECTION_WEIGHTS["test_roc_auc"] * 100,
        SELECTION_WEIGHTS["calibration"] * 100,
    )
    # FIX: Pre-format profit string to avoid %,.0f which Python's % operator
    # doesn't support (the comma flag only works with str.format / f-strings)
    for _, row in scored.iterrows():
        marker = "→" if row["model"] == best else " "
        profit_str = f"${row['net_profit']:,.0f}"
        logger.info(
            "  %s #%d %-22s │ composite=%.4f │ profit=%-12s  recall=%.3f  "
            "auc=%.4f  brier=%.4f │ %s",
            marker,
            int(row["overall_rank"]),
            row["model"],
            row["composite_score"],
            profit_str,
            row["test_recall"],
            row["test_roc_auc"],
            row["brier_score"],
            row.get("strengths", ""),
        )
    logger.info("  ══════════════════════════════════════")
    logger.info(
        "  🏆 Selected: %s (composite score: %.4f)",
        best,
        scored.iloc[0]["composite_score"],
    )

    return best, scored


# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def selection_sensitivity_analysis(comp_df: pd.DataFrame) -> dict[str, str]:
    """Check if the selected model changes under different selection criteria."""
    results = {}

    results["max_profit"] = comp_df.loc[comp_df["net_profit"].idxmax(), "model"]
    results["max_auc"] = comp_df.loc[comp_df["test_roc_auc"].idxmax(), "model"]
    results["max_recall"] = comp_df.loc[comp_df["test_recall"].idxmax(), "model"]
    results["best_calibration"] = comp_df.loc[comp_df["brier_score"].idxmin(), "model"]

    scored = composite_model_score(comp_df)
    results["composite"] = scored.iloc[0]["model"]

    alt_weights = {
        "profit_heavy": {
            "net_profit": 0.60,
            "test_recall": 0.20,
            "test_roc_auc": 0.10,
            "calibration": 0.10,
        },
        "recall_heavy": {
            "net_profit": 0.25,
            "test_recall": 0.45,
            "test_roc_auc": 0.15,
            "calibration": 0.15,
        },
        "balanced_equal": {
            "net_profit": 0.25,
            "test_recall": 0.25,
            "test_roc_auc": 0.25,
            "calibration": 0.25,
        },
    }

    for profile_name, weights in alt_weights.items():
        original = SELECTION_WEIGHTS.copy()
        SELECTION_WEIGHTS.update(weights)
        scored_alt = composite_model_score(comp_df)
        results[profile_name] = scored_alt.iloc[0]["model"]
        SELECTION_WEIGHTS.update(original)

    unique_winners = set(results.values())
    if len(unique_winners) == 1:
        logger.info("  ✅ Selection is ROBUST — same model wins under all criteria")
    else:
        logger.info("  ⚠️  Selection varies by criteria: %s", results)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RECALL-FOCUSED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def recall_analysis(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Show recall vs precision vs profit trade-off at different thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        profit = tp * (VALUE_TP - COST_FP) - fp * COST_FP - fn * COST_FN
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        rows.append(
            {
                "threshold": round(float(t), 2),
                "recall": round(recall, 3),
                "precision": round(precision, 3),
                "calls_made": int(tp + fp),
                "subscribers_caught": int(tp),
                "subscribers_missed": int(fn),
                "false_alarms": int(fp),
                "net_profit": round(profit, 0),
                "missed_nim_revenue": round(
                    fn * BANKING_ECONOMICS["annual_nim_revenue"], 0
                ),
            }
        )

    return pd.DataFrame(rows)
