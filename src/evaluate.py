"""
Model evaluation — metrics, cost-optimal threshold, and business impact.

This file answers three questions:
  1. How well does the model separate subscribers from non-subscribers? (compute_metrics)
  2. At what probability threshold should we start calling a customer? (find_optimal_threshold)
  3. In real dollar terms, how much better is our model than random guessing? (business_cost_analysis)

BANKING ECONOMICS USED THROUGHOUT
==================================================
When a customer places a term deposit, the bank earns via Net Interest Margin:

  Average term deposit = $10,000
  Bank pays depositor  =  2.0% interest
  Bank lends funds at  =  8.4% interest
  Net Interest Margin  =  6.4% per year
  Lifetime revenue     = $1,280 over a 2-year average term

  False Negative cost  = $200  (we miss a real subscriber — worst outcome)
  False Positive cost  =  $5   (we waste a phone call)
  True Positive value  = $200  (we correctly identify and call a subscriber)

  Cost ratio FN:FP = 40:1, so the model should lean toward high recall
  (it is 40 times worse to miss a real subscriber than to make one extra call).
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
    log_loss,
    matthews_corrcoef,
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
    """
    Compute the standard set of classification metrics.

    Each metric tells a different part of the story:
      accuracy   : overall % correct (misleading when classes are imbalanced)
      precision  : of all customers we called, what fraction subscribed?
      recall     : of all real subscribers, what fraction did we catch?
      f1         : harmonic mean of precision and recall
      roc_auc    : how well does the model rank-order customers? (1.0 = perfect)
      pr_auc     : precision-recall trade-off area (better than ROC for imbalanced data)
      log_loss   : how confident and correct are the probability scores?
      mcc        : Matthews Correlation Coefficient — a single balanced score
      brier_score: average squared error on probabilities (lower = better calibrated)
    """
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_true, y_prob)),
        "pr_auc":      float(average_precision_score(y_true, y_prob)),
        "log_loss":    float(log_loss(y_true, y_prob)),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
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
    """
    Find the probability threshold that maximises net business profit on the test set.

    How it works:
      - By default sklearn classifies as 1 if probability >= 0.5.
      - For this problem that is too conservative: a missed subscriber costs $200,
        but a wasted call only costs $5, so we accept more false positives.
      - We try every threshold from 0.05 to 0.95 in 200 steps and keep the one
        that produces the highest net profit (TP * $195 - FP * $5 - FN * $200).
    """
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
    """
    Translate the confusion matrix into dollar figures.

    Returns a dict with:
      - tp/fp/fn/tn   : Raw counts from the confusion matrix.
      - net_profit    : Total dollar value (TP revenue - FP costs - FN costs).
      - catch_rate    : Fraction of real subscribers we identified (= recall).
      - call_efficiency: Fraction of calls that resulted in a subscription (= precision).
    """
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


# -------------------------------------------------------------------------------
# Model selection
# -------------------------------------------------------------------------------

def select_best_model(comp_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Pick the best model: simply choose the one with the highest net profit.

    Net profit directly reflects the business goal (maximise revenue from
    term-deposit subscriptions, minus the cost of wasted calls and missed
    subscribers), so it is the most honest single criterion.

    Returns:
        best_name (str)         : name of the winning model
        comp_df (pd.DataFrame)  : the same table sorted by net_profit descending
    """
    # Sort the results table so the best model appears first
    sorted_df = comp_df.sort_values("net_profit", ascending=False).reset_index(drop=True)
    best_name = sorted_df.iloc[0]["model"]

    logger.info("")
    logger.info("=== MODEL SELECTION (by Net Profit) ===")
    for _, row in sorted_df.iterrows():
        marker = ">>> WINNER" if row["model"] == best_name else "       "
        profit_str = f"${row['net_profit']:,.0f}"
        logger.info(
            "  %s  %-22s | profit=%-12s  recall=%.3f  auc=%.4f",
            marker, row["model"], profit_str, row["test_recall"], row["test_roc_auc"],
        )
    logger.info("  Selected: %s", best_name)

    return best_name, sorted_df


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
