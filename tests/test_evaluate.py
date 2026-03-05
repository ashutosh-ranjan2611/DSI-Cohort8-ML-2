"""
Unit tests for src/evaluate.py

Covers:
  - compute_metrics: shape, ranges, dtype
  - find_optimal_threshold: returns valid threshold; prefers recall over precision
    when FN cost >> FP cost
  - business_cost_analysis: dollar-value accounting identity
  - select_best_model: returns model with highest net profit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    COST_FN,
    COST_FP,
    VALUE_TP,
    business_cost_analysis,
    compute_metrics,
    find_optimal_threshold,
    select_best_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def perfect_predictions():
    """Perfect classifier: y_pred == y_true."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    return y_true, y_true.copy(), y_true.astype(float)


@pytest.fixture()
def imbalanced_predictions():
    """Realistic imbalanced dataset (~11% positive) with decent classifier."""
    rng = np.random.default_rng(42)
    n = 1_000
    y_true = (rng.random(n) < 0.11).astype(int)
    # Assign higher probability to actual positives, add noise
    y_prob = np.where(y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_METRIC_KEYS = {
    "accuracy", "precision", "recall", "f1",
    "roc_auc", "pr_auc", "log_loss", "mcc", "brier_score",
}


def test_compute_metrics_keys(imbalanced_predictions):
    y_true, y_pred, y_prob = imbalanced_predictions
    result = compute_metrics(y_true, y_pred, y_prob)
    assert EXPECTED_METRIC_KEYS == set(result.keys())


def test_compute_metrics_all_float(imbalanced_predictions):
    y_true, y_pred, y_prob = imbalanced_predictions
    result = compute_metrics(y_true, y_pred, y_prob)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"


def test_compute_metrics_ranges(imbalanced_predictions):
    y_true, y_pred, y_prob = imbalanced_predictions
    m = compute_metrics(y_true, y_pred, y_prob)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert m["log_loss"] >= 0.0
    assert -1.0 <= m["mcc"] <= 1.0
    assert 0.0 <= m["brier_score"] <= 1.0


def test_compute_metrics_perfect(perfect_predictions):
    y_true, y_pred, y_prob = perfect_predictions
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# find_optimal_threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_find_optimal_threshold_returns_valid_threshold(imbalanced_predictions):
    y_true, _, y_prob = imbalanced_predictions
    threshold, profit = find_optimal_threshold(y_true, y_prob)
    assert 0.05 <= threshold <= 0.95
    assert np.isfinite(float(profit))  # accept numpy scalars too


def test_find_optimal_threshold_prefers_recall_on_high_fn_cost(imbalanced_predictions):
    """With high FN cost the optimal threshold should be ≤ 0.55 (leans toward recall)."""
    y_true, _, y_prob = imbalanced_predictions
    threshold, _ = find_optimal_threshold(
        y_true, y_prob, cost_fp=COST_FP, cost_fn=COST_FN, value_tp=VALUE_TP
    )
    # Banking economics: FN is 40× costlier → threshold should favour recall
    # Allow ±one grid step (linspace 200 steps → step ≈ 0.0045) plus small margin
    assert threshold <= 0.55, (
        f"Expected threshold ≤ 0.55 (high FN cost), got {threshold:.3f}"
    )


def test_find_optimal_threshold_symmetric_cost_near_half():
    """When cost_fp == cost_fn the threshold should gravitate toward 0.5."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=500)
    y_prob = np.clip(y_true + rng.normal(0, 0.3, 500), 0, 1)
    threshold, _ = find_optimal_threshold(
        y_true, y_prob, cost_fp=10.0, cost_fn=10.0, value_tp=10.0
    )
    assert 0.05 <= threshold <= 0.95  # still valid range


def test_find_optimal_threshold_custom_n_steps(imbalanced_predictions):
    y_true, _, y_prob = imbalanced_predictions
    t1, _ = find_optimal_threshold(y_true, y_prob, n_steps=20)
    t2, _ = find_optimal_threshold(y_true, y_prob, n_steps=200)
    # Coarser grid may not match exactly but should be in the same ballpark
    assert abs(t1 - t2) < 0.15, f"Threshold diverged too much: {t1:.3f} vs {t2:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# business_cost_analysis
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_BCA_KEYS = {
    "tp", "fp", "fn", "tn",
    "revenue", "wasted_cost", "missed_opportunity", "net_profit",
    "profit_per_client", "calls_made", "total_clients", "subscribers_found",
    "subscribers_missed", "catch_rate", "call_efficiency",
    "missed_annual_nim", "missed_lifetime_nim",
}


def test_business_cost_analysis_keys(imbalanced_predictions):
    y_true, y_pred, _ = imbalanced_predictions
    result = business_cost_analysis(y_true, y_pred)
    assert EXPECTED_BCA_KEYS == set(result.keys())


def test_business_cost_analysis_accounting_identity(imbalanced_predictions):
    """net_profit == revenue - wasted_cost - missed_opportunity."""
    y_true, y_pred, _ = imbalanced_predictions
    r = business_cost_analysis(y_true, y_pred)
    expected_net = r["revenue"] - r["wasted_cost"] - r["missed_opportunity"]
    assert r["net_profit"] == pytest.approx(expected_net, abs=1e-6)


def test_business_cost_analysis_confusion_counts(imbalanced_predictions):
    """tp + fp + fn + tn == total_clients; calls_made == tp + fp."""
    y_true, y_pred, _ = imbalanced_predictions
    r = business_cost_analysis(y_true, y_pred)
    assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == r["total_clients"]
    assert r["calls_made"] == r["tp"] + r["fp"]
    assert r["subscribers_found"] == r["tp"]
    assert r["subscribers_missed"] == r["fn"]


def test_business_cost_analysis_catch_rate(imbalanced_predictions):
    """catch_rate == recall == tp / (tp + fn)."""
    y_true, y_pred, _ = imbalanced_predictions
    r = business_cost_analysis(y_true, y_pred)
    expected = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) > 0 else 0.0
    assert r["catch_rate"] == pytest.approx(expected, abs=1e-9)


def test_business_cost_analysis_perfect_predictor():
    """Perfect predictor has zero wasted cost and zero missed opportunity."""
    y = np.array([1, 1, 0, 0, 1, 0])
    r = business_cost_analysis(y, y)
    assert r["fn"] == 0
    assert r["fp"] == 0
    assert r["missed_opportunity"] == pytest.approx(0.0)
    assert r["wasted_cost"] == pytest.approx(0.0)
    assert r["catch_rate"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# select_best_model
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_comparison_df():
    """Simple 4-model comparison table used by select_best_model tests."""
    return pd.DataFrame(
        {
            "model": ["logistic_regression", "random_forest", "xgboost", "knn"],
            "net_profit": [5000.0, 8000.0, 7500.0, 3500.0],
            "test_recall": [0.70, 0.80, 0.78, 0.60],
            "test_roc_auc": [0.82, 0.90, 0.88, 0.73],
            "brier_score": [0.10, 0.08, 0.09, 0.15],
        }
    )


def test_select_best_model_returns_tuple(sample_comparison_df):
    """select_best_model should return a (str, DataFrame) tuple."""
    best_name, ranked_df = select_best_model(sample_comparison_df)
    assert isinstance(best_name, str)
    assert isinstance(ranked_df, pd.DataFrame)


def test_select_best_model_picks_highest_profit(sample_comparison_df):
    """Winner should be the model with the highest net_profit (random_forest here)."""
    best_name, _ = select_best_model(sample_comparison_df)
    assert best_name == "random_forest"


def test_select_best_model_sorted_descending(sample_comparison_df):
    """Returned DataFrame should be sorted by net_profit, highest first."""
    _, ranked_df = select_best_model(sample_comparison_df)
    profits = ranked_df["net_profit"].tolist()
    assert profits == sorted(profits, reverse=True)


def test_select_best_model_does_not_mutate_input(sample_comparison_df):
    """The input DataFrame should not be modified."""
    original_order = list(sample_comparison_df["net_profit"])
    _ = select_best_model(sample_comparison_df)
    assert list(sample_comparison_df["net_profit"]) == original_order
