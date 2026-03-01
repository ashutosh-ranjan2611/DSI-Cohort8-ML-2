"""
=== FILE: tests/conftest.py ===
Shared test fixtures — synthetic data so tests never depend on real CSV files.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data():
    """200-row synthetic dataset matching bank-additional-full schema."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.randint(18, 90, n),
            "job": rng.choice(
                [
                    "admin.",
                    "technician",
                    "management",
                    "blue-collar",
                    "services",
                    "retired",
                    "unknown",
                ],
                n,
            ),
            "marital": rng.choice(
                ["married", "single", "divorced", "unknown"],
                n,
                p=[0.5, 0.3, 0.15, 0.05],
            ),
            "education": rng.choice(
                ["university.degree", "high.school", "basic.9y", "basic.4y", "unknown"],
                n,
            ),
            "default": rng.choice(["no", "unknown"], n, p=[0.8, 0.2]),
            "housing": rng.choice(["yes", "no", "unknown"], n, p=[0.5, 0.45, 0.05]),
            "loan": rng.choice(["yes", "no", "unknown"], n, p=[0.15, 0.8, 0.05]),
            "contact": rng.choice(["cellular", "telephone"], n, p=[0.65, 0.35]),
            "month": rng.choice(["may", "jul", "aug", "nov", "mar"], n),
            "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
            "campaign": rng.randint(1, 15, n),
            "pdays": rng.choice([999, 3, 6, 12], n, p=[0.85, 0.05, 0.05, 0.05]),
            "previous": rng.randint(0, 5, n),
            "poutcome": rng.choice(
                ["nonexistent", "failure", "success"], n, p=[0.85, 0.1, 0.05]
            ),
            "emp.var.rate": rng.uniform(-3.5, 1.5, n).round(1),
            "cons.price.idx": rng.uniform(92.0, 95.0, n).round(3),
            "cons.conf.idx": rng.uniform(-50.0, -25.0, n).round(1),
            "euribor3m": rng.uniform(0.5, 5.0, n).round(3),
            "nr.employed": rng.uniform(4960, 5230, n).round(1),
            "y": rng.choice([0, 1], n, p=[0.887, 0.113]),
        }
    )


@pytest.fixture
def synthetic_with_duration(synthetic_data):
    df = synthetic_data.copy()
    df["duration"] = np.random.RandomState(42).randint(10, 1000, len(df))
    return df


# ============================================================================
# === FILE: tests/test_clean.py ===
# ============================================================================
"""Unit tests for src/clean.py"""


def test_unknowns_imputed_in_job(synthetic_data):
    from src.clean import clean_unknowns

    result = clean_unknowns(synthetic_data)
    assert "unknown" not in result["job"].values


def test_unknowns_imputed_in_marital(synthetic_data):
    from src.clean import clean_unknowns

    result = clean_unknowns(synthetic_data)
    assert "unknown" not in result["marital"].values


def test_unknowns_kept_in_education(synthetic_data):
    from src.clean import clean_unknowns

    result = clean_unknowns(synthetic_data)
    if "unknown" in synthetic_data["education"].values:
        assert "unknown" in result["education"].values


def test_unknowns_kept_in_default(synthetic_data):
    from src.clean import clean_unknowns

    result = clean_unknowns(synthetic_data)
    if "unknown" in synthetic_data["default"].values:
        assert "unknown" in result["default"].values


def test_duration_dropped(synthetic_with_duration):
    from src.clean import clean_data

    result = clean_data(synthetic_with_duration, production=True)
    assert "duration" not in result.columns


def test_duration_kept_benchmark(synthetic_with_duration):
    from src.clean import clean_data

    result = clean_data(synthetic_with_duration, production=False)
    assert "duration" in result.columns


def test_empty_dataframe():
    import pandas as pd
    from src.clean import clean_unknowns

    empty = pd.DataFrame(
        columns=["job", "marital", "education", "default", "housing", "loan"]
    )
    result = clean_unknowns(empty)
    assert result.empty


# ============================================================================
# === FILE: tests/test_features.py ===
# ============================================================================
"""Unit tests for src/features.py"""
import numpy as np
import pytest


def test_pdays_sentinel():
    import pandas as pd
    from src.features import PdaysTransformer

    df = pd.DataFrame({"pdays": [999], "other": [1]})
    result = PdaysTransformer().fit_transform(df)
    assert "pdays" not in result.columns
    assert result["was_previously_contacted"].iloc[0] == 0
    assert result["pdays_log"].iloc[0] == 0.0


def test_pdays_contacted():
    import pandas as pd
    from src.features import PdaysTransformer

    df = pd.DataFrame({"pdays": [6], "other": [1]})
    result = PdaysTransformer().fit_transform(df)
    assert result["was_previously_contacted"].iloc[0] == 1
    assert result["pdays_log"].iloc[0] == pytest.approx(np.log1p(6), rel=1e-5)


def test_full_pipeline_fits(synthetic_data):
    from sklearn.linear_model import LogisticRegression
    from src.features import build_pipeline

    df = synthetic_data.drop(columns=["duration"], errors="ignore")
    X, y = df.drop(columns=["y"]), df["y"]

    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X, y)
    probs = pipe.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)


# ============================================================================
# === FILE: tests/test_schemas.py ===
# ============================================================================
"""Unit tests for evaluation functions."""
import numpy as np


def test_threshold_asymmetric_costs():
    """With FN 40x costlier than FP, threshold should be below 0.5."""
    from src.evaluate import find_optimal_threshold

    rng = np.random.RandomState(42)
    n = 1000
    y = rng.choice([0, 1], n, p=[0.9, 0.1])
    prob = np.where(y == 1, rng.uniform(0.3, 0.8, n), rng.uniform(0.05, 0.5, n))
    t, _ = find_optimal_threshold(y, prob)
    assert t < 0.5, f"Expected threshold < 0.5, got {t}"


def test_metrics_return_all_keys():
    from src.evaluate import compute_metrics

    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])
    prob = np.array([0.2, 0.6, 0.4, 0.8])
    m = compute_metrics(y, pred, prob)
    assert set(m.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }


def test_business_cost_analysis():
    from src.evaluate import business_cost_analysis

    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 0, 1, 1])
    cost = business_cost_analysis(y, pred)
    assert cost["tp"] == 2
    assert cost["fp"] == 1
    assert cost["fn"] == 1
    assert cost["tn"] == 1
    assert cost["calls_made"] == 3  # tp + fp
