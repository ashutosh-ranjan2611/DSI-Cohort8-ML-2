"""Unit tests for src/train.py — model registry, FIXED_PARAMS, training helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train import FIXED_PARAMS, MODEL_CLASSES, train_final_model, tune_model

# ── Helpers ───────────────────────────────────────────────────────────────────

EXPECTED_MODELS = {"logistic_regression", "random_forest", "xgboost", "knn"}


def _make_xy(n: int = 200, n_pos: int = 30, seed: int = 42) -> tuple:
    """Small synthetic dataset matching bank-additional schema columns."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "age": rng.randint(18, 90, n),
            "job": rng.choice(["admin.", "technician", "blue-collar", "retired"], n),
            "marital": rng.choice(["married", "single", "divorced"], n),
            "education": rng.choice(["university.degree", "high.school"], n),
            "default": rng.choice(["no", "unknown"], n),
            "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n),
            "contact": rng.choice(["cellular", "telephone"], n),
            "month": rng.choice(["may", "jul", "nov"], n),
            "day_of_week": rng.choice(["mon", "tue", "wed"], n),
            "campaign": rng.randint(1, 10, n),
            "pdays": rng.choice([999, 6], n),
            "previous": rng.randint(0, 3, n),
            "poutcome": rng.choice(["nonexistent", "failure", "success"], n),
            "emp.var.rate": rng.uniform(-3.5, 1.5, n).round(1),
            "cons.price.idx": rng.uniform(92.0, 95.0, n).round(3),
            "cons.conf.idx": rng.uniform(-50.0, -25.0, n).round(1),
            "euribor3m": rng.uniform(0.5, 5.0, n).round(3),
            "nr.employed": rng.uniform(4960, 5230, n).round(1),
        }
    )
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    return X, y


# ── Model registry completeness ───────────────────────────────────────────────


def test_model_classes_has_all_four_models():
    assert set(MODEL_CLASSES.keys()) == EXPECTED_MODELS


def test_fixed_params_has_all_four_models():
    assert set(FIXED_PARAMS.keys()) == EXPECTED_MODELS


def test_fixed_params_non_empty_for_tunable_models():
    """LR, RF, and XGB must have fixed params; KNN dict may be empty."""
    for name in EXPECTED_MODELS - {"knn"}:
        assert FIXED_PARAMS[name], f"FIXED_PARAMS['{name}'] is empty"


def test_xgboost_fixed_params_no_hardcoded_scale_pos_weight():
    """scale_pos_weight should NOT be hardcoded — it must be computed at runtime."""
    spw = FIXED_PARAMS["xgboost"].get("scale_pos_weight")
    assert spw is None or not isinstance(spw, (int, float)), (
        "scale_pos_weight must be computed dynamically, not hardcoded in FIXED_PARAMS"
    )


# ── train_final_model ─────────────────────────────────────────────────────────


def test_train_logistic_regression_returns_pipeline():
    X, y = _make_xy()
    params = {**FIXED_PARAMS["logistic_regression"], "C": 0.1}
    pipe = train_final_model("logistic_regression", params, X, y)
    assert isinstance(pipe, Pipeline)


def test_train_logistic_regression_predict_proba():
    X, y = _make_xy()
    params = {**FIXED_PARAMS["logistic_regression"], "C": 0.1}
    pipe = train_final_model("logistic_regression", params, X, y)
    probs = pipe.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((probs >= 0) & (probs <= 1))


def test_train_knn_predicts_binary_labels():
    X, y = _make_xy()
    params = {
        **FIXED_PARAMS["knn"],
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
    }
    pipe = train_final_model("knn", params, X, y)
    preds = pipe.predict(X[:10])
    assert set(preds).issubset({0, 1})


# ── tune_model (1-trial smoke tests) ─────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["logistic_regression", "knn"])
def test_tune_model_returns_expected_keys(model_name):
    """tune_model with n_trials=1 must complete and return best_params + best_auc."""
    X, y = _make_xy()
    result = tune_model(model_name, X, y, n_trials=1, cv_n_jobs=1)
    assert {"best_params", "best_auc", "study"}.issubset(result.keys())


@pytest.mark.parametrize("model_name", ["logistic_regression", "knn"])
def test_tune_model_auc_in_valid_range(model_name):
    X, y = _make_xy()
    result = tune_model(model_name, X, y, n_trials=1, cv_n_jobs=1)
    assert 0.0 <= result["best_auc"] <= 1.0


@pytest.mark.parametrize("model_name", ["logistic_regression", "knn"])
def test_tune_model_best_params_is_dict(model_name):
    X, y = _make_xy()
    result = tune_model(model_name, X, y, n_trials=1, cv_n_jobs=1)
    assert isinstance(result["best_params"], dict)
    assert len(result["best_params"]) > 0
