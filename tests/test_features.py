"""Unit tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import PdaysTransformer, build_full_pipeline


def test_pdays_transformer_sentinel():
    """999 should be converted to was_previously_contacted=0, pdays_log=0."""
    df = pd.DataFrame({"pdays": [999], "other_col": [1]})
    transformer = PdaysTransformer()
    result = transformer.fit_transform(df)
    assert "pdays" not in result.columns
    assert result["was_previously_contacted"].iloc[0] == 0
    assert result["pdays_log"].iloc[0] == 0.0


def test_pdays_transformer_contacted():
    """Non-999 values should be flagged and log-transformed."""
    df = pd.DataFrame({"pdays": [6], "other_col": [1]})
    transformer = PdaysTransformer()
    result = transformer.fit_transform(df)
    assert result["was_previously_contacted"].iloc[0] == 1
    assert result["pdays_log"].iloc[0] == pytest.approx(np.log1p(6), rel=1e-5)


def test_pdays_transformer_mixed():
    """Mixed 999 and non-999 values."""
    df = pd.DataFrame({"pdays": [999, 3, 999, 12]})
    transformer = PdaysTransformer()
    result = transformer.fit_transform(df)
    assert list(result["was_previously_contacted"]) == [0, 1, 0, 1]


def test_full_pipeline_fits(synthetic_bank_data):
    """Pipeline should fit and produce valid probabilities."""
    from sklearn.linear_model import LogisticRegression

    df = synthetic_bank_data.copy()
    # Remove duration if present (production mode)
    df = df.drop(columns=["duration"], errors="ignore")

    X = df.drop(columns=["y"])
    y = df["y"]

    pipeline = build_full_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X[:5])

    assert probs.shape == (5, 2)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    assert np.allclose(probs.sum(axis=1), 1.0)