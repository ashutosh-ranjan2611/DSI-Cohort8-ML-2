"""Unit tests for data cleaning module."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import clean_pipeline, clean_unknowns, drop_duration_for_production


def test_unknowns_imputed_in_job(synthetic_bank_data):
    cleaned = clean_unknowns(synthetic_bank_data)
    assert "unknown" not in cleaned["job"].values


def test_unknowns_imputed_in_marital(synthetic_bank_data):
    cleaned = clean_unknowns(synthetic_bank_data)
    assert "unknown" not in cleaned["marital"].values


def test_unknowns_imputed_in_housing(synthetic_bank_data):
    cleaned = clean_unknowns(synthetic_bank_data)
    assert "unknown" not in cleaned["housing"].values


def test_unknowns_imputed_in_loan(synthetic_bank_data):
    cleaned = clean_unknowns(synthetic_bank_data)
    assert "unknown" not in cleaned["loan"].values


def test_unknowns_kept_in_education(synthetic_bank_data):
    cleaned = clean_unknowns(synthetic_bank_data)
    # Education unknowns should be preserved
    if "unknown" in synthetic_bank_data["education"].values:
        assert "unknown" in cleaned["education"].values


def test_unknowns_kept_in_default(synthetic_bank_data):
    cleaned = clean_unknowns(synthetic_bank_data)
    if "unknown" in synthetic_bank_data["default"].values:
        assert "unknown" in cleaned["default"].values


def test_duration_dropped_in_production(synthetic_bank_data_with_duration):
    result = drop_duration_for_production(synthetic_bank_data_with_duration)
    assert "duration" not in result.columns


def test_duration_not_dropped_when_absent(synthetic_bank_data):
    # Should not error if duration doesn't exist
    result = drop_duration_for_production(synthetic_bank_data)
    assert "duration" not in result.columns


def test_clean_pipeline_production(synthetic_bank_data_with_duration):
    result = clean_pipeline(synthetic_bank_data_with_duration, production=True)
    assert "duration" not in result.columns
    assert "unknown" not in result["job"].values


def test_clean_pipeline_benchmark(synthetic_bank_data_with_duration):
    result = clean_pipeline(synthetic_bank_data_with_duration, production=False)
    assert "duration" in result.columns


def test_empty_dataframe():
    """Edge case: empty DataFrame should not crash."""
    empty_df = pd.DataFrame(columns=["job", "marital", "education", "default", "housing", "loan"])
    result = clean_unknowns(empty_df)
    assert result.empty


def test_no_unknowns_present():
    """If no unknowns exist, cleaning should be a no-op."""
    df = pd.DataFrame({
        "job": ["admin.", "technician"],
        "marital": ["married", "single"],
        "education": ["high.school", "university.degree"],
        "default": ["no", "no"],
        "housing": ["yes", "no"],
        "loan": ["no", "yes"],
    })
    result = clean_unknowns(df)
    assert result.equals(df)