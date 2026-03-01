"""Test Pydantic schema validation and column name mapping."""
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from src.deployment.schemas import PredictionInput, input_to_dataframe


def test_valid_input(sample_prediction_payload):
    inp = PredictionInput(**sample_prediction_payload)
    assert inp.age == 35
    assert inp.job == "management"


def test_age_below_minimum(sample_prediction_payload):
    payload = {**sample_prediction_payload, "age": 10}
    with pytest.raises(ValidationError):
        PredictionInput(**payload)


def test_age_above_maximum(sample_prediction_payload):
    payload = {**sample_prediction_payload, "age": 150}
    with pytest.raises(ValidationError):
        PredictionInput(**payload)


def test_invalid_marital_status(sample_prediction_payload):
    payload = {**sample_prediction_payload, "marital": "complicated"}
    with pytest.raises(ValidationError):
        PredictionInput(**payload)


def test_invalid_contact_type(sample_prediction_payload):
    payload = {**sample_prediction_payload, "contact": "email"}
    with pytest.raises(ValidationError):
        PredictionInput(**payload)


def test_negative_campaign(sample_prediction_payload):
    payload = {**sample_prediction_payload, "campaign": 0}
    with pytest.raises(ValidationError):
        PredictionInput(**payload)


def test_input_to_dataframe_column_mapping(sample_prediction_payload):
    """
    FIX #2 VERIFICATION: Ensure underscore field names are mapped
    to dot column names that the sklearn pipeline expects.
    """
    inp = PredictionInput(**sample_prediction_payload)
    df = input_to_dataframe(inp)

    # These columns must have dots (what the pipeline expects)
    assert "emp.var.rate" in df.columns
    assert "cons.price.idx" in df.columns
    assert "cons.conf.idx" in df.columns
    assert "nr.employed" in df.columns

    # These should NOT exist (Pydantic underscore versions)
    assert "emp_var_rate" not in df.columns
    assert "cons_price_idx" not in df.columns
    assert "cons_conf_idx" not in df.columns
    assert "nr_employed" not in df.columns

    assert df.shape == (1, 19)