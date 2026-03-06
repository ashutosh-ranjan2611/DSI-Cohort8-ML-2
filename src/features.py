"""
Feature engineering — preparing the bank marketing data for our models.

What happens in this file:
  1. PdaysTransformer  : fixes a tricky column (pdays uses 999 as a placeholder, not a real number)
  2. build_preprocessor: scales numbers, encodes categories ready for sklearn
  3. build_pipeline    : wraps everything into a single sklearn Pipeline

Why a Pipeline?
  - Prevents data leakage: the scaler learns only from training data.
  - Makes deployment easy: one object handles all pre-processing at prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "y"

EDUCATION_ORDER = [
    "illiterate",
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "professional.course",
    "university.degree",
    "unknown",
]

NOMINAL_FEATURES = [
    "job",
    "marital",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


class PdaysTransformer(BaseEstimator, TransformerMixin):
    """
    Fix the 'pdays' column before passing it to the model.

    The raw 'pdays' value is tricky:
      - 999 means the customer was NEVER contacted in a previous campaign.
        It does NOT mean "999 days ago" — it's just a placeholder.
      - Any other value (e.g. 6) means "last contacted 6 days ago".

    Treating 999 as a real number would mislead the model completely.
    Instead, we replace 'pdays' with two new, clearer columns:
      - 'was_previously_contacted' : 1 if they were called before, 0 if not
      - 'pdays_log'                : log(1 + days), set to 0 if never contacted

    Using log-transform on days compresses the tails — the difference between
    1 and 5 days matters more than the difference between 100 and 105 days.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["was_previously_contacted"] = (X["pdays"] != 999).astype(int)
        X["pdays_log"] = np.where(X["pdays"] != 999, np.log1p(X["pdays"]), 0.0)
        X = X.drop(columns=["pdays"])
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(["was_previously_contacted", "pdays_log"])
        out = [f for f in input_features if f != "pdays"]
        return np.array(out + ["was_previously_contacted", "pdays_log"])


# -------------------------------------------------------------------------------
# Pipeline builders
# -------------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer that prepares features for the models.

    Three types of features need different treatment:
      - Numbers      : StandardScaler centres them around zero (required by Logistic Regression,
                       harmless for tree models like Random Forest and XGBoost).
      - Education    : OrdinalEncoder — there is a clear ranking from 'illiterate' up to
                       'university.degree', so we assign 0, 1, 2 … instead of one-hot.
      - All other categories : OneHotEncoder — no natural order, so each category gets
                                its own binary 0/1 column.
    """
    # After PdaysTransformer, the raw 'pdays' column is gone and replaced by
    # 'was_previously_contacted' and 'pdays_log', so we adjust the numeric list.
    post_pdays_numeric = [f for f in NUMERIC_FEATURES if f != "pdays"] + [
        "was_previously_contacted",
        "pdays_log",
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), post_pdays_numeric),
            (
                "ord",
                Pipeline(
                    [
                        (
                            "ordinal",
                            OrdinalEncoder(
                                categories=[EDUCATION_ORDER],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        )
                    ]
                ),
                ["education"],
            ),
            (
                "nom",
                Pipeline(
                    [
                        (
                            "onehot",
                            OneHotEncoder(
                                drop="if_binary",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        )
                    ]
                ),
                list(NOMINAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """
    Assemble the full 3-step sklearn Pipeline for a given classifier.

    Step 1 — pdays_transform : Fixes the sentinel 999 in the 'pdays' column.
    Step 2 — preprocessor    : Scales numbers, encodes categories.
    Step 3 — classifier      : The actual model (LR, RF, XGBoost, or KNN).

    Using a Pipeline means:
      - The scaler sees ONLY training data (no leakage from the test set).
      - At prediction time, you just call pipeline.predict(X) — everything
        is applied in the correct order automatically.
    """
    return Pipeline(
        steps=[
            ("pdays_transform", PdaysTransformer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )
