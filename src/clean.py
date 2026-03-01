"""
Data cleaning — handle duplicates, outliers, skewness, 'unknown' values, and drop duration.

Strategy:
  Duplicates:     detect and remove exact duplicate rows
  Outliers:       clip numeric features at 1st/99th percentile
  Skewness:       log1p transform for highly skewed features (campaign, previous)
  Multicollinearity: detect and report VIF for numeric features
  Unknowns:
    - job, marital, housing, loan (low unknowns): impute with mode
    - education, default (high unknowns or informative): keep 'unknown' as category
  Duration:       DROPPED in production (post-hoc leakage)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IMPUTE_MODE_COLS = ["job", "marital", "housing", "loan"]
KEEP_UNKNOWN_COLS = ["education", "default"]

# Numeric columns to check for outliers
# (excluding binary/indicator columns and the target)
OUTLIER_COLS = [
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


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove exact duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Removed %d duplicate rows (%d → %d)", n_removed, n_before, len(df))
    else:
        logger.info("No duplicate rows found (%d rows)", len(df))
    return df


def handle_outliers(df: pd.DataFrame, method: str = "clip") -> pd.DataFrame:
    """
    Detect and handle outliers in numeric columns.

    Methods:
      'clip'   — cap values at 1st and 99th percentile (default, preserves rows)
      'report' — only log outlier counts, don't modify data

    We use percentile clipping rather than removal because:
      - Tree models (RF, XGB, LGBM) are inherently robust to outliers
      - Clipping preserves the rank ordering (important for trees)
      - Removing rows would lose valuable minority-class samples
      - LogReg benefits from reduced extreme value influence on coefficients
    """
    if df.empty:
        return df
    df = df.copy()

    outlier_summary = []
    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        iqr = q99 - q01

        # Count outliers (beyond 1st/99th percentile)
        n_low = (df[col] < q01).sum()
        n_high = (df[col] > q99).sum()
        n_outliers = n_low + n_high

        if n_outliers > 0:
            outlier_summary.append(
                {
                    "column": col,
                    "n_outliers": n_outliers,
                    "pct": round(n_outliers / len(df) * 100, 2),
                    "low_bound": round(q01, 2),
                    "high_bound": round(q99, 2),
                }
            )

            if method == "clip":
                df[col] = df[col].clip(lower=q01, upper=q99)

    if outlier_summary:
        total_outliers = sum(o["n_outliers"] for o in outlier_summary)
        action = "Clipped" if method == "clip" else "Detected"
        logger.info(
            "%s outliers in %d columns (%d total values affected)",
            action,
            len(outlier_summary),
            total_outliers,
        )
        for o in outlier_summary:
            logger.info(
                "  %s: %d outliers (%.1f%%) — bounds [%.2f, %.2f]",
                o["column"],
                o["n_outliers"],
                o["pct"],
                o["low_bound"],
                o["high_bound"],
            )
    else:
        logger.info("No significant outliers detected in numeric features")

    return df


def check_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and report skewed numeric features.
    Features with |skewness| > 2 are flagged — these benefit from log transforms.
    Note: We report but don't auto-transform here because:
      - Tree models (RF, XGB, LGBM) are invariant to monotonic transforms
      - The pipeline's StandardScaler handles scale for LogReg
      - PdaysTransformer already applies log1p to pdays
    """
    skew_cols = [
        "age",
        "campaign",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    available = [
        c
        for c in skew_cols
        if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]

    if not available:
        return df

    skew_data = []
    for col in available:
        skew_val = df[col].skew()
        kurt_val = df[col].kurtosis()
        skew_data.append(
            {
                "column": col,
                "skewness": round(skew_val, 2),
                "kurtosis": round(kurt_val, 2),
                "flag": "⚠️ HIGH" if abs(skew_val) > 2 else "OK",
            }
        )

    flagged = [s for s in skew_data if s["flag"] != "OK"]
    if flagged:
        logger.info("Skewness check — %d highly skewed features:", len(flagged))
        for s in flagged:
            logger.info(
                "  %s: skew=%.2f, kurtosis=%.2f %s",
                s["column"],
                s["skewness"],
                s["kurtosis"],
                s["flag"],
            )
    else:
        logger.info("Skewness check — all numeric features within normal range")

    return df


def check_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect highly correlated feature pairs (|r| > 0.8).

    Reports but does NOT remove features because:
      - Tree models are immune to multicollinearity
      - LogReg uses L1 regularization which handles it via sparsity
      - Removing features would reduce ensemble diversity
    """
    num_cols = [
        "age",
        "campaign",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    available = [
        c for c in num_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]

    if len(available) < 2:
        return df

    corr_matrix = df[available].corr().abs()
    high_corr_pairs = []

    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            r = corr_matrix.iloc[i, j]
            if r > 0.8:
                high_corr_pairs.append((available[i], available[j], round(r, 3)))

    if high_corr_pairs:
        logger.info(
            "Multicollinearity — %d highly correlated pairs (|r| > 0.8):",
            len(high_corr_pairs),
        )
        for f1, f2, r in high_corr_pairs:
            logger.info("  %s ↔ %s: r=%.3f", f1, f2, r)
        logger.info(
            "  → Handled by: L1 regularization (LogReg) + tree invariance (RF/XGB/LGBM)"
        )
    else:
        logger.info("Multicollinearity — no highly correlated pairs found")

    return df


def check_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check cardinality of categorical features and flag low-count levels.
    Low-count categories can cause issues with stratified splits and one-hot encoding.
    """
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) == 0:
        return df

    logger.info("Cardinality check for %d categorical features:", len(cat_cols))
    for col in cat_cols:
        n_unique = df[col].nunique()
        value_counts = df[col].value_counts()
        min_count = value_counts.min()
        min_level = value_counts.idxmin()

        flag = ""
        if min_count < 50:
            flag = f" ⚠️ rare level: '{min_level}' ({min_count} rows)"
        logger.info("  %s: %d levels, min=%d%s", col, n_unique, min_count, flag)

    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Log missing value counts across all columns (NaN, not 'unknown')."""
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) > 0:
        logger.info("Missing values (NaN) found in %d columns:", len(cols_with_nulls))
        for col, count in cols_with_nulls.items():
            pct = count / len(df) * 100
            logger.info("  %s: %d (%.1f%%)", col, count, pct)
    else:
        logger.info("No NaN missing values found (note: 'unknown' handled separately)")

    return df


def clean_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or keep 'unknown' values based on per-column strategy."""
    if df.empty:
        return df
    df = df.copy()

    for col in IMPUTE_MODE_COLS:
        if col not in df.columns:
            continue
        mask = df[col] == "unknown"
        n = mask.sum()
        if n > 0:
            known = df.loc[~mask, col]
            if not known.empty:
                mode_val = known.mode()[0]
                df.loc[mask, col] = mode_val
                logger.info("Imputed %d unknowns in '%s' → '%s'", n, col, mode_val)

    for col in KEEP_UNKNOWN_COLS:
        if col in df.columns:
            n = (df[col] == "unknown").sum()
            if n > 0:
                logger.info("Kept %d unknowns in '%s' as category", n, col)

    return df


def drop_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop 'duration' — only known after call ends → data leakage.
    UCI docs explicitly warn about this.
    """
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
        logger.info("Dropped 'duration' (production mode)")
    return df


def clean_data(df: pd.DataFrame, production: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Check for NaN missing values (report)
      2. Remove exact duplicates
      3. Check cardinality of categorical features (report)
      4. Handle outliers (clip at 1st/99th percentile)
      5. Check skewness of numeric features (report)
      6. Check multicollinearity (report)
      7. Clean 'unknown' string values (impute or keep)
      8. Drop duration column (production mode)
    """
    logger.info("Starting data cleaning pipeline...")
    df = check_missing_values(df)
    df = remove_duplicates(df)
    df = check_cardinality(df)
    df = handle_outliers(df, method="clip")
    df = check_skewness(df)
    df = check_multicollinearity(df)
    df = clean_unknowns(df)
    if production:
        df = drop_duration(df)
    logger.info("Cleaning complete: %d rows, %d columns", len(df), len(df.columns))
    return df
