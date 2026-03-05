"""Stratified train/val/test splitting preserving class ratio."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
REFERENCE_DIR = Path("data/reference")
TARGET = "y"
SEED = 42


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage stratified split into train/val/test."""
    df_train, df_rest = train_test_split(
        df, test_size=(val_size + test_size), stratify=df[TARGET], random_state=SEED
    )
    relative_test = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_rest, test_size=relative_test, stratify=df_rest[TARGET], random_state=SEED
    )

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        logger.info(
            "%s: %d records, positive rate: %.3f",
            name,
            len(split),
            split[TARGET].mean(),
        )

    return df_train, df_val, df_test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save splits as Parquet + training reference for drift detection."""
    for d in [PROCESSED_DIR, REFERENCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    train.to_parquet(REFERENCE_DIR / "train_reference.parquet", index=False)

    logger.info("Splits saved to %s", PROCESSED_DIR)
