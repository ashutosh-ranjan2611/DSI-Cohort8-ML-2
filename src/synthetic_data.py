"""
synthetic_data.py  –  Generate synthetic customer data for the Call Smarter app.

Produces records that match the **bank-additional-full.csv** schema used by
our trained pipeline (19 features, no `duration`, no `balance`).

Distributions and ranges are calibrated to the real training data so that
the model receives realistic input and returns meaningful probabilities.

Usage
-----
    python src/synthetic_data.py              # 100 rows (default)
    python src/synthetic_data.py --rows 500   # custom count
    python src/synthetic_data.py --seed 99    # reproducible different set
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Categorical value pools (from bank-additional-full.csv) ─────────
JOBS = [
    "admin.", "blue-collar", "technician", "services", "management",
    "retired", "entrepreneur", "self-employed", "housemaid", "unemployed",
    "student", "unknown",
]
JOB_WEIGHTS = [0.25, 0.22, 0.16, 0.10, 0.07, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01]

MARITAL = ["married", "single", "divorced", "unknown"]
MARITAL_WEIGHTS = [0.61, 0.28, 0.10, 0.01]

EDUCATION = [
    "university.degree", "high.school", "basic.9y", "professional.course",
    "basic.4y", "basic.6y", "unknown", "illiterate",
]
EDUCATION_WEIGHTS = [0.30, 0.23, 0.15, 0.13, 0.10, 0.06, 0.03, 0.00]  # illiterate ≈ 0.04%

CONTACT = ["cellular", "telephone"]
CONTACT_WEIGHTS = [0.64, 0.36]

MONTHS = ["may", "jul", "aug", "jun", "nov", "apr", "oct", "sep", "mar", "dec"]
MONTH_WEIGHTS = [0.33, 0.17, 0.15, 0.13, 0.10, 0.06, 0.02, 0.01, 0.01, 0.02]

DAYS = ["mon", "tue", "wed", "thu", "fri"]
DAY_WEIGHTS = [0.20, 0.20, 0.20, 0.20, 0.20]

POUTCOME = ["nonexistent", "failure", "success"]
POUTCOME_WEIGHTS = [0.86, 0.10, 0.04]

DEFAULT_VALS = ["no", "unknown", "yes"]
DEFAULT_WEIGHTS = [0.79, 0.21, 0.00]  # yes is extremely rare (< 0.1%)

HOUSING_VALS = ["yes", "no", "unknown"]
HOUSING_WEIGHTS = [0.53, 0.45, 0.02]

LOAN_VALS = ["no", "yes", "unknown"]
LOAN_WEIGHTS = [0.83, 0.15, 0.02]


def generate_synthetic_data(num_records: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer data matching the bank-additional-full schema.

    Parameters
    ----------
    num_records : int
        Number of rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with the 19 model-input features (no ``duration``, no ``y``).
    """
    rng = np.random.default_rng(seed)

    # ── Numeric features (calibrated to real distributions) ─────────
    age = rng.integers(17, 99, size=num_records)

    # campaign: right-skewed, most 1-3, a few outliers up to ~50
    campaign = np.clip(rng.exponential(scale=1.8, size=num_records).astype(int) + 1, 1, 56)

    # pdays: 86% are 999 (never contacted), rest 0-27
    pdays_mask = rng.random(num_records) < 0.86
    pdays = np.where(pdays_mask, 999, rng.integers(0, 28, size=num_records))

    # previous: mostly 0, max 7
    previous = np.clip(rng.poisson(lam=0.17, size=num_records), 0, 7)

    # Macroeconomic indicators (realistic ranges from training data)
    emp_var_rate = rng.uniform(-3.4, 1.4, size=num_records).round(1)
    cons_price_idx = rng.uniform(92.201, 94.767, size=num_records).round(3)
    cons_conf_idx = rng.uniform(-50.8, -26.9, size=num_records).round(1)
    euribor3m = rng.uniform(0.634, 5.045, size=num_records).round(3)
    nr_employed = rng.uniform(4963.6, 5228.1, size=num_records).round(1)

    # ── Categorical features ───────────────────────────────────────
    data = {
        "age": age,
        "job": rng.choice(JOBS, num_records, p=JOB_WEIGHTS),
        "marital": rng.choice(MARITAL, num_records, p=MARITAL_WEIGHTS),
        "education": rng.choice(EDUCATION, num_records, p=EDUCATION_WEIGHTS),
        "default": rng.choice(DEFAULT_VALS, num_records, p=DEFAULT_WEIGHTS),
        "housing": rng.choice(HOUSING_VALS, num_records, p=HOUSING_WEIGHTS),
        "loan": rng.choice(LOAN_VALS, num_records, p=LOAN_WEIGHTS),
        "contact": rng.choice(CONTACT, num_records, p=CONTACT_WEIGHTS),
        "month": rng.choice(MONTHS, num_records, p=MONTH_WEIGHTS),
        "day_of_week": rng.choice(DAYS, num_records, p=DAY_WEIGHTS),
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": rng.choice(POUTCOME, num_records, p=POUTCOME_WEIGHTS),
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    df = pd.DataFrame(data)

    # Ensure correct column order (same as FEATURES in app/main.py)
    col_order = [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "campaign", "pdays", "previous",
        "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed",
    ]
    df = df[col_order]

    # ── Save to data/raw/ ──────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "raw"
    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "synthetic_customers.csv"
    df.to_csv(output_path, index=False)  # comma-separated (standard CSV)
    print(f"Generated {num_records} synthetic customer records → {output_path}")

    return df


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic bank marketing data")
    parser.add_argument("--rows", type=int, default=100, help="Number of records (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    df = generate_synthetic_data(args.rows, args.seed)

    # Print summary
    print(f"\nDataset Summary ({len(df)} records, {len(df.columns)} features)")
    print("=" * 55)

    print("\nCategorical distributions:")
    for col in ["job", "marital", "education", "contact", "poutcome"]:
        print(f"\n  {col}:")
        counts = df[col].value_counts()
        for val, n in counts.items():
            print(f"    {val:.<28s} {n:>4d}  ({n/len(df):.0%})")

    print("\nNumeric statistics:")
    num_cols = ["age", "campaign", "pdays", "previous",
                "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                "euribor3m", "nr.employed"]
    print(df[num_cols].describe().round(2).to_string())
