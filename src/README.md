# src/ — Core Pipeline Modules

## Overview

This directory contains the six Python modules that form the backbone of the Bank Marketing ML pipeline. Each module encapsulates a single responsibility within the end-to-end workflow, from raw data ingestion through model evaluation and selection. The modules are designed to be called sequentially by `scripts/run_pipeline.py`, but each is independently importable and testable.

**Pipeline execution order:**

```
ingest.py --> clean.py --> split.py --> features.py --> train.py --> evaluate.py
```

---

## Module Reference

### 1. ingest.py — Data Acquisition and Validation

**Purpose:** Download, extract, and validate the UCI Bank Marketing dataset from its nested ZIP archive structure.

**Key challenge solved:** The UCI archive ships as a ZIP containing another ZIP (`bank+marketing.zip` > `bank-additional.zip` > `bank-additional-full.csv`). This module handles the full extraction chain automatically.

| Function | Description |
|---|---|
| `download_and_extract()` | Downloads the archive from UCI, extracts through nested ZIPs, and writes the target CSV to `data/raw/`. Skips download if the file already exists. |
| `load_raw_data()` | Reads the CSV with semicolon delimiter, validates the expected shape (41,188 rows x 21 columns), and maps the target column (`y`) from `yes/no` to `1/0`. Raises `ValueError` on schema violations. |

**Constants:**

| Name | Value | Description |
|---|---|---|
| `DATA_URL` | UCI static URL | Source archive location |
| `EXPECTED_SHAPE` | (41188, 21) | Validation gate for data integrity |
| `TARGET_COL` | `"y"` | Binary target column name |

**Design decisions:**
- Auto-download is idempotent: repeated calls are no-ops if data exists.
- Shape validation catches silent corruption (e.g., partial downloads, schema drift).
- The outer ZIP is deleted after extraction to avoid storing redundant archives.

---

### 2. clean.py — Data Quality and Preprocessing

**Purpose:** Execute all data quality checks and transformations required before feature engineering. This module addresses seven distinct data quality concerns in a fixed order.

**Cleaning pipeline (executed in sequence by `clean_data()`):**

| Step | Function | Action | Rationale |
|---|---|---|---|
| 1 | `check_missing_values()` | Log NaN counts per column | Distinguish true nulls from domain-coded "unknown" strings |
| 2 | `remove_duplicates()` | Drop exact duplicate rows | Prevent training bias from repeated observations |
| 3 | `check_cardinality()` | Report unique levels and rare categories per categorical feature | Flag levels with fewer than 50 samples that may cause issues in stratified splits or one-hot encoding |
| 4 | `handle_outliers()` | Clip numeric features at the 1st and 99th percentiles | Reduces extreme value influence on logistic regression coefficients while preserving rank ordering for tree-based models |
| 5 | `check_skewness()` | Report skewness and kurtosis for all numeric features | Features with absolute skewness above 2.0 are flagged; tree models are invariant to monotonic transforms, so we report rather than transform |
| 6 | `check_multicollinearity()` | Report feature pairs with Pearson correlation above 0.8 | Known pairs include `euribor3m` / `emp.var.rate` / `nr.employed`; handled downstream by L1 regularization (LogReg) and inherent tree-model invariance |
| 7 | `clean_unknowns()` | Impute or retain "unknown" string values per column strategy | See strategy table below |
| 8 | `drop_duration()` | Remove `duration` column in production mode | Call duration is only known after the call ends; including it constitutes data leakage. UCI documentation explicitly warns against using this feature for predictive purposes. |

**Unknown value strategy:**

| Column(s) | Unknown Count | Strategy | Justification |
|---|---|---|---|
| `job`, `marital`, `housing`, `loan` | Low (80-990) | Mode imputation | Small proportion; imputation introduces minimal bias |
| `education` | 1,731 (4.2%) | Retain as category | "Unknown" education may correlate with specific demographics |
| `default` | 8,597 (20.9%) | Retain as category | High proportion; "unknown" default status is itself informative — banks may lack credit history for certain client segments |

**Outlier handling rationale:**

Percentile clipping (rather than row removal) was chosen for three reasons:
1. Tree-based models (RF, XGBoost, LightGBM) split on rank order and are inherently robust to outliers; clipping preserves rank.
2. Removing rows would disproportionately discard minority-class samples in an already imbalanced dataset (11.3% positive).
3. Logistic regression benefits from bounded feature ranges, as extreme values exert disproportionate influence on coefficient estimation.

---

### 3. split.py — Stratified Data Partitioning

**Purpose:** Split the cleaned dataset into train, validation, and test sets while preserving the target class ratio across all partitions.

| Function | Description |
|---|---|
| `stratified_split()` | Two-stage stratified split: first separates training data, then divides the remainder into validation and test sets. Default ratio is 70/15/15. |
| `save_splits()` | Writes all three splits as Parquet files to `data/processed/` and saves a training reference copy to `data/reference/` for downstream drift detection. |

**Split allocation:**

| Partition | Size | Purpose |
|---|---|---|
| Train (70%) | 28,831 rows | Model fitting and cross-validation |
| Validation (15%) | 6,178 rows | Threshold optimization — never seen during training |
| Test (15%) | 6,179 rows | Final, single-use evaluation — touched only once |

**Design decisions:**
- Stratification ensures all partitions maintain the 11.3% positive rate, which is critical for reliable metric estimation on imbalanced data.
- The validation set is used exclusively for cost-optimal threshold search. This prevents threshold overfitting to the test set.
- Parquet format is used for storage efficiency and schema preservation (column types survive round-trips, unlike CSV).
- A fixed random seed (`SEED = 42`) ensures reproducible splits across pipeline runs.

---

### 4. features.py — Feature Engineering and Preprocessing

**Purpose:** Transform raw features into model-ready representations through domain-driven engineering, encoding, and scaling. This module defines the full preprocessing pipeline that sits upstream of every classifier.

**Pipeline architecture (assembled by `build_pipeline()`):**

```
Raw DataFrame
    |
    v
PdaysTransformer          -- Sentinel handling: 999 --> binary flag + log transform
    |
    v
NonLinearBinningTransformer  -- Domain-driven bins for age, campaign, euribor3m
    |
    v
ColumnTransformer
    |-- num: StandardScaler        (10 numeric features)
    |-- ord: OrdinalEncoder        (education -- natural order)
    |-- nom: OneHotEncoder         (9 categorical + 3 bin features)
    |
    v
Classifier
```

**Custom transformers:**

#### PdaysTransformer

Addresses the `pdays` column, which uses 999 as a sentinel value meaning "client was never previously contacted." Raw usage would treat 999 as a meaningful numeric distance, distorting model behavior.

| Input | Output Features | Logic |
|---|---|---|
| `pdays` | `was_previously_contacted` | Binary: 1 if pdays != 999, else 0 |
| `pdays` | `pdays_log` | `log1p(pdays)` if contacted, else 0.0 |

The original `pdays` column is dropped after transformation.

#### NonLinearBinningTransformer

Adds categorical bin columns for features with empirically observed non-linear relationships to the target. The original numeric columns are retained (trees benefit from raw values); bins provide additional signal for linear models.

| Feature | Bins | Labels | Observed Pattern |
|---|---|---|---|
| `age` | 0-30, 30-45, 45-60, 60+ | young, prime, middle, senior | U-shaped: students and retirees subscribe at higher rates than working-age adults |
| `campaign` | 0-2, 2-5, 5+ | low, moderate, high | Diminishing returns: conversion drops sharply after 3-5 contact attempts |
| `euribor3m` | <1.5, 1.5-3.5, 3.5+ | low_rate, mid_rate, high_rate | Regime-dependent: fundamentally different economic environments produce different subscription behaviors |

Bin boundaries are domain-driven (not arbitrary quantiles) and were validated against the non-linear relationship visualization in the EDA step (figure `08b`).

**Encoding strategy:**

| Feature Type | Method | Rationale |
|---|---|---|
| Numeric (10 features) | StandardScaler | Required for logistic regression convergence; harmless for tree models |
| Education (1 feature) | OrdinalEncoder | Natural ordering exists (illiterate < basic.4y < ... < university.degree); ordinal encoding preserves this |
| All other categoricals (9 + 3 bins) | OneHotEncoder (drop="if_binary") | No natural ordering; binary features drop one level to avoid multicollinearity |

**Feature inventory after transformation: 59 total features.**

---

### 5. train.py — Model Training and Hyperparameter Optimization

**Purpose:** Define model configurations, run Bayesian hyperparameter optimization via Optuna, and train final models on the full training set.

**Supported models:**

| Model | Class | Key Configuration | Why Included |
|---|---|---|---|
| Logistic Regression | `LogisticRegression` | L1/L2 penalty, SAGA solver, balanced class weights | Interpretable baseline; odds ratio extraction; fast inference |
| Random Forest | `RandomForestClassifier` | Balanced class weights, parallelized | Ensemble of decorrelated trees; robust to hyperparameter choices; built-in feature importance |
| XGBoost | `XGBClassifier` | `scale_pos_weight=7.9`, AUC-PR eval metric | Gradient boosting typically achieves best performance on tabular data; explicit imbalance handling |

LightGBM is handled separately in `run_pipeline.py` due to its optional dependency status.

**Optuna search spaces:**

| Model | Hyperparameters Searched | Fixed Parameters |
|---|---|---|
| Logistic Regression | `C` (1e-4 to 10, log scale), `penalty` (L1/L2) | solver=saga, max_iter=2000, class_weight=balanced |
| Random Forest | `n_estimators` (100-600), `max_depth` (4-18), `min_samples_split` (5-40), `min_samples_leaf` (2-15) | class_weight=balanced, n_jobs=-1 |
| XGBoost | `n_estimators` (100-800), `max_depth` (3-9), `learning_rate` (0.01-0.3), `subsample` (0.6-1.0), `colsample_bytree` (0.6-1.0), `reg_alpha`, `reg_lambda` | scale_pos_weight=7.9, eval_metric=aucpr |

**Optimization protocol:**
- Objective: maximize 5-fold stratified cross-validated AUC-ROC.
- Default trial budget: 30 trials per model (configurable via `--n-trials`).
- Each trial builds the complete pipeline (preprocessing + classifier) to ensure hyperparameters are evaluated in the context of the full feature transformation.

| Function | Description |
|---|---|
| `tune_model()` | Runs Optuna study for a named model. Returns best parameters and best CV AUC. |
| `train_final_model()` | Fits a pipeline with given parameters on the full training set. Returns the fitted pipeline object. |

**Class imbalance handling:**

SMOTE and other synthetic oversampling methods were deliberately avoided. Instead, native class weighting is used:
- Logistic Regression and Random Forest: `class_weight='balanced'` adjusts loss contributions inversely proportional to class frequency.
- XGBoost: `scale_pos_weight=7.9` (ratio of negative to positive samples) amplifies the gradient contribution of minority-class errors.

This approach avoids the known risks of synthetic data generation: inflated cross-validation scores, introduction of artifacts in feature space, and potential information leakage when applied before cross-validation splits.

---

### 6. evaluate.py — Metrics, Threshold Optimization, and Model Selection

**Purpose:** Compute evaluation metrics, optimize decision thresholds against business economics, and select the best model using composite weighted scoring.

#### Banking Economics Framework

All cost parameters are derived from banking fundamentals rather than arbitrary assumptions:

```
Average term deposit:           $10,000
Bank pays depositor (interest): 2.0%
Bank lends funds at:            8.4%
Net Interest Margin (NIM):      6.4%  (8.4% - 2.0%)
Annual NIM revenue:             $640  ($10,000 x 6.4%)
Lifetime revenue (2-year term): $1,280

Conservative FN cost:           $200  (adjusted for conversion uncertainty and churn)
FP cost (wasted call):          $5    (agent time + telephony)
TP net value:                   $195  ($200 lifetime value - $5 call cost)
TN:                             $0    (correctly skipped, no action taken)

Cost asymmetry ratio (FN:FP):   40:1
```

The 40:1 cost asymmetry is the fundamental driver behind threshold optimization: missing a potential subscriber costs 40 times more than wasting a phone call. This justifies models that favor recall over precision.

#### Threshold Optimization

`find_optimal_threshold()` searches 200 evenly spaced thresholds in [0.05, 0.95] on the **validation set** (never the test set) and selects the threshold that maximizes:

```
Net Profit = TP x $195 - FP x $5 - FN x $200
```

This consistently produces thresholds between 0.05 and 0.20 — far below the default 0.50 — because the severe FN penalty incentivizes aggressive positive classification.

#### Composite Model Selection

Single-metric selection is fragile:

| Selection Criterion | Failure Mode |
|---|---|
| Profit only | Entirely dependent on cost assumptions; if the $200 FN estimate is wrong, the winner changes |
| AUC only | Threshold-agnostic; high AUC does not guarantee good performance at the operating threshold |
| Recall only | Trivially maximized by predicting all positive; destroys precision and business value |
| Calibration only | Well-calibrated probabilities do not imply good discrimination |

The composite score addresses these weaknesses by combining four normalized metrics with empirically justified weights:

| Component | Weight | Source Column | Direction | Rationale |
|---|---|---|---|---|
| Net Profit | 40% | `net_profit` | Higher is better | Primary business outcome |
| Recall | 25% | `test_recall` | Higher is better | FN is 40x costlier than FP |
| AUC-ROC | 20% | `test_roc_auc` | Higher is better | Overall discriminative ability |
| Calibration | 15% | `brier_score` | Lower is better (inverted) | Trustworthiness of probability estimates |

**Scoring procedure:**
1. Each metric is min-max normalized to [0, 1] across all candidate models.
2. Normalized scores are multiplied by their weights and summed.
3. Models are ranked by composite score; the highest score wins.

#### Sensitivity Analysis

`selection_sensitivity_analysis()` verifies robustness by checking whether the selected model changes under eight alternative selection criteria:

- Pure profit, pure AUC, pure recall, best calibration
- Composite (default weights)
- Profit-heavy profile (60/20/10/10)
- Recall-heavy profile (25/45/15/15)
- Equal-weight profile (25/25/25/25)

If the same model wins under all eight criteria, the selection is classified as **robust**. If multiple models win under different criteria, the composite balances the trade-offs and the analysis is reported for stakeholder transparency.

#### Recall Analysis

`recall_analysis()` generates a threshold trade-off table showing, for each threshold value, the number of subscribers caught, subscribers missed, calls made, and net profit. This table is designed for stakeholder conversations where the question is: *"How many potential subscribers are we willing to miss?"*

**Full function reference:**

| Function | Description |
|---|---|
| `compute_metrics()` | Returns accuracy, precision, recall, F1, AUC-ROC, PR-AUC, and Brier score |
| `find_optimal_threshold()` | Grid search over validation set probabilities to maximize net profit |
| `business_cost_analysis()` | Translates confusion matrix into dollar amounts with banking economics context |
| `composite_model_score()` | Computes weighted composite score, normalizes metrics, ranks models |
| `select_best_model()` | Wrapper that scores and returns the winning model name and scored DataFrame |
| `selection_sensitivity_analysis()` | Tests selection stability under 8 alternative criteria |
| `recall_analysis()` | Generates threshold vs. recall vs. profit trade-off table |
| `get_cost_derivation_text()` | Returns formatted string explaining cost assumptions for stakeholder documentation |

---

## Cross-Module Dependencies

```
ingest.py        (standalone -- no internal dependencies)
    |
clean.py         (standalone -- no internal dependencies)
    |
split.py         (standalone -- imports only sklearn and pandas)
    |
features.py      (standalone -- defines TARGET, transformers, pipeline builder)
    |
train.py         (imports features.build_pipeline)
    |
evaluate.py      (standalone -- defines cost constants, metrics, selection logic)
```

All modules are orchestrated by `scripts/run_pipeline.py`, which imports from each and executes the full pipeline in sequence. The `app/main.py` Streamlit dashboard imports only the serialized model artifacts (`.joblib` files) and the `features.py` module for custom transformer deserialization.

---

## Testing

Each module has corresponding test coverage in `tests/`:

| Module | Test File | Coverage Focus |
|---|---|---|
| `clean.py` | `test_clean.py` | Unknown imputation, duplicate removal, outlier clipping, duration drop |
| `features.py` | `test_features.py` | PdaysTransformer output shape, binning labels, pipeline end-to-end |
| Schema validation | `test_schemas.py` | Column presence, type enforcement, target encoding |

Tests use synthetic data fixtures defined in `tests/conftest.py` to avoid dependency on the real dataset during CI runs.

```bash
pytest tests/ -v
```

---

## Configuration Constants

Key constants that control pipeline behavior are defined at the module level rather than in external config files, ensuring that the pipeline is fully self-contained and reproducible without environment-specific configuration.

| Constant | Module | Value | Purpose |
|---|---|---|---|
| `TARGET` | `features.py` | `"y"` | Target column name |
| `SEED` | `split.py`, `train.py` | `42` | Reproducibility seed |
| `EXPECTED_SHAPE` | `ingest.py` | `(41188, 21)` | Schema validation gate |
| `COST_FP` | `evaluate.py` | `5` | Dollar cost of a false positive |
| `COST_FN` | `evaluate.py` | `200` | Dollar cost of a false negative |
| `VALUE_TP` | `evaluate.py` | `200` | Dollar value of a true positive |
| `SELECTION_WEIGHTS` | `evaluate.py` | `{profit: 0.40, recall: 0.25, auc: 0.20, calibration: 0.15}` | Composite scoring weights |
| `EDUCATION_ORDER` | `features.py` | 8 ordered levels | Ordinal encoding sequence |
| `OUTLIER_COLS` | `clean.py` | 9 numeric columns | Columns subject to percentile clipping |