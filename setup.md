# 🛠️ Project Setup Guide

**DSI-Cohort8-ML-2 — Bank Marketing ML Pipeline v4**

This guide covers environment setup, dependency installation, pipeline execution, and dashboard launch.

---

## Prerequisites

| Requirement | Version | Check Command |
|:---|:---|:---|
| Python | 3.11+ (3.12 recommended) | `python --version` |
| pip or uv | Latest | `pip --version` / `uv --version` |
| Git | Any | `git --version` |
| OS | macOS, Linux, or Windows | — |

> **Note:** ~4GB free disk space recommended (models, data, figures).

---

## 1. Clone the Repository

```bash
git clone https://github.com/ashutosh-ranjan2611/DSI-Cohort8-ML-2.git
cd DSI-Cohort8-ML-2
```

---

## 2. Environment Setup

### Option A: Using `uv` (Recommended — Faster)

```bash
# Install uv if you don't have it
pip install uv

# Create virtual environment
uv venv .venv --python 3.12

# Activate
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install all dependencies from pyproject.toml
uv sync --active

# Verify
uv pip list
```

### Option B: Using `pip`

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install in editable mode
pip install -e .

# Verify
pip list
```

---

## 3. Run the Pipeline

### Full Pipeline (Recommended First Run)

```bash
python scripts/run_pipeline.py
```

This single command will:

1. **Download** the UCI Bank Marketing dataset (if not present)
2. **Clean** data — handle unknowns, drop duration (leakage prevention)
3. **Split** — stratified 70/15/15 train/val/test
4. **Generate** 10+ EDA figures (stakeholder-friendly)
5. **Compute** feature importance (CV-averaged, leakage-free)
6. **Sweep** feature counts (2 to all features, 3 models)
7. **Tune** 4 models with Optuna (30 trials each by default)
8. **Build** diverse voting ensemble
9. **Select** best model via composite scoring (profit + recall + AUC + calibration)
10. **Run** sensitivity analysis (robustness check)
11. **Generate** evaluation figures (ROC, PR, confusion, calibration, composite)
12. **Compute** SHAP explanations (summary, bar, waterfall, business-friendly)
13. **Save** all models, metrics, and figures

**Expected runtime:** ~15-25 minutes (depends on hardware and `--n-trials`).

### Pipeline Options

```bash
# Quick run — fewer Optuna trials (good for testing setup)
python scripts/run_pipeline.py --n-trials 10

# Skip slow steps
python scripts/run_pipeline.py --skip-shap                # Skip SHAP (~5 min saved)
python scripts/run_pipeline.py --skip-feature-sweep        # Skip sweep (~3 min saved)
python scripts/run_pipeline.py --skip-shap --skip-feature-sweep  # Both

# Ablation study — disable non-linear binning
python scripts/run_pipeline.py --no-binning

# Combine everything for fastest possible run
python scripts/run_pipeline.py --n-trials 5 --skip-shap --skip-feature-sweep
```

---

## 4. Launch the Dashboard

```bash
streamlit run app/main.py
```

Opens at **http://localhost:8501** with 5 stakeholder tabs:

| Tab | Content |
|:---|:---|
| Executive Summary | ROI, banking economics, strategy comparison |
| Call Centre Ops | Segments, priority rules, recall trade-offs |
| Model & Data Science | SHAP, calibration, full model comparison |
| Predict Client | Single YES/NO prediction with factors |
| Batch Predict | File upload / URL bulk scoring |

> **Note:** The dashboard loads only the composite-selected best model for fast startup.

---

## 5. Run Tests

```bash
pytest tests/ -v
```

Tests use synthetic data fixtures (no real data needed):

| Test File | What It Tests |
|:---|:---|
| `test_clean.py` | Unknown imputation, duration removal |
| `test_features.py` | PdaysTransformer, NonLinearBinning, pipeline output shapes |
| `test_schemas.py` | Data schema validation, column expectations |

---

## 6. Interactive Notebooks

For exploration and development:

```bash
jupyter notebook experiments/
```

Run in order: `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_model_training.ipynb` → `04_shap_explainability.ipynb`

---

## 7. What Gets Created

After a full pipeline run, your directory will contain:

```
models/
├── logistic_regression.joblib      # Trained pipelines
├── random_forest.joblib
├── xgboost.joblib
├── lightgbm.joblib
├── voting_ensemble.joblib          # Dict: {preprocess, voter}
└── threshold.json                  # Best model, threshold, composite weights

reports/figures/                     # 18+ auto-generated PNG figures
reports/metrics/
├── comparison.json                 # All models with composite scores & ranks
├── comparison.csv
├── recall_analysis.csv             # Threshold vs recall trade-off table
├── feature_count_sweep.csv
└── shap_importance.json

data/
├── raw/bank-additional-full.csv    # Auto-downloaded (41,188 rows)
└── processed/
    ├── train.parquet               # 70% — training
    ├── val.parquet                 # 15% — threshold tuning
    └── test.parquet                # 15% — final evaluation (touched once)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

The pipeline and dashboard add the project root to `sys.path` automatically. If running scripts from a different directory, ensure you're in the project root:

```bash
cd DSI-Cohort8-ML-2
python scripts/run_pipeline.py
```

### `LightGBM not found`

LightGBM is optional. The pipeline will skip it and use LR + RF + XGB + Ensemble. To install:

```bash
pip install lightgbm
# or
uv pip install lightgbm
```

### FutureWarning spam in terminal

Suppressed by default in v4. If you see warnings, they don't affect results.

### Streamlit `PdaysTransformer` import error

The dashboard adds the project root to `sys.path`. If you still get errors:

```bash
# Run from project root
cd DSI-Cohort8-ML-2
streamlit run app/main.py
```

### Dataset download fails

If UCI servers are slow, manually download from [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), extract `bank-additional-full.csv`, and place it in `data/raw/`.

---

## Useful Commands Reference

```bash
# Deactivate environment
deactivate

# Update dependencies (uv)
uv sync --active

# Add a new package (uv)
uv pip install <package-name>

# Regenerate figures only (skip training)
# → Run pipeline with pre-trained models by re-running SHAP/EDA steps in notebooks

# Check model metadata
cat models/threshold.json | python -m json.tool
```

---

## Version History

| Version | Date | Changes |
|:---|:---|:---|
| v1 | Feb 2025 | Initial pipeline — 3 models, basic threshold |
| v2 | Feb 2025 | Added 4 models, feature sweep, color logging |
| v3 | Mar 2025 | Removed underperformers, diverse ensemble, Brier tracking |
| **v4** | **Mar 2025** | **Composite selection, non-linear binning, banking economics, 5-tab dashboard** |

---

For any issues, contact the repository owner or file a GitHub issue.