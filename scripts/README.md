# scripts/

This folder contains CLI scripts for running the end-to-end pipeline and setting up the environment.

---

## Files

| File                | Description                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `run_pipeline.py`   | End-to-end ML pipeline orchestrator — runs all steps from data download to model selection          |
| `setup_env.py`      | Cross-platform environment setup helper — creates venv, installs dependencies, handles macOS OpenMP |
| `synthetic_data.py` | Generates synthetic customer data for testing and demonstration purposes                            |

---

## `run_pipeline.py`

The main orchestrator. Runs the full pipeline in a single command — downloads the dataset, cleans it, splits it, engineers features, trains and tunes four models, evaluates them, and saves all outputs.

### Usage

```bash
# Full pipeline (default: 50 Optuna trials)
python scripts/run_pipeline.py

# Specify number of tuning trials
python scripts/run_pipeline.py --n-trials 30

# Skip SHAP computation (faster iteration)
python scripts/run_pipeline.py --skip-shap

# Skip feature count sweep
python scripts/run_pipeline.py --skip-feature-sweep

# Ablation study — disable binning transformer
python scripts/run_pipeline.py --no-binning

# Combine flags
python scripts/run_pipeline.py --n-trials 20 --skip-shap --skip-feature-sweep
```

### Pipeline Steps

| Step | What happens                                                               |
| ---- | -------------------------------------------------------------------------- |
| 1–3  | Download raw data from UCI, clean it, split into train/val/test (70/15/15) |
| 5    | Rank features by importance using Random Forest on 5-fold CV               |
| 5c   | Train 4 models with default settings (pre-tuning baseline)                 |
| 6–7  | Tune each model with Optuna (30 trials, 5-fold CV), evaluate on test set   |
| 7b   | Before vs after tuning comparison                                          |

### Models

| Model               | Algorithm                  | Class imbalance handling           |
| ------------------- | -------------------------- | ---------------------------------- |
| Logistic Regression | Linear classifier          | `class_weight='balanced'`          |
| Random Forest       | Ensemble of decision trees | `class_weight='balanced'`          |
| XGBoost             | Gradient boosting          | `scale_pos_weight` (auto-computed) |
| KNN                 | K-Nearest Neighbours       | Threshold tuning                   |

### Outputs

| Output                | Location                        |
| --------------------- | ------------------------------- |
| All model pickles     | `models/*.pkl`                  |
| Best production model | `models/production/<model>.pkl` |
| Threshold config      | `models/threshold.json`         |
| Model manifest        | `models/models_manifest.json`   |
| Metric reports        | `reports/metrics/`              |
| Figures               | `reports/figures/`              |

---

## `setup_env.py`

Cross-platform setup helper that automates venv creation and dependency installation.

```bash
python scripts/setup_env.py              # interactive
python scripts/setup_env.py --yes        # non-interactive (auto-approve)
python scripts/setup_env.py --dev        # include dev extras (pytest, ruff, jupyter)
python scripts/setup_env.py --yes --dev  # non-interactive with dev extras
```

See [docs/setup.md](../docs/setup.md) for full setup instructions.

---

## `synthetic_data.py`

Generates synthetic customer records that mimic the UCI bank marketing schema. Useful for testing the dashboard or pipeline without the real dataset.
