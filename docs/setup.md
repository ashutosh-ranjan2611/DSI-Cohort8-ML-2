# Local Setup & App Execution Guide

**Project:** Call Smarter — Predicting Term Deposit Subscribers
**Repo:** [ashutosh-ranjan2611/DSI-Cohort8-ML-2](https://github.com/ashutosh-ranjan2611/DSI-Cohort8-ML-2)

---

## Prerequisites

| Requirement | Version | Notes                                                                                                                              |
| ----------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Python      | 3.12    | Required. 3.12                                                                                                                     |
| uv          | latest  | Fast package manager. Install via `pip install uv` or [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| Git         | any     | For cloning the repository                                                                                                         |

---

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ashutosh-ranjan2611/DSI-Cohort8-ML-2.git
   cd DSI-Cohort8-ML-2
   ```

---

2. **Create and activate a virtual environment:**

   **NOTE: Be careful to create this uv in same location as other uv from earlier in your system**

   ```bash
   # Create virtual environment with Python 3.12
   uv venv .venv --python 3.12

   # macOS / Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\activate

   # Activate — Windows (Command Prompt)
   .venv\Scripts\activate.bat
   ```

---

3. **Install dependencies:**

   ```bash
   # Install all project dependencies from pyproject.toml
   uv sync --active

   # Alternatively, using pip
   pip install -r requirements.txt
   ```

---

4. **Run the end-to-end pipeline:**

   The pipeline script handles everything in one command: downloads the dataset, cleans it,
   splits it, engineers features, trains and tunes all models, evaluates them, selects the
   best, and saves all models and reports.

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

   **What the pipeline produces:**

   | Output              | Location                           | Description                                   |
   | ------------------- | ---------------------------------- | --------------------------------------------- |
   | All compared models | `models/*.pkl`                     | LR, RF, XGBoost as pickle files               |
   | Best model          | `models/production/xgboost.pkl`    | Optuna-tuned best pipeline                    |
   | Threshold config    | `models/production/threshold.json` | Cost-optimal threshold + test metrics         |
   | Model manifest      | `models/models_manifest.json`      | Model listing and selection rationale         |
   | Metric reports      | `reports/metrics/`                 | comparison, shap, recall, tuning CSVs + JSONs |
   | Figures             | `reports/figures/`                 | All matplotlib/seaborn plots                  |

---

5. **Launch the Streamlit dashboard:**

   The dashboard requires the pipeline to have run first (needs `models/production/xgboost.pkl`
   and `reports/metrics/comparison.json`).

   ```bash
   # Start the dashboard
   streamlit run app/main.py

   # Specify a custom port
   streamlit run app/main.py --server.port 8502

   # Disable browser auto-open
   streamlit run app/main.py --server.headless true
   ```

   The app will open at **http://localhost:8501** in your default browser.

   **Dashboard tabs:**

   | Tab            | Audience        | Content                                                   |
   | -------------- | --------------- | --------------------------------------------------------- |
   | Predict Client | Analysts        | Single-client YES/NO prediction with SHAP explanation     |
   | Batch Predict  | Operations team | File upload or URL bulk scoring with downloadable results |

---

6. **Run the Jupyter Notebooks (Optional)**

   Notebooks are for exploration and can be run independently after completing steps 1–3.
   Run them in order for a complete walkthrough:

   ```bash
   # Launch Jupyter in the experiments/ folder
   jupyter lab experiments/

   # Or open in VS Code — just click the .ipynb file
   ```

   | Notebook                          | Purpose                                         |
   | --------------------------------- | ----------------------------------------------- |
   | `01_eda.ipynb`                    | Exploratory data analysis                       |
   | `02_feature_engineering.ipynb`    | Step-by-step pipeline walkthrough               |
   | `03_model_comparison.ipynb`       | Train all models, Optuna tuning, save models    |
   | `04_shap_analysis.ipynb`          | SHAP explainability (requires `03` to have run) |
   | `05_pipeline_visualization.ipynb` | Pipeline architecture diagrams                  |

---

7. **Run Tests**

   ```bash
   # Run all tests
   pytest tests/ -v --tb=short

   # Run with verbose output
   pytest tests/ -v

   # Run a specific test file
   pytest tests/test_clean.py -v
   pytest tests/test_features.py -v
   pytest tests/test_train.py -v
   pytest tests/test_evaluate.py -v

   # With coverage:
   pytest tests/ --cov=src --cov-report=term-missing
   ```

---

## Interactive Pipeline Diagram

For a zoomable, scrollable view of the full pipeline architecture, open in your browser:

```
docs/pipeline-flow.html
```

Or view the static diagram directly in [README.md](../README.md#pipeline-architecture).
