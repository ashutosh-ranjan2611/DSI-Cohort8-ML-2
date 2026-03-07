# app/

This folder contains the Streamlit interactive dashboard for the **Call Smarter** project.

---

## File

| File      | Description                                            |
| --------- | ------------------------------------------------------ |
| `main.py` | Streamlit application — the full dashboard entry point |

---

## Overview

`main.py` is a single-file Streamlit app that loads the trained model pipeline and provides an interactive interface for two stakeholder workflows:

| Tab                | Audience        | What it does                                                                                  |
| ------------------ | --------------- | --------------------------------------------------------------------------------------------- |
| **Predict Client** | Analysts        | Enter a single customer's profile → get a YES/NO prediction with a SHAP waterfall explanation |
| **Batch Predict**  | Operations team | Upload a CSV/Excel file or paste a URL → bulk-score customers → download a ranked lead list   |

---

## How to Run

```bash
# From the repository root (with venv activated)
streamlit run app/main.py

# Custom port
streamlit run app/main.py --server.port 8502
```

The app opens at **http://localhost:8501** in your browser.

> **Prerequisite:** The pipeline must have been run first so that `models/production/` and `reports/metrics/comparison.json` exist. See [docs/setup.md](../docs/setup.md) for full setup instructions.

---

## Model Artifacts Required

| Artifact                          | Purpose                                     |
| --------------------------------- | ------------------------------------------- |
| `models/production/<model>.pkl`   | Trained scikit-learn pipeline (pickle)      |
| `models/threshold.json`           | Cost-optimal threshold + winning model name |
| `reports/metrics/comparison.json` | Model comparison metrics for the dashboard  |
| `reports/figures/`                | Pre-generated charts displayed in the app   |

---

## Key Dependencies

- `streamlit` — web app framework
- `pickle` — model loading
- `pandas` / `numpy` — data handling
- `plotly` — interactive charts
- `shap` — SHAP waterfall explanations
- `requests` — URL-based batch input
