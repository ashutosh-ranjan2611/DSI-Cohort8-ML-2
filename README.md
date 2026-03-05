<h1 align="center">Call Smarter: Predicting Term Deposit Subscribers</h1>
<p align="center"><b>Data Science Institute — Cohort 8 — ML Team 2</b></p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Numeric-NumPy-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Boost-XGBoost%20%7C%20LightGBM-2E8B57" />
  <img src="https://img.shields.io/badge/Tuning-Optuna-4B0082" />
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-E53935" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-8E44AD" />
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey" />
</p>

---

## Table of Contents

- [Project Overview and Purpose](#project-overview-and-purpose)
- [Business Problem](#business-problem)
- [Business Objectives and Goals](#business-objectives-and-goals)
- [Dataset Details](#dataset-details)
- [Risks and Unknowns](#risks-and-unknowns)
- [How We Approached the Analysis](#how-we-approached-the-analysis)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Local Setup](#local-setup)
- [Team Roles and Responsibilities](#team-roles-and-responsibilities)
- [Reflection Videos](#reflection-videos)
- [Acknowledgments](#acknowledgments)

---

## Project Overview and Purpose

This project tackles a real-world problem that banks face every day: figuring out which customers to call during a marketing campaign. Right now, banks call almost everyone on their list, which means most calls go to people who will never sign up. That wastes time, money, and burns out call centre staff.

We built a machine learning system that looks at a customer's profile — their age, job, whether they have loans, how the economy is doing — and predicts whether that person is likely to subscribe to a term deposit. Instead of calling everyone, the bank can now focus on the people most likely to say yes.

The entire system runs with a single command. It downloads the data, cleans it, trains multiple models, picks the best one, and generates a dashboard where non-technical stakeholders can see results, understand why the model makes its decisions, and even score new customers in real time.

---

## Business Problem

A bank runs phone-based marketing campaigns to sell term deposits (a type of savings product where the customer locks in their money for a fixed period at a guaranteed interest rate). The current process looks like this:

- The bank has a list of existing customers.
- Call centre agents work through the list, calling each person to pitch the term deposit.
- Out of every 100 calls, roughly 11 people subscribe. The other 89 calls are wasted.
- Each call costs the bank approximately $5 in agent time and phone charges.
- Each new subscriber generates approximately $200 in lifetime value through the bank's net interest margin (the difference between what the bank pays the depositor and what it earns by lending those funds out).

The core issue is **asymmetry**: missing a potential subscriber costs the bank 40 times more than making a wasted call. A single missed subscriber is $200 in lost revenue. A wasted call is just $5. This means the bank should lean toward calling more people rather than fewer — but it still needs to be smarter than calling everyone.

**Where these cost numbers come from:**

We did not pick $200 and $5 arbitrarily. We derived them from actual banking economics:

| Parameter                                   | Value   | Source                                                                                                |
| ------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| Average term deposit amount                 | $10,000 | Industry standard for retail banking                                                                  |
| Interest paid to depositor                  | 2.0%    | Typical fixed deposit rate                                                                            |
| Interest earned on lending                  | 8.4%    | Average retail lending rate                                                                           |
| Net Interest Margin                         | 6.4%    | 8.4% minus 2.0%                                                                                       |
| Annual NIM revenue per subscriber           | $640    | $10,000 x 6.4%                                                                                        |
| Lifetime revenue (2-year average term)      | $1,280  | $640 x 2 years                                                                                        |
| Conservative per-subscriber value (FN cost) | $200    | Adjusted down from $1,280 to account for conversion uncertainty, early withdrawal, and customer churn |
| Call cost (FP cost)                         | $5      | Agent time plus telephony infrastructure                                                              |
| Cost ratio (FN to FP)                       | 40:1    | $200 / $5                                                                                             |

---

## Business Objectives and Goals

We set out to answer four questions:

1. **Who should the bank call?** — Build a model that scores every customer by their likelihood to subscribe, so the marketing team can prioritize high-probability leads and skip the rest.

2. **How much money does this save?** — Translate model performance into actual dollar impact. We want to show the CFO a clear before-and-after comparison: here is what you spend now, and here is what you would spend with ML targeting.

3. **Why does the model make its predictions?** — Use SHAP (a model explanation method) to show which factors push a customer toward subscribing or not. This gives the marketing team actionable insight — not just "call this person" but "call this person because they are a retiree who was contacted via mobile during a low interest rate period."

4. **Can we trust the model's probabilities?** — A model that says "70% chance of subscribing" should be right about 70% of the time. We measure this through calibration analysis, so stakeholders know whether to take the numbers at face value.

**Who benefits from this system:**

| Stakeholder         | What They Get                                                |
| ------------------- | ------------------------------------------------------------ |
| Marketing Director  | A ranked list of who to call first, with reasons             |
| Call Centre Manager | Smaller, higher-quality call lists that reduce agent burnout |
| CFO                 | Measurable ROI — dollars saved per campaign cycle            |
| Compliance / Ethics | Transparent, explainable predictions (no black box)          |

---

## Dataset Details

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**Citation:** Moro et al., 2014. "A data-driven approach to predict the success of bank telemarketing." Decision Support Systems, Elsevier.

**License:** CC BY 4.0

| Property        | Value                                                        |
| --------------- | ------------------------------------------------------------ |
| Total records   | 41,188                                                       |
| Features        | 20 input + 1 target                                          |
| Target variable | `y` — did the client subscribe to a term deposit? (yes / no) |
| Positive rate   | 11.3% (heavily imbalanced)                                   |
| Time period     | May 2008 to November 2010                                    |
| Geography       | Portugal                                                     |
| File format     | CSV, semicolon-delimited, inside a nested ZIP archive        |

### Feature Categories

**Client attributes (7 features):**
These describe who the customer is. Age, job type, marital status, education level, whether they have credit in default, a housing loan, or a personal loan.

**Campaign attributes (7 features):**
These describe the current and past marketing interactions. How the customer was contacted (phone or mobile), which month and day of the week the call was made, how many times they have been called this campaign, how many days since the last contact, how many times they were contacted in previous campaigns, and what happened in the previous campaign (success, failure, or no previous campaign).

**Economic indicators (5 features):**
These capture the macroeconomic environment at the time of the call. Employment variation rate, consumer price index, consumer confidence index, the 3-month Euribor interest rate, and the number of employees in the economy. These turned out to be among the strongest predictors — when interest rates are low and the economy is uncertain, people are more receptive to safe savings products.

### The Duration Problem

There is one feature in this dataset that requires special attention: `duration`, which records how long each phone call lasted (in seconds). At first glance, it looks like a powerful predictor — and it is. Longer calls tend to result in subscriptions.

But here is the catch: you only know how long a call lasted **after the call has ended**. If we are trying to predict who to call **before** the campaign starts, call duration does not exist yet. Including it in a predictive model is data leakage — the model looks incredible on paper but would be useless in practice.

The UCI dataset documentation explicitly warns about this. We dropped `duration` from all production models. We do not use it anywhere in our predictions.

---

## Risks and Unknowns

We identified the following risks during our analysis and documented how we addressed each one:

### Data Quality Risks

| Risk                                  | What We Found                                                                                                               | How We Handled It                                                                                                                                                                                                                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Missing values coded as "unknown"** | Six columns contain "unknown" entries instead of NaN. The worst is `default` at 20.9%.                                      | For columns with few unknowns (`job`, `marital`, `housing`, `loan`), we imputed with the most common value. For `education` and `default`, we kept "unknown" as its own category because it may carry signal — a bank not knowing a customer's credit default status is itself informative.                |
| **Class imbalance**                   | Only 11.3% of customers subscribed. A model that predicts "no" for everyone would be 88.7% accurate but completely useless. | We used class weighting in all models (`class_weight='balanced'` for scikit-learn models, `scale_pos_weight` for XGBoost) so the model penalizes misses on the minority class more heavily. We avoided synthetic oversampling (SMOTE) because it can inflate cross-validation scores with artificial data. |
| **Outliers**                          | `campaign` (number of calls) has extreme values — some customers were called 40+ times. `previous` is heavily right-skewed. | We clip numeric features at the 1st and 99th percentiles. This reduces extreme value influence on logistic regression without removing rows (which would lose minority-class samples). Tree models are naturally robust to outliers.                                                                       |
| **Duplicate rows**                    | Checked for exact duplicate records.                                                                                        | Duplicates are detected and removed automatically during cleaning.                                                                                                                                                                                                                                         |
| **Multicollinearity**                 | `euribor3m`, `emp.var.rate`, and `nr.employed` are correlated above 0.9 — they all reflect the state of the economy.        | We report this but do not drop features. Tree models are unaffected by correlated inputs, and logistic regression uses L1 regularization which naturally zeroes out redundant features.                                                                                                                    |

### Modeling Risks

| Risk                                 | Impact                                                                                                                                                                                                                     | Mitigation                                                                                                                                                                                                                     |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Duration leakage**                 | Including call duration would inflate all metrics by 15-20 percentage points but produce a model that cannot be used in practice.                                                                                          | Dropped `duration` from all production models.                                                                                                                                                                                 |
| **Overfitting**                      | With 30 Optuna trials per model, there is a risk of overfitting to the validation set.                                                                                                                                     | We use 5-fold stratified cross-validation during tuning. The validation set is used only for threshold selection. The test set is touched exactly once for final reporting.                                                    |
| **Cost assumption sensitivity**      | Our profit calculations depend on the $200 FN and $5 FP estimates. If these are wrong, the "best" model might change.                                                                                                      | We run a sensitivity analysis that tests model selection under eight different criteria and weight profiles. If the same model wins across all of them, we have confidence the selection is robust.                            |
| **Economic regime change**           | The dataset covers 2008-2010 (during and after the financial crisis). Models trained on this period may not generalize to fundamentally different economic conditions.                                                     | We include macroeconomic features (Euribor rate, employment) so the model can adapt to changing conditions. However, we flag that retraining would be needed if deployed in a different economic era.                          |
| **Non-linear feature relationships** | Age has a U-shaped relationship with subscription (young students and older retirees subscribe more than middle-aged working adults). Campaign calls show diminishing returns. Standard linear models miss these patterns. | We added a custom binning transformer that creates categorical bins for age, campaign, and euribor3m based on domain knowledge. This helps logistic regression capture non-linear effects while being neutral for tree models. |

---

## How We Approached the Analysis

### Why These Models

We will use four individual models plus one ensemble:

| Model               | Role in the Pipeline               | Strength                                                                                                                 |
| ------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression | Interpretable baseline             | Coefficients map directly to odds ratios. Fast to train. Stakeholders can understand it without ML background.           |
| Random Forest       | Robust ensemble                    | Averages hundreds of decision trees. Resistant to overfitting. Handles mixed feature types well.                         |
| XGBoost             | High-performance gradient boosting | Typically the strongest performer on structured/tabular data. Explicit support for class imbalance.                      |
| LightGBM            | Alternative gradient boosting      | Faster training than XGBoost on large datasets. Leaf-wise tree growth often finds better splits.                         |
| Voting Ensemble     | Diversity-based combination        | Soft vote across logistic regression + best tree model + second tree model. Combines linear and non-linear perspectives. |

### Why Composite Scoring Instead of Just Picking the Highest Accuracy

Accuracy is misleading on imbalanced data — a model that always predicts "no" gets 88.7% accuracy but catches zero subscribers. We also avoid picking the model with the highest single metric (like profit or AUC) because:

- Profit depends on our cost assumptions. If those assumptions shift, the winner could change.
- AUC measures overall ranking quality but ignores the specific threshold we use in production.
- Recall can be trivially maximized by predicting everyone as positive.

Instead, we combine four metrics into a weighted composite score. The weights reflect what matters most to the business: profit is weighted highest (40%), followed by recall (25%), AUC (20%), and calibration quality (15%). We then verify the selection is stable by testing it under eight different weighting schemes.

---

## Key Results

### Model Performance (Test Set — No Duration) - _place holder to be updated in future_

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| **Random Forest**   |          |           |        |          |
| Logistic Regression |          |           |        |          |
| XGBoost             |          |           |        |          |
| LightGBM            |          |           |        |          |
| Voting Ensemble     |          |           |        |          |

_Exact values will be generated when the pipeline runs._

---

## Pipeline Architecture

The entire analysis will be automated through a single Python script that will execute the following steps in order:

> Click the button below to open the **interactive version** — supports mouse scroll to zoom, drag to pan, and zoom buttons.

[![Open Interactive Pipeline View](https://img.shields.io/badge/%F0%9F%94%8D%20Open%20Interactive%20Pipeline%20View-Click%20to%20Zoom%20%26%20Pan-4A90D9?style=for-the-badge&logo=githubpages&logoColor=white)](docs/pipeline-flow.html)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "primaryTextColor": "#1a1a2e", "lineColor": "#555"}} }%%
flowchart TD
    subgraph DATA["  Data Layer  "]
        RAW["**data/raw/**\nbank-additional-full.csv\n41,188 rows · 21 columns"]
        PROC["**data/processed/**\ntrain / val / test  Parquet"]
        REF["**data/reference/**\ndrift detection baseline"]
    end

    subgraph SRC["  src/  —  Core ML Pipeline  "]
        ING["**ingest.py**\nDownload and Validate UCI ZIP"]
        CLN["**clean.py**\nDedup · Clip Outliers · Impute Unknowns\nDrop duration in production"]
        SPL["**split.py**\n70 / 15 / 15 Stratified Split  ·  Seed 42"]
        FEA["**features.py**\nPdaysTransformer\nNonLinearBinningTransformer\nColumnTransformer"]
        TRN["**train.py**\nLR · RF · XGBoost · LightGBM\nOptuna Hyperparameter Tuning"]
        EVL["**evaluate.py**\nMetrics · Optimal Threshold\nBusiness Cost Analysis"]
    end

    subgraph EXP["  experiments/  —  Jupyter Notebooks  "]
        N01["**01_eda.ipynb**\nExploratory Data Analysis"]
        N02["**02_feature_engineering.ipynb**\nStep-by-Step Pipeline Walkthrough"]
        N03["**03_model_comparison.ipynb**\nTrain All Models and Compare"]
        N04["**04_shap_analysis.ipynb**\nSHAP Global and Local Explainability"]
        N05["**05_pipeline_visualization.ipynb**\nArchitecture and Flow Diagrams"]
    end

    subgraph OUT["  Outputs  "]
        MDLALL["**models/**\nAll compared models\nlogistic_regression.pkl\nrandom_forest.pkl\nxgboost.pkl\nmodels_manifest.json"]
        MDL["**models/production/**\nxgboost.pkl  best model\nxgboost_booster.bin"]
        THR["**models/production/threshold.json**\ncost-optimal threshold + metrics"]
        RPT["**reports/metrics/**\ncomparison · shap · recall · tuning"]
        FIG["**reports/figures/**\nall plots and charts"]
    end

    SCR["**scripts/run_pipeline.py**\nSingle-command End-to-End Orchestrator"]
    APP["**app/main.py**\nStreamlit Dashboard\n5 Stakeholder Tabs"]

    RAW --> ING
    ING --> CLN
    CLN --> SPL
    SPL --> PROC
    SPL --> REF
    PROC --> FEA
    FEA --> TRN
    TRN --> EVL
    EVL --> MDLALL
    EVL --> MDL
    EVL --> THR
    EVL --> RPT
    EVL --> FIG
    MDL --> APP
    THR --> APP
    FIG --> APP

    SCR -. orchestrates .-> ING
    SCR -. orchestrates .-> CLN
    SCR -. orchestrates .-> SPL
    SCR -. orchestrates .-> FEA
    SCR -. orchestrates .-> TRN
    SCR -. orchestrates .-> EVL

    N01 -. uses .-> RAW
    N03 -. produces .-> MDLALL
    N03 -. produces .-> MDL
    N04 -. uses .-> MDL

    style DATA fill:#EFF6FF,color:#1E3A8A,stroke:#93C5FD,stroke-width:2px
    style SRC  fill:#F0FDF4,color:#14532D,stroke:#86EFAC,stroke-width:2px
    style EXP  fill:#FFFBEB,color:#78350F,stroke:#FCD34D,stroke-width:2px
    style OUT  fill:#F5F3FF,color:#3B0764,stroke:#C4B5FD,stroke-width:2px

    classDef dataNode fill:#BFDBFE,color:#1E3A8A,stroke:#3B82F6,stroke-width:1.5px,font-weight:bold
    classDef srcNode  fill:#BBF7D0,color:#14532D,stroke:#22C55E,stroke-width:1.5px,font-weight:bold
    classDef expNode  fill:#FDE68A,color:#78350F,stroke:#F59E0B,stroke-width:1.5px,font-weight:bold
    classDef outNode  fill:#DDD6FE,color:#3B0764,stroke:#8B5CF6,stroke-width:1.5px,font-weight:bold
    classDef appNode  fill:#FECDD3,color:#881337,stroke:#F43F5E,stroke-width:2px,font-weight:bold
    classDef scrNode  fill:#E2E8F0,color:#1E293B,stroke:#64748B,stroke-width:2px,font-weight:bold

    class RAW,PROC,REF dataNode
    class ING,CLN,SPL,FEA,TRN,EVL srcNode
    class N01,N02,N03,N04,N05 expNode
    class MDLALL,MDL,THR,RPT,FIG outNode
    class APP appNode
    class SCR scrNode
```

### Folder Reference

```
DSI-Cohort8-ML-2/
|
|-- app/                   Streamlit App to test and visualize the final model in real time
|-- data/
|   |-- raw/               Source CSV from UCI (auto-downloaded)
|   |-- processed/         Train / val / test splits (Parquet)
|   |-- reference/         Training-time snapshot for drift detection
|
|-- docs/                  Interactive pipeline diagram (HTML)
|-- experiments/           Jupyter notebooks (EDA -> features -> models -> SHAP -> viz)
|-- mlruns/                MLflow experiment tracking artifacts
|-- models/
|   |-- logistic_regression.pkl   All compared models (pickle)
|   |-- random_forest.pkl
|   |-- xgboost.pkl
|   |-- models_manifest.json      Model listing + selection rationale
|   |-- production/
|       |-- xgboost.pkl           Best model (Optuna-tuned pipeline)
|       |-- xgboost_booster.bin   Native XGBoost booster
|       |-- threshold.json        Cost-optimal threshold + test metrics
|-- reports/
|   |-- figures/           All generated plots and charts
|   |-- metrics/           CSV / JSON metric outputs
|
|-- scripts/               run_pipeline.py  (end-to-end CLI runner)
|-- src/                   Core Python modules (ingest clean split features train evaluate)
|-- tests/                 pytest unit tests
|
|-- pyproject.toml
|-- requirements.txt
|-- README.md
```

---

## Local Setup

### Prerequisites

- Python 3.12
- uv

### Setup and Run

```bash
# 1. Clone the repository
git clone https://github.com/ashutosh-ranjan2611/DSI-Cohort8-ML-2.git
cd DSI-Cohort8-ML-2

# 2. Create a virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# 3. Install dependencies
uv sync --active
```

---

## Team Roles and Responsibilities

Our team of four split the work across the major components of the project. While everyone contributed to discussions and code reviews, each person took primary ownership of specific areas.

| Member              | Primary Responsibility                             | Deliverables                                                                                                                                                                                                                                                                               |
| ------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Neha Bondade**    | Data pipeline and cleaning                         | Building the data ingestion module that handles the nested ZIP extraction from UCI. Designing the cleaning strategy — the per-column unknown handling rules, outlier clipping approach, and data quality checks (duplicates, skewness, multicollinearity, cardinality).                    |
| **Suresh Jannu**    | Model training and tuning                          | Implementing the training pipeline with Optuna hyperparameter search for all four model types. Defining the search spaces, configuring class imbalance handling, and building the before-versus-after tuning comparison.                                                                   |
| **Pavan**           | Evaluation and explainability                      | Designing the composite model selection framework with the weighted scoring approach. Building the banking economics cost derivation, threshold optimization, sensitivity analysis, and recall trade-off analysis. Implementing SHAP explainability with business-friendly feature labels. |
| **Ashutosh Ranjan** | Integration, Dashboard, documentation, and testing | Will design and develop end-to-end final code for the project. Building the Streamlit dashboard with five stakeholder tabs (Executive Summary, Call Centre Operations, Model and Data Science, Predict Client, Batch Predict).                                                             |

### How We Collaborated

- All code changes went through pull requests on GitHub with at least one reviewer before merging.
- We held weekly standups to sync on progress and blockers.
- Major design decisions were discussed and agreed upon as a team before implementation.

---

## Reflection Videos

#### _place holder to be updated in future_

- [Neha Bondade — Reflection Video](https://youtube.com/placeholder)
- [Suresh Jannu — Reflection Video](https://youtube.com/placeholder)
- [Pavan — Reflection Video](https://youtube.com/placeholder)
- [Ashutosh Ranjan — Reflection Video](https://youtube.com/placeholder)

---

## Week 2 - WIP

#### _place holder to be updated in future_

---

## Acknowledgments

- **Dataset:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Original research:** Moro, S., Cortez, P., and Rita, P. (2014). "A data-driven approach to predict the success of bank telemarketing." Decision Support Systems, 62, 22-31.
- **Course:** Data Science Institute, University of Toronto — Cohort 8
- **Tools:** scikit-learn, XGBoost, LightGBM, Optuna, SHAP, Streamlit, Plotly

---

<p align="center"><i>Built by DSI Cohort 8 — ML Team 2</i></p>
