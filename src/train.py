"""
Model training — 4 models with Optuna hyperparameter tuning and 5-fold cross-validation.

We compare these 4 classifiers:
  - Logistic Regression : the simple, explainable baseline (like a scorecard)
  - Random Forest       : hundreds of decision trees that vote together
  - XGBoost             : trees that learn from each other's mistakes one step at a time
  - KNN                 : "what did the most similar customers do?"

For each model, Optuna automatically finds the best settings (hyperparameters)
by trying different combinations and keeping the one with the highest AUC score.
"""

from __future__ import annotations

import logging

import mlflow
import optuna

# Suppress noisy MLflow log lines that clutter the terminal output
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.store.db.utils").setLevel(logging.ERROR)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.features import build_pipeline

logger = logging.getLogger(__name__)

# Fixed random seed so every run gives the same results
SEED = 42

# 5-fold cross-validation: split the training data into 5 chunks,
# train on 4 chunks and validate on the 5th, rotate, then average.
# This gives a reliable performance estimate without touching the test set.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# The 4 models we train and compare.
# Each one approaches the prediction problem in a different way.
MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,   # Linear model — fast and easy to explain
    "random_forest":       RandomForestClassifier, # Builds many trees and takes a majority vote
    "xgboost":             XGBClassifier,          # Builds trees sequentially, each fixing the last
    "knn":                 KNeighborsClassifier,   # Finds the K most similar customers and copies them
}

# Fixed settings that Optuna does NOT search over — these are design decisions
# we made once and keep constant across all experiments.
FIXED_PARAMS = {
    "logistic_regression": {
        "solver": "saga",       # Supports both L1 and L2 regularisation
        "max_iter": 2000,       # Allow plenty of iterations to converge
        "class_weight": "balanced",  # Automatically up-weights the minority class
        "random_state": SEED,
    },
    "random_forest": {
        "class_weight": "balanced",  # Same: minority class gets higher weight
        "random_state": SEED,
        "n_jobs": -1,  # Use all available CPU cores
    },
    "xgboost": {
        "eval_metric": "aucpr",  # Optimise precision-recall area during training
        "random_state": SEED,
        "n_jobs": -1,
    },
    "knn": {
        # KNN has no built-in class balancing — imbalance is handled through
        # threshold tuning at prediction time instead
        "n_jobs": -1,
    },
}


def _search_space(trial: optuna.Trial, name: str) -> dict:
    """
    Tell Optuna what values to try for each model's key settings.

    Optuna calls this once per trial, picks a combination, trains the model,
    scores it with cross-validation, and then tries a smarter combination next.
    After n_trials attempts it returns the settings that scored highest.
    """
    if name == "logistic_regression":
        return {
            # C = regularisation strength. Small C = simpler model, large C = fits harder.
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            # L1 penalty sets some weights to exactly zero (useful for feature selection).
            # L2 just shrinks all weights proportionally.
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    elif name == "random_forest":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=100),  # Number of trees
            "max_depth":        trial.suggest_int("max_depth", 4, 18),                   # How tall each tree grows
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 40),          # Min samples needed to split a node
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),            # Min samples allowed in a leaf
        }
    elif name == "xgboost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  # How much each tree contributes
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),                 # Fraction of rows used per tree
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),          # Fraction of features per tree
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),     # L1 regularisation
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),    # L2 regularisation
        }
    elif name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),  # How many neighbours to look at
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),   # Equal vote vs weighted by closeness
            "metric":      trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),  # How to measure "distance"
        }
    raise ValueError(f"Unknown model name: '{name}'")


def tune_model(name: str, X_train, y_train, n_trials: int = 30, cv_n_jobs: int = -1) -> dict:
    """
    Run Optuna optimization for a given model.

    Args:
        cv_n_jobs: Passed to cross_val_score. Use 1 in test environments to
                   avoid joblib multiprocessing issues on Windows.

    Returns dict with best_params, best_auc.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Compute class-imbalance ratio once so it can be used inside the objective
    _extra_fixed: dict = {}
    if name == "xgboost":
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        _extra_fixed["scale_pos_weight"] = round(neg / pos, 2)

    def objective(trial):
        search_params = _search_space(trial, name)
        all_params = {**search_params, **FIXED_PARAMS[name], **_extra_fixed}
        model = MODEL_CLASSES[name](**all_params)
        pipeline = build_pipeline(model)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=cv_n_jobs
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {**study.best_params, **FIXED_PARAMS[name], **_extra_fixed}
    logger.info(
        "%s → best AUC: %.4f | params: %s", name, study.best_value, study.best_params
    )

    # Log tuning result to MLflow (no-op when no active run)
    with mlflow.start_run(run_name=f"tune_{name}", nested=True):
        mlflow.log_metric(f"{name}_best_auc", study.best_value)
        mlflow.log_params({f"{name}_{k}": v for k, v in study.best_params.items()})

    return {"best_params": best_params, "best_auc": study.best_value, "study": study}


def train_final_model(name: str, params: dict, X_train, y_train):
    """Train a model with given params on full training set. Returns fitted pipeline."""
    model = MODEL_CLASSES[name](**params)
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    # Log final model metadata to MLflow (no-op when no active run)
    with mlflow.start_run(run_name=f"final_{name}", nested=True):
        mlflow.log_params({f"final_{k}": v for k, v in params.items() if not callable(v)})
        mlflow.sklearn.log_model(pipeline, name=f"model_{name}")

    return pipeline
