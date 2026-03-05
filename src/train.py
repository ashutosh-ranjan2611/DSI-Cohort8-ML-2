"""
Model training — 3 models, Optuna hyperparameter tuning, cross-validation.

Models: Logistic Regression (baseline), Random Forest, XGBoost.
"""

from __future__ import annotations

import logging

import mlflow
import optuna

# ── Suppress noisy MLflow warnings ──────────────────────────────────────────
# 1. sklearn pickle serialisation advisory — emitted via logging, not warnings
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
# 2. pip version resolution failure in conda.yaml generation
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
# 3. MLflow DB migration noise (only relevant on first run)
logging.getLogger("mlflow.store.db.utils").setLevel(logging.ERROR)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.features import build_pipeline

logger = logging.getLogger(__name__)

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def _make_linear_svm(**kwargs) -> CalibratedClassifierCV:
    """Wrap LinearSVC in CalibratedClassifierCV for probability outputs.

    LinearSVC is O(n) vs RBF-SVC's O(n^2–n^3), so it scales to the full
    training set without subsampling. Isotonic regression calibration
    (cv=5) gives well-calibrated probabilities.
    """
    return CalibratedClassifierCV(LinearSVC(**kwargs), cv=5, method="isotonic")


MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "svm": _make_linear_svm,   # LinearSVC + isotonic calibration
    "knn": KNeighborsClassifier,
}

# Fixed params that Optuna doesn't search over
FIXED_PARAMS = {
    "logistic_regression": {
        "solver": "saga",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": SEED,
    },
    "random_forest": {"class_weight": "balanced", "random_state": SEED, "n_jobs": -1},
    "xgboost": {
        "scale_pos_weight": 7.9,
        "eval_metric": "aucpr",
        "random_state": SEED,
        "n_jobs": -1,
    },
    "svm": {
        # LinearSVC fixed params (no kernel/probability needed)
        "class_weight": "balanced",
        "random_state": SEED,
        "max_iter": 5000,
    },
    "knn": {
        "n_jobs": -1,
        # SMOTE-style imbalance handling not built in — rely on
        # recall-tuned threshold selection at inference time
    },
}


def _search_space(trial: optuna.Trial, name: str) -> dict:
    """Define Optuna search space per model."""
    if name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    elif name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
        }
    elif name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    elif name == "svm":
        # LinearSVC — l1 requires dual=False, l2 can use dual=True (faster)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "penalty": penalty,
            "dual": penalty == "l2",  # l1 requires dual=False
        }
    elif name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        }
    raise ValueError(f"Unknown model: {name}")


def tune_model(name: str, X_train, y_train, n_trials: int = 30) -> dict:
    """
    Run Optuna optimization for a given model.

    Returns dict with best_params, best_auc.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        search_params = _search_space(trial, name)
        all_params = {**search_params, **FIXED_PARAMS[name]}
        model = MODEL_CLASSES[name](**all_params)
        pipeline = build_pipeline(model)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {**study.best_params, **FIXED_PARAMS[name]}
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
