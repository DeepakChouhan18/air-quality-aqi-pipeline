"""
Model training for next-day AQI prediction.

CHRONOLOGICAL SPLIT ONLY — no random shuffling.
WHY: Air quality is a time series. If we randomly split, tomorrow's data
ends up in the training set and today's in the test set, which leaks
future information and produces falsely optimistic metrics. Chronological
split correctly simulates what happens in production: you train on past
data and evaluate on future data you've never seen.

Split ratios (from config): 70% train / 15% validation / 15% test.

BASELINES:
We always compare ML models against naive baselines. If your ML model
doesn't beat "just use today's AQI as tomorrow's prediction", the
complexity isn't justified. Baselines here:
  - Persistence: y_hat[t+1] = y[t]      (tomorrow = today)
  - 7-day MA:    y_hat[t+1] = mean(y[t-6..t]) (rolling 7-day average)

HYPERPARAMETER TUNING:
Uses TimeSeriesSplit on the training fold ONLY — never touches validation
or test data during tuning. This avoids look-ahead bias in tuning.

FIXED RANDOM SEED: RANDOM_SEED=42 from config — ensures every run
produces the same model if data hasn't changed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.ml.features import get_training_data, FEATURE_COLS, TARGET_REG, TARGET_CLF
from src.ml.evaluate import (
    evaluate_regression, evaluate_classification,
    persistence_baseline, moving_average_baseline,
    save_comparison_table, save_regression_plots, save_classification_plots,
)
from src.config import (
    TARGET_CITY, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_REGRESSION,
    MLFLOW_EXPERIMENT_CLASSIFICATION,
    MLFLOW_REGISTERED_REGRESSOR, MLFLOW_REGISTERED_CLASSIFIER,
    REPORTS_DIR,
)

# Try to import XGBoost/LightGBM; fall back to GradientBoosting if not installed
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[train] XGBoost not found — using GradientBoostingRegressor as fallback.")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-sorted DataFrame chronologically into train/val/test.

    Args:
        df: DataFrame sorted by date (ascending)
        train_ratio: fraction for training (default 0.70)
        val_ratio:   fraction for validation (default 0.15)
        (test = 1 - train_ratio - val_ratio = 0.15)

    Returns:
        (train_df, val_df, test_df)  — non-overlapping, chronological order
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    print(
        f"[train] Chronological split: "
        f"train={len(train)} rows ({train['date'].min().date()}→{train['date'].max().date()}), "
        f"val={len(val)} rows ({val['date'].min().date()}→{val['date'].max().date()}), "
        f"test={len(test)} rows ({test['date'].min().date()}→{test['date'].max().date()})"
    )
    return train, val, test


def make_imputer_pipeline(estimator):
    """
    Wrap any estimator in a Pipeline with a median imputer.

    WHY: Lag features at the boundary of the training window may be NaN
    (the first few days after the train/val split edge). SimpleImputer
    with median fills these without data leakage because it's fit only
    on the training set inside the pipeline.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def tune_with_tscv(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    scoring: str = "neg_root_mean_squared_error",
) -> Pipeline:
    """
    Hyperparameter tuning with TimeSeriesSplit on the training fold only.
    Returns the best estimator (already fitted on all of X_train).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=tscv,
        scoring=scoring,
        refit=True,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    print(f"[train]   Best params: {gs.best_params_}, CV score: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ── Regression ─────────────────────────────────────────────────────────────────

def build_regression_models() -> dict:
    """
    Return a dict of {model_name: (pipeline, param_grid)} for regression.
    All models wrapped in imputer pipeline for NaN handling.
    """
    models = {}

    # Ridge (linear baseline — interpretable, fast)
    models["Ridge"] = (
        make_imputer_pipeline(Ridge(random_state=RANDOM_SEED)),
        {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    )

    # Random Forest (handles non-linearity, gives feature importance)
    models["RandomForest"] = (
        make_imputer_pipeline(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        ),
        {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3]},
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            make_imputer_pipeline(
                XGBRegressor(
                    n_estimators=200, learning_rate=0.05,
                    random_state=RANDOM_SEED, verbosity=0,
                )
            ),
            {"model__max_depth": [3, 5], "model__n_estimators": [100, 200]},
        )
    elif LIGHTGBM_AVAILABLE:
        models["LightGBM"] = (
            make_imputer_pipeline(
                LGBMRegressor(n_estimators=200, random_state=RANDOM_SEED)
            ),
            {"model__num_leaves": [31, 63], "model__n_estimators": [100, 200]},
        )
    else:
        # Always have at least one gradient boosting model
        models["GradientBoosting"] = (
            make_imputer_pipeline(
                GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_SEED)
            ),
            {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]},
        )

    return models


def build_classification_models(classes: list) -> dict:
    """Return {model_name: (pipeline, param_grid)} for classification."""
    models = {}

    models["LogisticRegression"] = (
        make_imputer_pipeline(
            LogisticRegression(
                max_iter=500, random_state=RANDOM_SEED, class_weight="balanced"
            )
        ),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    )

    models["RandomForest"] = (
        make_imputer_pipeline(
            RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_SEED,
                class_weight="balanced", n_jobs=-1,
            )
        ),
        {"model__max_depth": [None, 5, 10]},
    )

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            make_imputer_pipeline(
                XGBClassifier(
                    n_estimators=200, learning_rate=0.05,
                    random_state=RANDOM_SEED, verbosity=0,
                    use_label_encoder=False, eval_metric="mlogloss",
                )
            ),
            {"model__max_depth": [3, 5], "model__n_estimators": [100, 200]},
        )
    elif LIGHTGBM_AVAILABLE:
        models["LightGBM"] = (
            make_imputer_pipeline(
                LGBMClassifier(n_estimators=200, random_state=RANDOM_SEED)
            ),
            {"model__num_leaves": [31, 63]},
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        models["GradientBoosting"] = (
            make_imputer_pipeline(
                GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED)
            ),
            {"model__max_depth": [3, 5]},
        )

    return models


# ── Main training function ────────────────────────────────────────────────────

def run_training(city: str = TARGET_CITY, log_to_mlflow: bool = True):
    """
    Full training pipeline:
    1. Load features (calls build_features → saves to Postgres + parquet)
    2. Chronological split
    3. Compute baselines
    4. Train + tune all regression models with TimeSeriesSplit CV
    5. Train + tune all classification models
    6. Evaluate on val and test sets
    7. Log everything to MLflow
    8. Save comparison tables and plots to reports/

    Returns dict with best models and their metrics.
    """
    import mlflow
    import mlflow.sklearn

    # ── Load and split data ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[train] Loading feature data …")
    df = get_training_data(city=city)
    train_df, val_df, test_df = chronological_split(df)

    X_train = train_df[FEATURE_COLS].values
    y_reg_train = train_df[TARGET_REG].values
    y_clf_train = train_df[TARGET_CLF].values

    X_val = val_df[FEATURE_COLS].values
    y_reg_val = val_df[TARGET_REG].values
    y_clf_val = val_df[TARGET_CLF].values

    X_test = test_df[FEATURE_COLS].values
    y_reg_test = test_df[TARGET_REG].values
    y_clf_test = test_df[TARGET_CLF].values

    # Encode AQI category labels to integers for XGBoost/LightGBM classifiers
    le = LabelEncoder()
    le.fit(y_clf_train)
    y_clf_train_enc = le.transform(y_clf_train)
    y_clf_val_enc = le.transform(y_clf_val)
    y_clf_test_enc = le.transform(y_clf_test)

    # ── Compute baselines ──────────────────────────────────────────────────
    print("\n[train] Computing baselines …")
    persist_val  = persistence_baseline(val_df)
    persist_test = persistence_baseline(test_df)
    ma7_val      = moving_average_baseline(val_df, train_df)
    ma7_test     = moving_average_baseline(test_df, pd.concat([train_df, val_df]))

    baselines = {
        "Persistence_val":  evaluate_regression(y_reg_val,  persist_val,  "Persistence-Val"),
        "Persistence_test": evaluate_regression(y_reg_test, persist_test, "Persistence-Test"),
        "MA7_val":          evaluate_regression(y_reg_val,  ma7_val,      "MA7-Val"),
        "MA7_test":         evaluate_regression(y_reg_test, ma7_test,     "MA7-Test"),
    }

    # ── MLflow setup ──────────────────────────────────────────────────────
    if log_to_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    all_results = {"regression": {}, "classification": {}}
    best_reg_model  = None
    best_reg_rmse   = float("inf")
    best_clf_model  = None
    best_clf_f1     = -float("inf")
    best_reg_name   = None
    best_clf_name   = None

    # ── Regression models ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[train] Training REGRESSION models …")
    reg_models = build_regression_models()

    for model_name, (pipeline, param_grid) in reg_models.items():
        print(f"\n[train] → {model_name}")
        if log_to_mlflow:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_REGRESSION)

        with (mlflow.start_run(run_name=model_name) if log_to_mlflow else _null_context()):
            # Tune on train split with TimeSeriesSplit
            best_pipeline = tune_with_tscv(
                pipeline, param_grid, X_train, y_reg_train,
                scoring="neg_root_mean_squared_error",
            )

            # Evaluate
            val_metrics  = evaluate_regression(y_reg_val,  best_pipeline.predict(X_val),  f"{model_name}-Val")
            test_metrics = evaluate_regression(y_reg_test, best_pipeline.predict(X_test), f"{model_name}-Test")

            if log_to_mlflow:
                _log_regression_run(
                    mlflow, best_pipeline, model_name, val_metrics, test_metrics,
                    train_df, test_df,
                )

            all_results["regression"][model_name] = {
                "val": val_metrics, "test": test_metrics, "model": best_pipeline,
            }

            if val_metrics["rmse"] < best_reg_rmse:
                best_reg_rmse  = val_metrics["rmse"]
                best_reg_model = best_pipeline
                best_reg_name  = model_name

    # ── Classification models ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("[train] Training CLASSIFICATION models …")
    clf_models = build_classification_models(list(le.classes_))

    for model_name, (pipeline, param_grid) in clf_models.items():
        print(f"\n[train] → {model_name}")
        if log_to_mlflow:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_CLASSIFICATION)

        with (mlflow.start_run(run_name=model_name) if log_to_mlflow else _null_context()):
            # For classifiers that need encoded labels (XGBoost)
            needs_encoding = ("XGBoost" in model_name or "LightGBM" in model_name)
            y_tr = y_clf_train_enc if needs_encoding else y_clf_train
            y_va = y_clf_val_enc   if needs_encoding else y_clf_val
            y_te = y_clf_test_enc  if needs_encoding else y_clf_test

            best_pipeline = tune_with_tscv(
                pipeline, param_grid, X_train, y_tr,
                scoring="f1_macro",
            )

            preds_val  = best_pipeline.predict(X_val)
            preds_test = best_pipeline.predict(X_test)

            # Decode if needed
            if needs_encoding:
                preds_val  = le.inverse_transform(preds_val)
                preds_test = le.inverse_transform(preds_test)
                y_va = le.inverse_transform(y_va)
                y_te = le.inverse_transform(y_te)

            val_metrics  = evaluate_classification(y_va,  preds_val,  f"{model_name}-Val")
            test_metrics = evaluate_classification(y_te,  preds_test, f"{model_name}-Test")

            if log_to_mlflow:
                _log_classification_run(
                    mlflow, best_pipeline, model_name, val_metrics, test_metrics,
                    train_df, test_df, le,
                )

            all_results["classification"][model_name] = {
                "val": val_metrics, "test": test_metrics, "model": best_pipeline,
                "label_encoder": le,
            }

            if val_metrics["macro_f1"] > best_clf_f1:
                best_clf_f1   = val_metrics["macro_f1"]
                best_clf_model = best_pipeline
                best_clf_name  = model_name

    # ── Save comparison report ─────────────────────────────────────────────
    print("\n[train] Saving comparison tables and plots …")
    save_comparison_table(all_results, baselines)
    save_regression_plots(
        all_results["regression"],
        val_df, test_df,
        X_val, X_test,
        y_reg_val, y_reg_test,
    )
    save_classification_plots(
        all_results["classification"],
        X_val, X_test,
        y_clf_val, y_clf_test,
        le,
    )

    print(f"\n{'='*60}")
    print(f"[train] Best regression model:     {best_reg_name} (val RMSE={best_reg_rmse:.2f})")
    print(f"[train] Best classification model: {best_clf_name} (val F1={best_clf_f1:.3f})")
    print(f"[train] Reports → {REPORTS_DIR}/")

    return {
        "best_regressor":  best_reg_model,
        "best_classifier": best_clf_model,
        "best_reg_name":   best_reg_name,
        "best_clf_name":   best_clf_name,
        "label_encoder":   le,
        "all_results":     all_results,
        "splits": {
            "train": train_df, "val": val_df, "test": test_df,
        },
    }


# ── MLflow logging helpers ────────────────────────────────────────────────────

def _log_regression_run(mlflow, model, name, val_m, test_m, train_df, test_df):
    """Log params, metrics, and model artifact to MLflow."""
    import subprocess
    mlflow.log_param("model_name", name)
    mlflow.log_param("city", TARGET_CITY)
    mlflow.log_param("train_start", str(train_df["date"].min().date()))
    mlflow.log_param("train_end",   str(train_df["date"].max().date()))
    mlflow.log_param("test_start",  str(test_df["date"].min().date()))
    mlflow.log_param("test_end",    str(test_df["date"].max().date()))
    mlflow.log_param("features", ",".join(FEATURE_COLS))
    mlflow.log_param("random_seed", RANDOM_SEED)

    # Try to capture git commit for full reproducibility
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True
        ).stdout.decode().strip()
        mlflow.log_param("git_commit", commit)
    except Exception:
        pass

    for k, v in val_m.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"val_{k}", v)
    for k, v in test_m.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"test_{k}", v)

    mlflow.sklearn.log_model(model, artifact_path="model")


def _log_classification_run(mlflow, model, name, val_m, test_m, train_df, test_df, le):
    import subprocess
    mlflow.log_param("model_name", name)
    mlflow.log_param("city", TARGET_CITY)
    mlflow.log_param("train_start", str(train_df["date"].min().date()))
    mlflow.log_param("train_end",   str(train_df["date"].max().date()))
    mlflow.log_param("features", ",".join(FEATURE_COLS))
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("classes", ",".join(le.classes_))

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True
        ).stdout.decode().strip()
        mlflow.log_param("git_commit", commit)
    except Exception:
        pass

    for k, v in val_m.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"val_{k}", v)
    for k, v in test_m.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"test_{k}", v)

    mlflow.sklearn.log_model(model, artifact_path="model")


class _null_context:
    """Context manager that does nothing — used when MLflow is disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


if __name__ == "__main__":
    results = run_training(log_to_mlflow=True)
    print("\n[train] Training complete. Run 'make register' to promote best models.")
