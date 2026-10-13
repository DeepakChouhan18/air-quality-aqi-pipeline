"""
Register the best regression and classification models from MLflow into
the Model Registry and promote them to the 'production' alias.

WHY A SEPARATE REGISTRATION SCRIPT:
Training and registration are two different decisions:
  - Training: find the best hyperparameters on this dataset.
  - Registration: decide whether this new model is good enough to replace
    the current production model.
This script enforces the second decision explicitly and logs the reason.

HOW IT WORKS:
1. Finds the best run in MLFLOW_EXPERIMENT_REGRESSION by lowest val_rmse.
2. Finds the best run in MLFLOW_EXPERIMENT_CLASSIFICATION by highest val_macro_f1.
3. Registers both under aqi_regressor / aqi_classifier in the Model Registry.
4. Sets the 'production' alias on the new version.
5. If a production model already exists, compares metrics first —
   only promotes if new model is strictly better on the validation metric.

Usage:
    python -m src.ml.register_best
    # or:
    make register
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_REGRESSION,
    MLFLOW_EXPERIMENT_CLASSIFICATION,
    MLFLOW_REGISTERED_REGRESSOR,
    MLFLOW_REGISTERED_CLASSIFIER,
    MLFLOW_PRODUCTION_ALIAS,
)


def get_best_run(client: MlflowClient, experiment_name: str, metric: str, mode: str = "min"):
    """
    Find the run with the best value of `metric` in a given experiment.

    Args:
        client:          MLflow client
        experiment_name: name of the experiment to search
        metric:          metric name (must have been logged with mlflow.log_metric)
        mode:            'min' for RMSE, 'max' for F1/accuracy

    Returns:
        The best mlflow.entities.Run, or None if no runs exist.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"[register] Experiment '{experiment_name}' not found. "
              "Run 'make train' first.")
        return None

    import math

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
    )
    valid_runs = [
        r for r in runs
        if metric in r.data.metrics and not math.isnan(r.data.metrics[metric])
    ]
    if not valid_runs:
        print(f"[register] No completed runs found in '{experiment_name}'.")
        return None

    if mode == "min":
        best = min(valid_runs, key=lambda r: r.data.metrics[metric])
    else:
        best = max(valid_runs, key=lambda r: r.data.metrics[metric])

    print(
        f"[register] Best run in '{experiment_name}': run_id={best.info.run_id}, "
        f"{metric}={best.data.metrics.get(metric, 'N/A'):.4f}"
    )
    return best


def should_promote(
    client: MlflowClient,
    registered_name: str,
    new_metric_value: float,
    metric_key: str,
    mode: str = "min",
) -> bool:
    """
    Compare the new model's metric against the currently production-aliased version.
    Returns True if no production model exists OR if new model is strictly better.

    WHY THIS GATE:
    Without this check, a re-run that uses less data or a different random
    seed could silently downgrade the production model. This ensures production
    only ever moves forward.
    """
    try:
        # Get the current production model version
        prod_versions = client.get_model_version_by_alias(
            registered_name, MLFLOW_PRODUCTION_ALIAS
        )
        prod_run = client.get_run(prod_versions.run_id)
        prod_metric = prod_run.data.metrics.get(metric_key, None)

        if prod_metric is None:
            print(f"[register] No metric '{metric_key}' on current production model — promoting.")
            return True

        if mode == "min":
            should = new_metric_value < prod_metric
        else:
            should = new_metric_value > prod_metric

        direction = "↓" if mode == "min" else "↑"
        print(
            f"[register] Production {metric_key}={prod_metric:.4f} | "
            f"New {metric_key}={new_metric_value:.4f} {direction} | "
            f"Promote: {should}"
        )
        return should

    except mlflow.exceptions.MlflowException:
        # No production alias exists yet — always promote first model
        print(f"[register] No existing production alias for '{registered_name}' — promoting.")
        return True


def register_and_promote(
    client: MlflowClient,
    run: mlflow.entities.Run,
    registered_name: str,
    metric_key: str,
    mode: str = "min",
) -> str:
    """
    Register the model from `run` in the Model Registry and set the
    'production' alias if it beats the current production model.

    Returns the new model version string, or None if not promoted.
    """
    if run is None:
        return None

    new_metric = run.data.metrics.get(metric_key, float("inf") if mode == "min" else -1)

    if not should_promote(client, registered_name, new_metric, metric_key, mode):
        print(f"[register] Skipping promotion — new model is not better than production.")
        return None

    # Register the model artifact from this run
    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
    version = mv.version

    print(f"[register] Registered '{registered_name}' version {version} "
          f"from run {run.info.run_id}")

    # Set the production alias on this version
    client.set_registered_model_alias(
        name=registered_name,
        alias=MLFLOW_PRODUCTION_ALIAS,
        version=version,
    )
    print(f"[register] Set alias '{MLFLOW_PRODUCTION_ALIAS}' on "
          f"'{registered_name}' v{version}")

    # Add descriptive tags
    client.set_model_version_tag(
        registered_name, version, "metric_key", metric_key
    )
    client.set_model_version_tag(
        registered_name, version, "metric_value", f"{new_metric:.4f}"
    )

    return version


def run_registration():
    """Entry point: register best regression AND classification models."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("\n" + "="*60)
    print("[register] Registering best REGRESSION model …")
    reg_run = get_best_run(client, MLFLOW_EXPERIMENT_REGRESSION, "val_rmse", mode="min")
    reg_version = register_and_promote(
        client, reg_run, MLFLOW_REGISTERED_REGRESSOR, "val_rmse", mode="min"
    )

    print("\n" + "="*60)
    print("[register] Registering best CLASSIFICATION model …")
    clf_run = get_best_run(client, MLFLOW_EXPERIMENT_CLASSIFICATION, "val_macro_f1", mode="max")
    clf_version = register_and_promote(
        client, clf_run, MLFLOW_REGISTERED_CLASSIFIER, "val_macro_f1", mode="max"
    )

    print("\n" + "="*60)
    if reg_version:
        print(f"[register] ✓ Regressor  → '{MLFLOW_REGISTERED_REGRESSOR}' v{reg_version} "
              f"[{MLFLOW_PRODUCTION_ALIAS}]")
    else:
        print(f"[register] ✗ Regressor  → NOT promoted (existing model is better)")
    if clf_version:
        print(f"[register] ✓ Classifier → '{MLFLOW_REGISTERED_CLASSIFIER}' v{clf_version} "
              f"[{MLFLOW_PRODUCTION_ALIAS}]")
    else:
        print(f"[register] ✗ Classifier → NOT promoted (existing model is better)")


def load_production_model(registered_name: str, alias: str = MLFLOW_PRODUCTION_ALIAS):
    """
    Load a model from the registry by alias (NOT by hardcoded file path).

    WHY BY ALIAS, NOT BY FILE PATH:
    File paths are machine-specific and break when you move the MLflow
    server or retrain. The alias 'production' always points to the right
    version regardless of where it lives, and the registry handles versioning.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{registered_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"[register] Loaded model from registry: {model_uri}")
    return model


if __name__ == "__main__":
    run_registration()
