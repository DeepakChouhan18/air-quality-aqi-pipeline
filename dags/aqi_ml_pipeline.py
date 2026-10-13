"""
Airflow DAG: ML pipeline for AQI prediction.

Pipeline flow:

  refresh_features ─► run_monitoring ─► check_drift_branch ─┬─► (no trigger) END
                                                              └─► train_models
                                                                   └─► register_if_better

WHY THIS STRUCTURE:
- refresh_features: rebuilds ml.features_daily from the latest analytics data
  so the monitoring and retraining always use fresh features.
- run_monitoring: runs data quality + drift + performance checks.
- check_drift_branch: BranchPythonOperator — only proceeds to training if
  a retrain trigger fired. This avoids unnecessary retraining on every
  daily run when the model is still performing well.
- train_models: runs the full train.py pipeline (all models, TimeSeriesSplit CV).
- register_if_better: promotes the new model only if it beats production.

STYLE: Matches the existing air_quality_pipeline.py (PythonOperator,
default_args, @daily, XCom for task outputs, lineage via PipelineRun).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


# ── Task functions ─────────────────────────────────────────────────────────────

def _refresh_features(**context):
    """
    Rebuild ml.features_daily from the latest analytics.daily_aqi data.
    This ensures monitoring and training always reflect the newest data.
    """
    from src.ml.features import build_features
    from src.config import TARGET_CITY
    df = build_features(city=TARGET_CITY, save=True)
    context["ti"].xcom_push(key="n_feature_rows", value=len(df))
    print(f"[ml_dag] Feature table refreshed: {len(df)} rows")


def _run_monitoring(**context):
    """
    Run full monitoring: data quality + feature drift + model performance.
    Pushes the monitoring result dict to XCom so the branch task can read it.
    """
    from src.monitoring.monitor import run_monitoring
    from src.config import TARGET_CITY
    result = run_monitoring(city=TARGET_CITY)
    context["ti"].xcom_push(key="monitoring_result", value=result)
    context["ti"].xcom_push(key="retrain_triggered", value=result.get("retrain_triggered", False))
    print(f"[ml_dag] Monitoring complete. Retrain triggered: {result.get('retrain_triggered')}")


def _check_retrain_branch(**context):
    """
    BranchPythonOperator: return the next task_id to run.
    If retrain is triggered, go to 'train_models'; otherwise 'no_retrain'.
    """
    retrain = context["ti"].xcom_pull(key="retrain_triggered", task_ids="run_monitoring")
    if retrain:
        print("[ml_dag] Retrain triggered — branching to train_models")
        return "train_models"
    else:
        print("[ml_dag] No retrain needed — ending pipeline")
        return "no_retrain"


def _train_models(**context):
    """
    Run full training pipeline (all models, eval, plots).
    Training is always run with MLflow logging enabled.
    """
    from src.ml.train import run_training
    from src.config import TARGET_CITY
    results = run_training(city=TARGET_CITY, log_to_mlflow=True)
    context["ti"].xcom_push(key="best_reg_name",  value=results["best_reg_name"])
    context["ti"].xcom_push(key="best_clf_name",  value=results["best_clf_name"])
    print(f"[ml_dag] Training done. Best regressor: {results['best_reg_name']}, "
          f"Best classifier: {results['best_clf_name']}")


def _register_if_better(**context):
    """
    Register the best model from the latest training run.
    Internally compares against the current production model —
    only promotes if the new model wins.
    """
    from src.ml.register_best import run_registration
    run_registration()
    print("[ml_dag] Model registration complete.")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="aqi_ml_pipeline",
    description="ML pipeline: refresh features → monitor drift → optionally retrain + register",
    default_args=default_args,
    start_date=datetime(2026, 9, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["aqi", "mlops", "ml"],
) as dag:

    refresh_features = PythonOperator(
        task_id="refresh_features",
        python_callable=_refresh_features,
    )

    run_monitoring = PythonOperator(
        task_id="run_monitoring",
        python_callable=_run_monitoring,
    )

    check_retrain_branch = BranchPythonOperator(
        task_id="check_retrain_branch",
        python_callable=_check_retrain_branch,
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=_train_models,
        execution_timeout=timedelta(hours=2),  # training can be slow
    )

    register_if_better = PythonOperator(
        task_id="register_if_better",
        python_callable=_register_if_better,
    )

    no_retrain = EmptyOperator(task_id="no_retrain")

    # DAG edges
    refresh_features >> run_monitoring >> check_retrain_branch
    check_retrain_branch >> [train_models, no_retrain]
    train_models >> register_if_better
