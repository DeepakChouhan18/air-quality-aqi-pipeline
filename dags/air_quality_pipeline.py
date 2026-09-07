"""
Airflow DAG: end-to-end AQI pipeline.

    extract_openaq ─┐
                     ├─► clean_openaq ─┐
    extract_weather ┘                  ├─► load_analytics ─► sync_lineage
                     clean_weather ────┘

WHY THESE TASK BOUNDARIES:
Each task does ONE thing (extract, clean, load) so that if, say,
cleaning fails, Airflow can retry just that task without re-hitting
the APIs. This also means each stage's lineage record is independent
and precisely tells you which stage broke.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path regardless of how Airflow worker was launched
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.ingestion.openaq_api import run_openaq_ingestion
from src.ingestion.weather_api import run_weather_ingestion
from src.transformation.clean_air_quality import clean_and_save as clean_openaq
from src.transformation.clean_weather import clean_and_save as clean_weather
from src.database.load_postgres import (
    load_air_quality,
    load_weather,
    build_and_load_analytics,
    load_pipeline_runs_from_csv,
)
from src.config import LOGS_DIR, RAW_DIR

import pandas as pd

default_args = {
    "owner": "data-eng",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _extract_openaq(**context):
    """
    WHY WE CHECK FOR EXISTING FILES FIRST:
    Airflow retries a failed task from scratch. If extract_openaq already
    wrote raw data before crashing, re-running it would make another API
    call, wasting rate-limit quota and potentially getting blocked. We
    check for today's file first so the task is idempotent.
    """
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    existing = RAW_DIR / "openaq" / today / "measurements.json"
    if existing.exists():
        print(f"[extract_openaq] Raw file already exists for today, reusing: {existing}")
        path = existing
    else:
        path = run_openaq_ingestion()
    context["ti"].xcom_push(key="openaq_raw_path", value=str(path))


def _extract_weather(**context):
    """
    Same idempotency pattern as _extract_openaq — reuse today's file
    if Open-Meteo was already called successfully in this pipeline cycle.
    """
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    existing = RAW_DIR / "weather" / today / "weather.json"
    if existing.exists():
        print(f"[extract_weather] Raw file already exists for today, reusing: {existing}")
        path = existing
    else:
        path = run_weather_ingestion()
    context["ti"].xcom_push(key="weather_raw_path", value=str(path))


def _clean_openaq(**context):
    raw_path = context["ti"].xcom_pull(key="openaq_raw_path", task_ids="extract_openaq")
    staging_path, _ = clean_openaq(raw_path)
    context["ti"].xcom_push(key="openaq_staging_path", value=str(staging_path))


def _clean_weather(**context):
    raw_path = context["ti"].xcom_pull(key="weather_raw_path", task_ids="extract_weather")
    staging_path = clean_weather(raw_path)
    context["ti"].xcom_push(key="weather_staging_path", value=str(staging_path))


def _load_analytics(**context):
    openaq_staging = context["ti"].xcom_pull(key="openaq_staging_path", task_ids="clean_openaq")
    weather_staging = context["ti"].xcom_pull(key="weather_staging_path", task_ids="clean_weather")

    pollutant_df = load_air_quality(openaq_staging)
    weather_df = load_weather(weather_staging)
    build_and_load_analytics(pollutant_df, weather_df)


def _sync_lineage(**context):
    load_pipeline_runs_from_csv(LOGS_DIR / "pipeline_runs.csv")


with DAG(
    dag_id="air_quality_pipeline",
    description="Extract, validate, transform, and load AQI + weather data",
    default_args=default_args,
    start_date=datetime(2026, 9, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["aqi", "data-engineering"],
) as dag:

    extract_openaq = PythonOperator(task_id="extract_openaq", python_callable=_extract_openaq)
    extract_weather = PythonOperator(task_id="extract_weather", python_callable=_extract_weather)

    clean_openaq_task = PythonOperator(task_id="clean_openaq", python_callable=_clean_openaq)
    clean_weather_task = PythonOperator(task_id="clean_weather", python_callable=_clean_weather)

    load_analytics_task = PythonOperator(task_id="load_analytics", python_callable=_load_analytics)
    sync_lineage_task = PythonOperator(task_id="sync_lineage", python_callable=_sync_lineage)

    extract_openaq >> clean_openaq_task
    extract_weather >> clean_weather_task
    [clean_openaq_task, clean_weather_task] >> load_analytics_task >> sync_lineage_task
