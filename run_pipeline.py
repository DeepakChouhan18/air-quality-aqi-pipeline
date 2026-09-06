#!/usr/bin/env python3
"""
run_pipeline.py — Manual end-to-end pipeline runner (no Airflow required).

Run this script to verify the full pipeline works before wiring it into Airflow.
Usage:
    python run_pipeline.py              # full run: ingest + transform + load
    python run_pipeline.py --skip-ingest  # re-use today's raw files (saves API calls)

WHY THIS EXISTS:
Debugging inside Airflow is painful — log files are spread across tasks,
XCom paths can be wrong, and the DAG won't even import if there's a syntax
error in the task functions. Running the pipeline as a plain Python script
first lets you catch errors quickly, inspect intermediate DataFrames, and
confirm data lands in Postgres before introducing the orchestration layer.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.config import RAW_DIR, LOGS_DIR


def find_todays_raw_files():
    """Return today's raw file paths if they already exist."""
    today = datetime.now(timezone.utc).date().isoformat()
    aq_path  = RAW_DIR / "openaq"  / today / "measurements.json"
    wx_path  = RAW_DIR / "weather" / today / "weather.json"
    return aq_path, wx_path


def main(skip_ingest: bool = False):
    separator = "=" * 60

    print(f"\n{separator}")
    print("STAGE 1: EXTRACT (Ingestion)")
    print(separator)

    aq_path, wx_path = find_todays_raw_files()

    if skip_ingest and aq_path.exists() and wx_path.exists():
        print(f"[skip] Using existing raw files:\n  {aq_path}\n  {wx_path}")
    else:
        print("[openaq ] Fetching locations + latest measurements...")
        aq_path = run_openaq_ingestion()
        print("[weather] Fetching hourly weather data...")
        wx_path = run_weather_ingestion()

    print(f"\n{separator}")
    print("STAGE 2: TRANSFORM & VALIDATE")
    print(separator)

    print("[openaq ] Parsing + validating air quality records...")
    aq_staging, aq_rejected = clean_openaq(aq_path)

    print("[weather] Parsing weather records...")
    wx_staging = clean_weather(wx_path)

    print(f"\n{separator}")
    print("STAGE 3: LOAD -> core schema")
    print(separator)

    print("[db] Loading air quality into core.air_quality ...")
    pollutant_df = load_air_quality(aq_staging)
    print(f"     -> {len(pollutant_df)} rows")

    print("[db] Loading weather into core.weather ...")
    weather_df = load_weather(wx_staging)
    print(f"     -> {len(weather_df)} rows")

    print(f"\n{separator}")
    print("STAGE 4: BUILD ANALYTICS -> analytics schema")
    print(separator)

    build_and_load_analytics(pollutant_df, weather_df)

    print(f"\n{separator}")
    print("STAGE 5: SYNC LINEAGE -> staging.pipeline_runs")
    print(separator)

    load_pipeline_runs_from_csv(LOGS_DIR / "pipeline_runs.csv")
    print("[lineage] CSV pipeline log synced to Postgres.")

    print(f"\n{separator}")
    print("PIPELINE COMPLETE - check Postgres tables:")
    print("    SELECT COUNT(*) FROM core.air_quality;")
    print("    SELECT COUNT(*) FROM core.weather;")
    print("    SELECT * FROM analytics.daily_aqi LIMIT 5;")
    print(separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AQI pipeline end-to-end without Airflow.")
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip API calls and reuse today's existing raw JSON files."
    )
    args = parser.parse_args()
    main(skip_ingest=args.skip_ingest)
