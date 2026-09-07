"""
Parses raw Open-Meteo JSON (column-oriented hourly arrays) into a tidy
row-per-hour DataFrame, then saves it to staging.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import STAGING_DIR, TARGET_CITY
from src.ingestion.ingestion_utils import PipelineRun


def parse_openmeteo_weather(raw_json_path: Path) -> pd.DataFrame:
    """
    Open-Meteo returns hourly data as parallel arrays:
    {"hourly": {"time": [...], "temperature_2m": [...], ...}}
    We convert this into one row per timestamp.
    """
    with open(raw_json_path) as f:
        data = json.load(f)

    hourly = data.get("hourly", {})
    df = pd.DataFrame({
        "recorded_at": hourly.get("time", []),
        "temperature": hourly.get("temperature_2m", []),
        "humidity": hourly.get("relative_humidity_2m", []),
        "precipitation": hourly.get("precipitation", []),
        "wind_speed": hourly.get("wind_speed_10m", []),
    })
    df["city"] = TARGET_CITY
    return df


def clean_and_save(raw_json_path) -> Path:
    raw_json_path = Path(raw_json_path)  # XCom passes str; ensure Path object
    run = PipelineRun(source="open-meteo", stage="transform")
    today = datetime.now(timezone.utc).date().isoformat()

    try:
        df = parse_openmeteo_weather(raw_json_path)
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")
        df = df.dropna(subset=["recorded_at"])

        staging_path = STAGING_DIR / f"weather_{today}.csv"
        df.to_csv(staging_path, index=False)

        run.mark_success(rows_extracted=len(df), rows_valid=len(df), rows_rejected=0)
        print(f"[open-meteo] {len(df)} rows cleaned -> {staging_path.name}")
        return staging_path

    except Exception as e:
        run.mark_failure(e)
        raise


if __name__ == "__main__":
    import sys
    clean_and_save(Path(sys.argv[1]))
