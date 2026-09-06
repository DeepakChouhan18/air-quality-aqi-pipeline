"""
Parses raw OpenAQ JSON into a tidy pandas DataFrame, runs validation,
and writes valid + rejected records to the staging/rejected folders
(and later, Postgres, via load_postgres.py).

WHY WE SEPARATE "PARSE" FROM "VALIDATE":
Parsing turns nested JSON into flat rows. Validation then checks
business rules on those rows. Keeping them as separate functions makes
each one independently testable (see tests/test_data_quality.py).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import STAGING_DIR, REJECTED_DIR
from src.ingestion.ingestion_utils import PipelineRun
from src.transformation.validation import run_all_validations


def parse_openaq_measurements(raw_json_path) -> pd.DataFrame:
    # Ensure Path object — XCom passes paths as plain strings from Airflow tasks
    raw_json_path = Path(raw_json_path)
    """
    Flattens the OpenAQ v3 'latest measurements' JSON (a list of per-station
    responses) into one row per (station, parameter, timestamp).

    CRITICAL DISCOVERY from live API inspection:
    The v3 /locations/{id}/latest endpoint does NOT include parameter names or
    units in each measurement row. A measurement looks like:
        {
          "sensorsId":   1518,
          "locationsId": 412,
          "value":       25.02,
          "datetime":    {"utc": "2018-02-22T02:45:00Z", "local": "..."},
          "coordinates": {...}
        }
    The pollutant name (pm25, no2, etc.) and units live in the /locations
    response under locations[].sensors[].parameter. The ingestion step saves
    this as sensors_map.json: {sensorsId -> {parameter, unit}}.

    WHY JOIN HERE (not in ingestion):
    Raw files must be exact copies of API responses (immutable audit trail).
    The join/enrichment is a transformation, not an ingestion concern.
    Keeping them separate means you can re-run this transform against the
    same raw files with an updated mapping without re-hitting the API.
    """
    with open(raw_json_path) as f:
        raw_data = json.load(f)

    # Load the sensor lookup map saved by the ingestion step.
    # Keys in sensors_map.json are strings (JSON object keys are always strings).
    sensors_map_path = raw_json_path.parent / "sensors_map.json"
    sensors_map = {}
    if sensors_map_path.exists():
        with open(sensors_map_path) as f:
            sensors_map = json.load(f)
    else:
        print(f"[WARNING] sensors_map.json not found at {sensors_map_path}. "
              "Parameter names and units will be null. Re-run ingestion to generate it.")

    rows = []
    for station_response in raw_data:
        for result in station_response.get("results", []):
            sensors_id = result.get("sensorsId")
            # Look up parameter info via sensorsId (JSON keys are strings)
            sensor_info = sensors_map.get(str(sensors_id), {})

            # datetime is always a dict with "utc" and "local" keys in v3
            dt = result.get("datetime", {})
            recorded_at = dt.get("utc") if isinstance(dt, dict) else result.get("date")

            rows.append({
                "station_id":  result.get("locationsId"),
                "parameter":   sensor_info.get("parameter"),
                "value":       result.get("value"),
                "unit":        sensor_info.get("unit"),
                "recorded_at": recorded_at,
            })

    return pd.DataFrame(rows)


def clean_and_save(raw_json_path) -> tuple[Path, Path]:
    """
    Full clean pipeline for one raw OpenAQ file:
    parse -> validate -> save valid to staging, rejected to rejected/.
    Returns (staging_path, rejected_path).
    """
    raw_json_path = Path(raw_json_path)  # XCom passes str; ensure Path object
    run = PipelineRun(source="openaq", stage="transform")
    today = datetime.now(timezone.utc).date().isoformat()

    try:
        df = parse_openaq_measurements(raw_json_path)
        valid_df, rejected_df = run_all_validations(df)

        staging_path = STAGING_DIR / f"openaq_{today}.csv"
        valid_df.to_csv(staging_path, index=False)

        rejected_path = REJECTED_DIR / f"openaq_rejected_{today}.csv"
        rejected_df.to_csv(rejected_path, index=False)

        run.mark_success(
            rows_extracted=len(df),
            rows_valid=len(valid_df),
            rows_rejected=len(rejected_df),
        )
        print(f"[openaq] {len(valid_df)} valid / {len(rejected_df)} rejected "
              f"-> {staging_path.name}")
        return staging_path, rejected_path

    except Exception as e:
        run.mark_failure(e)
        raise


if __name__ == "__main__":
    import sys
    clean_and_save(Path(sys.argv[1]))
