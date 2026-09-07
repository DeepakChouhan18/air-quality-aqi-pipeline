"""
Extracts weather data from the Open-Meteo API and saves it to the raw
landing zone. No API key required for non-commercial use.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import OPEN_METEO_BASE_URL, TARGET_LAT, TARGET_LON, RAW_DIR
from src.ingestion.ingestion_utils import fetch_with_retry, PipelineRun


def get_weather(past_days: int = 7) -> dict:
    """Fetch hourly weather variables for the target coordinates."""
    params = {
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "past_days": past_days,
        "timezone": "auto",
    }
    response = fetch_with_retry(OPEN_METEO_BASE_URL, params=params)
    return response.json()


def run_weather_ingestion() -> Path:
    """Full ingestion run for weather data, with lineage logging."""
    run = PipelineRun(source="open-meteo", stage="extract")
    today = datetime.now(timezone.utc).date().isoformat()
    out_dir = RAW_DIR / "weather" / today
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = get_weather()
        out_path = out_dir / "weather.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        row_count = len(data.get("hourly", {}).get("time", []))
        run.mark_success(rows_extracted=row_count)
        print(f"[open-meteo] Saved {row_count} hourly records -> {out_path}")
        return out_path

    except Exception as e:
        run.mark_failure(e)
        print(f"[open-meteo] Ingestion FAILED: {e}")
        raise


if __name__ == "__main__":
    run_weather_ingestion()
