"""
Historical data backfill: pulls ≥12 months of daily air-quality and
weather history for the target city and loads it into the SAME
core.air_quality and core.weather tables used by Part 1.

WHY WE NEED THIS:
Part 1 only collects "latest" measurements — roughly 4 days of data.
Next-day AQI forecasting (Part 2 ML) needs lag features going back 7
days, rolling windows of 3 and 7 days, and a large enough training set
to learn seasonal patterns. With only 4 days we cannot do any of that.

DATA SOURCES (both free, no sign-up required):
1. Open-Meteo Air Quality API (historical):
   https://air-quality-api.open-meteo.com/v1/air-quality
   Variables: pm2_5, pm10, nitrogen_dioxide, sulphur_dioxide,
              ozone, carbon_monoxide, us_aqi
   → These map directly to the core.air_quality schema.

2. Open-Meteo Archive API (historical weather):
   https://archive-api.open-meteo.com/v1/archive
   Variables: temperature_2m, relative_humidity_2m, precipitation,
              wind_speed_10m
   → Maps to core.weather schema.

IMPORTANT: If either API is unreachable, this script raises immediately
with a clear error — it does NOT fabricate data. Rate limits are handled
with retries (same fetch_with_retry used in Part 1).

Usage:
    python -m src.ingestion.backfill_history
    # or via Makefile:
    make backfill
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    TARGET_CITY,
    TARGET_LAT,
    TARGET_LON,
    BACKFILL_MONTHS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_AQ_URL,
    RAW_DIR,
    STAGING_DIR,
    REJECTED_DIR,
)
from src.ingestion.ingestion_utils import fetch_with_retry, PipelineRun
from src.transformation.validation import run_all_validations
from src.database.load_postgres import _upsert_df, build_and_load_analytics
from src.database.connection import get_engine

# ── Date range ────────────────────────────────────────────────────────────────
# Open-Meteo archive goes up to ~5 days before today (data processing lag).
# We pull BACKFILL_MONTHS of history (default 14, giving ≥12 months of usable
# data after lag features eat the first 7 days).

def _date_range() -> tuple[date, date]:
    """
    Compute start/end dates for the backfill.
    End date is 3 days ago (archive lag), start is BACKFILL_MONTHS before that.
    """
    end = date.today() - timedelta(days=3)
    # Approximate: 30.44 days per month
    start = end - timedelta(days=int(BACKFILL_MONTHS * 30.44))
    return start, end


# ── Air quality backfill ───────────────────────────────────────────────────────

def fetch_aq_history(start: date, end: date) -> dict:
    """
    Call Open-Meteo Air Quality historical API.

    WHY HOURLY: The existing core.air_quality table stores hourly records
    (recorded_at is TIMESTAMPTZ with time component). Keeping the same
    granularity means our analytics aggregation code works unchanged.
    """
    params = {
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON,
        "hourly": ",".join([
            "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
            "ozone", "carbon_monoxide", "us_aqi",
        ]),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }
    print(f"[backfill] Fetching AQ history {start} → {end} from Open-Meteo AQ API …")
    response = fetch_with_retry(OPEN_METEO_AQ_URL, params=params, max_retries=3)
    return response.json()


def parse_aq_response(data: dict) -> pd.DataFrame:
    """
    Convert Open-Meteo AQ JSON → tidy DataFrame matching core.air_quality schema.

    The API returns parallel arrays (one per variable), all indexed by a
    shared "time" array. We melt them into one row per (timestamp, pollutant).

    Parameter name mapping: Open-Meteo uses different names from OpenAQ.
    We normalise to the same short names already in core.air_quality so
    existing analytics queries (which filter on parameter='pm25') still work.
    """
    # Open-Meteo AQ API column → core.air_quality parameter name + unit
    PARAM_MAP = {
        "pm2_5":             ("pm25",  "µg/m³"),
        "pm10":              ("pm10",  "µg/m³"),
        "nitrogen_dioxide":  ("no2",   "µg/m³"),
        "sulphur_dioxide":   ("so2",   "µg/m³"),
        "ozone":             ("o3",    "µg/m³"),
        "carbon_monoxide":   ("co",    "µg/m³"),
        # us_aqi is the composite AQI — store it but under a distinct name
        "us_aqi":            ("us_aqi", "AQI"),
    }

    times = data.get("hourly", {}).get("time", [])
    if not times:
        raise ValueError("[backfill] AQ API returned no time values — check date range or API status.")

    rows = []
    # A synthetic station_id for backfilled data — prefixed so it's distinguishable
    station_id = f"open-meteo-aq-{TARGET_LAT:.4f}-{TARGET_LON:.4f}"

    for api_col, (param_name, unit) in PARAM_MAP.items():
        values = data.get("hourly", {}).get(api_col, [])
        for t, v in zip(times, values):
            if v is None:
                continue   # skip NaN hours (API returns null for missing)
            rows.append({
                "station_id":  station_id,
                "parameter":   param_name,
                "value":       float(v),
                "unit":        unit,
                "recorded_at": pd.Timestamp(t, tz="UTC"),
            })

    if not rows:
        raise ValueError("[backfill] AQ API returned all-null values. Cannot load empty data.")

    return pd.DataFrame(rows)


# ── Weather backfill ───────────────────────────────────────────────────────────

def fetch_weather_history(start: date, end: date) -> dict:
    """Call Open-Meteo Archive API for historical weather."""
    params = {
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON,
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "precipitation", "wind_speed_10m",
        ]),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }
    print(f"[backfill] Fetching weather history {start} → {end} from Open-Meteo Archive …")
    response = fetch_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, max_retries=3)
    return response.json()


def parse_weather_response(data: dict) -> pd.DataFrame:
    """
    Convert Open-Meteo Archive JSON → tidy DataFrame matching core.weather schema.
    Column mapping: temperature_2m→temperature, relative_humidity_2m→humidity,
                    precipitation→precipitation, wind_speed_10m→wind_speed.
    """
    times = data.get("hourly", {}).get("time", [])
    if not times:
        raise ValueError("[backfill] Weather archive returned no time values.")

    hourly = data["hourly"]
    df = pd.DataFrame({
        "city":          TARGET_CITY,
        "recorded_at":   pd.to_datetime(times, utc=True),
        "temperature":   hourly.get("temperature_2m", [None] * len(times)),
        "humidity":      hourly.get("relative_humidity_2m", [None] * len(times)),
        "precipitation": hourly.get("precipitation", [None] * len(times)),
        "wind_speed":    hourly.get("wind_speed_10m", [None] * len(times)),
    })
    # Drop rows where all weather values are null (shouldn't happen but defensive)
    df = df.dropna(subset=["temperature", "humidity", "precipitation", "wind_speed"], how="all")
    return df


# ── Validation (reusing Part 1 logic) ────────────────────────────────────────

def validate_aq(df: pd.DataFrame, run: PipelineRun) -> pd.DataFrame:
    """
    Run the SAME validation pipeline used in Part 1 (clean_air_quality.py).
    Rejected records are saved to data/rejected/ for auditability.
    """
    valid_df, rejected_df = run_all_validations(df)

    if not rejected_df.empty:
        rej_path = REJECTED_DIR / f"backfill_aq_rejected_{date.today()}.csv"
        rejected_df.to_csv(rej_path, index=False)
        print(f"[backfill] {len(rejected_df)} AQ rows rejected → {rej_path}")

    run.rows_extracted = len(df)
    run.rows_rejected = len(rejected_df)
    run.rows_valid = len(valid_df)
    return valid_df


# ── DB load (reusing Part 1 upsert pattern) ───────────────────────────────────

def load_aq_to_db(df: pd.DataFrame):
    """
    Upsert into core.air_quality using ON CONFLICT DO NOTHING — idempotent,
    safe to re-run if the backfill is interrupted midway.
    """
    engine = get_engine()
    # Ensure recorded_at is timezone-aware (Postgres TIMESTAMPTZ requires it)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    _upsert_df(df, "air_quality", "core", ["station_id", "parameter", "recorded_at"], engine)
    print(f"[backfill] Upserted {len(df)} AQ rows into core.air_quality")


def load_weather_to_db(df: pd.DataFrame):
    """Upsert into core.weather — idempotent."""
    engine = get_engine()
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    _upsert_df(df, "weather", "core", ["city", "recorded_at"], engine)
    print(f"[backfill] Upserted {len(df)} weather rows into core.weather")


# ── Main backfill orchestrator ────────────────────────────────────────────────

def run_backfill():
    """
    Full historical backfill:
      1. Fetch AQ history from Open-Meteo Air Quality API
      2. Parse + validate (same rules as Part 1)
      3. Upsert into core.air_quality
      4. Fetch weather history from Open-Meteo Archive API
      5. Upsert into core.weather
      6. Log lineage to pipeline_runs.csv (same as Part 1)
    """
    start, end = _date_range()
    today_str = date.today().isoformat()

    # ── Step 1: Air quality ──────────────────────────────────────────────────
    aq_run = PipelineRun(source="open-meteo-aq-history", stage="backfill")
    try:
        raw_aq_dir = RAW_DIR / "backfill" / today_str
        raw_aq_dir.mkdir(parents=True, exist_ok=True)

        aq_data = fetch_aq_history(start, end)

        # Save raw response (raw landing zone — same principle as Part 1)
        raw_aq_path = raw_aq_dir / "aq_history.json"
        with open(raw_aq_path, "w") as f:
            json.dump(aq_data, f)
        print(f"[backfill] Raw AQ response saved → {raw_aq_path}")

        aq_df = parse_aq_response(aq_data)
        aq_df = validate_aq(aq_df, aq_run)

        # Save staging CSV
        staging_aq_path = STAGING_DIR / f"backfill_aq_{today_str}.csv"
        aq_df.to_csv(staging_aq_path, index=False)

        load_aq_to_db(aq_df)
        aq_run.mark_success(
            rows_extracted=aq_run.rows_extracted,
            rows_valid=aq_run.rows_valid,
            rows_rejected=aq_run.rows_rejected,
        )
        print(f"[backfill] ✓ AQ backfill complete: {aq_run.rows_valid} valid rows loaded.")

    except Exception as e:
        aq_run.mark_failure(e)
        print(f"[backfill] ✗ AQ backfill FAILED: {e}")
        raise   # fail loudly — do not fabricate data

    # ── Step 2: Weather ──────────────────────────────────────────────────────
    wx_run = PipelineRun(source="open-meteo-archive-history", stage="backfill")
    try:
        wx_data = fetch_weather_history(start, end)

        raw_wx_path = raw_aq_dir / "weather_history.json"
        with open(raw_wx_path, "w") as f:
            json.dump(wx_data, f)
        print(f"[backfill] Raw weather response saved → {raw_wx_path}")

        wx_df = parse_weather_response(wx_data)
        staging_wx_path = STAGING_DIR / f"backfill_weather_{today_str}.csv"
        wx_df.to_csv(staging_wx_path, index=False)

        load_weather_to_db(wx_df)
        wx_run.mark_success(
            rows_extracted=len(wx_df),
            rows_valid=len(wx_df),
        )
        print(f"[backfill] ✓ Weather backfill complete: {len(wx_df)} rows loaded.")

        print("[backfill] Updating analytics layer (analytics.daily_aqi) from core tables …")
        engine = get_engine()
        with engine.connect() as conn:
            all_aq = pd.read_sql("SELECT * FROM core.air_quality", conn)
            all_wx = pd.read_sql("SELECT * FROM core.weather", conn)
        build_and_load_analytics(all_aq, all_wx)

    except Exception as e:
        wx_run.mark_failure(e)
        print(f"[backfill] ✗ Weather backfill FAILED: {e}")
        raise

    print(
        f"\n[backfill] Done. Date range: {start} → {end} "
        f"({(end - start).days} days). "
        f"AQ: {aq_run.rows_valid} rows | Weather: {len(wx_df)} rows.\n"
        "Now run: make features → make train → make register"
    )


if __name__ == "__main__":
    run_backfill()
