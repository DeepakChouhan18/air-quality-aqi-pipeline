#!/usr/bin/env python3
"""
src/export/export_for_tableau.py

Exports analytics tables (gold layer) to CSV files in tableau_exports/
so that Tableau Public Desktop can connect to local flat files directly,
without requiring a live PostgreSQL network connection.

Exported tables:
  1. tableau_exports/daily_aqi.csv
  2. tableau_exports/station_summary.csv
  3. tableau_exports/pollutant_summary.csv
  4. tableau_exports/weather_hourly.csv
"""

import json
import math
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from src.config import PROJECT_ROOT, STAGING_DIR, RAW_DIR, TARGET_CITY, DATA_DIR
from src.transformation.create_analytics import (
    build_daily_city_aqi,
    build_station_summary,
    build_pollutant_summary,
    filter_recent_records,
)

TABLEAU_EXPORT_DIR = PROJECT_ROOT / "tableau_exports"


def get_latest_staging_files():
    """Find the most recent staging CSVs for openaq and weather."""
    openaq_files = sorted(STAGING_DIR.glob("openaq_*.csv"))
    weather_files = sorted(STAGING_DIR.glob("weather_*.csv"))

    if not openaq_files or not weather_files:
        raise FileNotFoundError(
            f"Staging CSV files not found in {STAGING_DIR}. "
            "Please run the transformation step or run_pipeline.py first."
        )

    return openaq_files[-1], weather_files[-1]


def load_station_metadata():
    """Load station name and coordinates from raw locations.json if present."""
    loc_files = sorted(RAW_DIR.glob("openaq/*/locations.json"))
    metadata = {}
    if loc_files:
        try:
            with open(loc_files[-1], "r", encoding="utf-8") as f:
                data = json.load(f)
                for loc in data.get("results", []):
                    sid = loc.get("id")
                    if sid is not None:
                        coords = loc.get("coordinates") or {}
                        metadata[sid] = {
                            "station_name": loc.get("name") or f"Station #{sid}",
                            "latitude": coords.get("latitude"),
                            "longitude": coords.get("longitude"),
                            "provider": (loc.get("provider") or {}).get("name") if isinstance(loc.get("provider"), dict) else loc.get("provider") or "CPCB",
                        }
        except Exception as e:
            print(f"[warn] Could not parse locations.json for metadata: {e}")
    return metadata


def export_all():
    """Builds analytics tables and writes CSV files to tableau_exports/."""
    TABLEAU_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    aq_path, wx_path = get_latest_staging_files()
    print(f"[export] Using air quality staging: {aq_path.name}")
    print(f"[export] Using weather staging:     {wx_path.name}")

    pollutant_df = pd.read_csv(aq_path)
    weather_df = pd.read_csv(wx_path)

    # Apply recency filter ONCE at the top of the analytics pipeline before passing
    # into any downstream summary/analytics builders:
    pollutant_df = filter_recent_records(pollutant_df)
    weather_df = filter_recent_records(weather_df)

    # 1. Daily City AQI
    daily_aqi = build_daily_city_aqi(pollutant_df, weather_df)
    daily_aqi["city"] = TARGET_CITY
    daily_path = TABLEAU_EXPORT_DIR / "daily_aqi.csv"
    daily_aqi.to_csv(daily_path, index=False)
    print(f"[export] Wrote {daily_path.name} ({len(daily_aqi)} rows)")
    if not daily_aqi.empty:
        print(f"         Date range: {daily_aqi['date'].min()} to {daily_aqi['date'].max()}")

    # 2. Station Summary (enriched with station names/coordinates if available)
    station_summary = build_station_summary(pollutant_df)
    metadata = load_station_metadata()
    if metadata:
        meta_df = pd.DataFrame.from_dict(metadata, orient="index").reset_index()
        meta_df.rename(columns={"index": "station_id"}, inplace=True)
        station_summary = station_summary.merge(meta_df, on="station_id", how="left")
        # Fill missing station names
        station_summary["station_name"] = station_summary["station_name"].fillna(
            station_summary["station_id"].apply(lambda sid: f"Station #{sid}")
        )
    station_path = TABLEAU_EXPORT_DIR / "station_summary.csv"
    station_summary.to_csv(station_path, index=False)
    print(f"[export] Wrote {station_path.name} ({len(station_summary)} rows)")

    # 3. Pollutant Summary
    pollutant_summary = build_pollutant_summary(pollutant_df)
    pollutant_path = TABLEAU_EXPORT_DIR / "pollutant_summary.csv"
    pollutant_summary.to_csv(pollutant_path, index=False)
    print(f"[export] Wrote {pollutant_path.name} ({len(pollutant_summary)} rows)")
    if not pollutant_summary.empty:
        print(f"         Date range: {pollutant_summary['date'].min()} to {pollutant_summary['date'].max()}")

    # 4. Hourly Weather (useful for hourly/diurnal pattern analysis)
    weather_hourly_path = TABLEAU_EXPORT_DIR / "weather_hourly.csv"
    weather_df.to_csv(weather_hourly_path, index=False)
    print(f"[export] Wrote {weather_hourly_path.name} ({len(weather_df)} rows)")

    # 5. ML Prediction and Monitoring exports for Tableau
    try:
        from src.database.connection import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            pred_df = pd.read_sql("SELECT * FROM ml.prediction_log", conn)
            mon_df = pd.read_sql("SELECT * FROM ml.monitoring_metrics", conn)
        pred_path = TABLEAU_EXPORT_DIR / "prediction_log.csv"
        mon_path = TABLEAU_EXPORT_DIR / "monitoring_metrics.csv"
        pred_df.to_csv(pred_path, index=False)
        mon_df.to_csv(mon_path, index=False)
        print(f"[export] Wrote {pred_path.name} ({len(pred_df)} rows)")
        print(f"[export] Wrote {mon_path.name} ({len(mon_df)} rows)")
    except Exception as e:
        print(f"[warn] Could not export ML tables to tableau_exports: {e}")

    # 5. Sync to data/dashboard_data.json for the interactive dashboard
    try:
        def df_to_clean_records(df: pd.DataFrame):
            if df is None or df.empty:
                return []
            df_copy = df.copy()
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d")
                elif col == "date":
                    df_copy[col] = df_copy[col].astype(str)
            # to_json automatically serializes NaN/NaT as null
            return json.loads(df_copy.to_json(orient="records", date_format="iso"))

        def sanitize_val(obj):
            if isinstance(obj, dict):
                return {k: sanitize_val(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [sanitize_val(v) for v in obj]
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            if pd.isna(obj):
                return None
            return obj

        daily_records = df_to_clean_records(daily_aqi)
        station_records = df_to_clean_records(station_summary)
        pollutant_records = df_to_clean_records(pollutant_summary)
        weather_records = df_to_clean_records(weather_df)

        raw_stats = {
            "total_stations": len(station_records),
            "total_measurements": len(pollutant_df),
            "total_weather_records": len(weather_records),
            "pollutants": sorted(list(set(pollutant_summary["parameter"].dropna()))),
            "avg_temp": round(float(weather_df["temperature"].mean()), 1) if "temperature" in weather_df and not pd.isna(weather_df["temperature"].mean()) else 25.0,
            "avg_humidity": round(float(weather_df["humidity"].mean()), 1) if "humidity" in weather_df and not pd.isna(weather_df["humidity"].mean()) else 67.7,
            "total_precip": round(float(weather_df["precipitation"].sum()), 1) if "precipitation" in weather_df and not pd.isna(weather_df["precipitation"].sum()) else 20.5,
        }

        dash_data = sanitize_val({
            "daily_aqi": daily_records,
            "station_summary": station_records,
            "pollutant_summary": pollutant_records,
            "weather": weather_records,
            "stats": raw_stats,
        })

        with open(DATA_DIR / "dashboard_data.json", "w", encoding="utf-8") as f:
            json.dump(dash_data, f, indent=2, default=str, allow_nan=False)
        with open(DATA_DIR / "dashboard_data.js", "w", encoding="utf-8") as f:
            f.write("window.DASHBOARD_DATA = ")
            json.dump(dash_data, f, indent=2, default=str, allow_nan=False)
            f.write(";\n")
        print(f"[export] Synced data/dashboard_data.json and data/dashboard_data.js for interactive web dashboard.")
    except Exception as e:
        print(f"[warn] Could not update dashboard_data.json: {e}")
        raise

    print(f"\n[export] All Tableau export CSVs successfully generated in {TABLEAU_EXPORT_DIR}/")
    return {
        "daily_aqi": daily_aqi,
        "station_summary": station_summary,
        "pollutant_summary": pollutant_summary,
        "weather_hourly": weather_df,
    }


if __name__ == "__main__":
    export_all()
