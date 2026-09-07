"""
Loads cleaned staging CSVs into core.* tables, then triggers the
analytics build and writes gold-layer tables. Also writes lineage
records into staging.pipeline_runs (mirrors the CSV log so the audit
trail lives in the database once it's available).
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine
from src.transformation.create_analytics import (
    build_daily_city_aqi,
    build_station_summary,
    build_pollutant_summary,
    filter_recent_records,
)
from src.ingestion.ingestion_utils import PipelineRun
from src.config import TARGET_CITY


def _upsert_df(df: pd.DataFrame, table: str, schema: str, conflict_cols: list[str], engine):
    """
    Insert rows using PostgreSQL's ON CONFLICT DO NOTHING pattern.

    WHY NOT to_sql(if_exists='append'):
    pandas to_sql blindly appends rows. When Airflow retries a task (e.g.,
    after a partial network failure), it re-inserts data that already landed,
    hitting the UNIQUE constraint and raising IntegrityError. ON CONFLICT DO
    NOTHING skips duplicates silently, making every load fully idempotent.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy import Table, MetaData

    meta = MetaData()
    meta.reflect(bind=engine, schema=schema, only=[table])
    tbl = meta.tables[f"{schema}.{table}"]

    with engine.begin() as conn:
        for _, row in df.iterrows():
            stmt = pg_insert(tbl).values(**row.to_dict())
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
            conn.execute(stmt)


def load_air_quality(staging_csv) -> pd.DataFrame:
    staging_csv = Path(staging_csv)  # XCom passes str; ensure Path object
    engine = get_engine()
    df = pd.read_csv(staging_csv, parse_dates=["recorded_at"])
    # First do a fast bulk load attempt; fall back to upsert on conflict
    try:
        df.to_sql("air_quality", engine, schema="core", if_exists="append", index=False)
    except Exception:
        # On conflict (re-run), use per-row upsert to skip existing rows
        _upsert_df(df, "air_quality", "core", ["station_id", "parameter", "recorded_at"], engine)
    return df


def load_weather(staging_csv) -> pd.DataFrame:
    staging_csv = Path(staging_csv)  # XCom passes str; ensure Path object
    engine = get_engine()
    df = pd.read_csv(staging_csv, parse_dates=["recorded_at"])
    try:
        df.to_sql("weather", engine, schema="core", if_exists="append", index=False)
    except Exception:
        _upsert_df(df, "weather", "core", ["city", "recorded_at"], engine)
    return df


def load_pipeline_runs_from_csv(csv_path: Path):
    """Sync the local CSV lineage log into Postgres (idempotent: skips existing run_ids)."""
    csv_path = Path(csv_path)
    engine = get_engine()
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    try:
        df.to_sql("pipeline_runs", engine, schema="staging", if_exists="append", index=False)
    except Exception:
        _upsert_df(df, "pipeline_runs", "staging", ["run_id"], engine)


def build_and_load_analytics(pollutant_df: pd.DataFrame, weather_df: pd.DataFrame):
    """Builds all gold-layer tables and writes them to Postgres (full refresh)."""
    engine = get_engine()
    run = PipelineRun(source="analytics", stage="load")

    try:
        pollutant_df = filter_recent_records(pollutant_df)
        weather_df = filter_recent_records(weather_df)

        daily = build_daily_city_aqi(pollutant_df, weather_df)
        daily["city"] = TARGET_CITY
        daily.to_sql("daily_aqi", engine, schema="analytics", if_exists="replace", index=False)

        station_summary = build_station_summary(pollutant_df)
        station_summary.to_sql(
            "station_summary", engine, schema="analytics", if_exists="replace", index=False
        )

        pollutant_summary = build_pollutant_summary(pollutant_df)
        pollutant_summary.to_sql(
            "pollutant_summary", engine, schema="analytics", if_exists="replace", index=False
        )

        run.mark_success(rows_valid=len(daily))
        print(f"[analytics] Loaded {len(daily)} daily_aqi rows, "
              f"{len(station_summary)} stations, {len(pollutant_summary)} pollutant rows.")

    except Exception as e:
        run.mark_failure(e)
        raise
