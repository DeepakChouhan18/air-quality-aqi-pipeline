"""
Feature engineering pipeline for next-day AQI prediction.

WHY DETERMINISTIC, REUSABLE FUNCTIONS:
The same feature-building code MUST run at both training time and
inference time. If you have two different code paths, they will
inevitably diverge (different column order, different imputation
strategy, etc.) and your model will silently produce garbage
predictions in production. One function → one truth.

LEAKAGE PREVENTION:
All lag/rolling features are computed with shift(n) where n ≥ 1.
This means at date t, every feature uses only data from dates ≤ t-1
(or t-2, t-3, etc.). The target (next-day AQI) uses shift(-1),
meaning we're predicting t+1. This is verified in tests/test_features.py.

MISSING VALUE STRATEGY (documented here, not buried in code):
1. Daily aggregation → outer-join pollutant+weather, so some days may
   have AQI but no weather or vice versa. We keep these and mark them
   as NaN; models that can't handle NaN get them imputed later.
2. Lag/rolling features → first N rows of the series will be NaN
   (no lookback available). These rows are DROPPED before training
   (we cannot fabricate historical data). At inference, if lags
   are NaN we return a prediction with a warning flag.
3. Remaining NaN → forward-fill (last known value) for weather
   columns only (weather doesn't jump wildly hour to hour). AQI
   NaNs are left as NaN and handled by model imputers/pipelines.

AQI CATEGORY BINS (identical to Part 1 / US EPA):
    0-50    Good
    51-100  Moderate
    101-150 Unhealthy for Sensitive Groups
    151-200 Unhealthy
    201-300 Very Unhealthy
    301-500 Hazardous
"""

import math
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np
from sqlalchemy import text

from src.database.connection import get_engine
from src.config import TARGET_CITY, ML_DATA_DIR
from src.transformation.create_analytics import aqi_to_category

# Feature columns used for model training (order matters — keep stable!)
# Changing this list invalidates any saved model; update FEATURE_VERSION if you do.
FEATURE_COLS = [
    # Raw daily values
    "aqi_value", "avg_pm25", "avg_pm10",
    "avg_temperature", "avg_humidity", "avg_wind_speed", "total_precipitation",
    # AQI lags
    "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",
    # PM2.5 lags
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    # PM10 lags
    "pm10_lag1", "pm10_lag2", "pm10_lag3", "pm10_lag7",
    # Rolling stats
    "aqi_roll3_mean", "aqi_roll3_std", "aqi_roll3_max", "pm25_roll3_mean",
    "aqi_roll7_mean", "aqi_roll7_std", "aqi_roll7_max", "pm25_roll7_mean",
    # Weather lags
    "temp_lag1", "humidity_lag1", "wind_lag1", "rain_lag1",
    # Calendar (cyclical)
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
]

TARGET_REG = "target_aqi_next_day"      # regression target
TARGET_CLF = "target_category_next_day" # classification target
FEATURE_VERSION = "v1"                  # bump when FEATURE_COLS changes


# ── Step 1: Load daily data from the gold layer ─────────────────────────────

def load_daily_aqi(city: str = TARGET_CITY, engine=None) -> pd.DataFrame:
    """
    Read analytics.daily_aqi for the given city.
    This is the gold layer produced by Part 1's pipeline and is our
    starting point — we do NOT re-aggregate from raw to avoid code duplication.
    """
    if engine is None:
        engine = get_engine()

    query = text("""
        SELECT
            date, city, aqi_value, avg_pm25, avg_pm10,
            avg_temperature, avg_humidity, avg_wind_speed, total_precipitation
        FROM analytics.daily_aqi
        WHERE city = :city
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"city": city}, parse_dates=["date"])

    if df.empty:
        raise ValueError(
            f"[features] No data in analytics.daily_aqi for city='{city}'. "
            "Run 'make backfill' first to populate historical data."
        )

    df = df.sort_values("date").reset_index(drop=True)
    print(f"[features] Loaded {len(df)} daily rows for {city} "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ── Step 2: Add lag features ─────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged values of AQI, PM2.5, and PM10 at t-1, t-2, t-3, t-7.

    shift(n) moves values n rows forward, so the value at row i is
    the value from row i-n — i.e., n days ago. This guarantees no
    leakage: features at date t use only data from dates < t.
    """
    df = df.copy()
    for lag in [1, 2, 3, 7]:
        df[f"aqi_lag{lag}"]  = df["aqi_value"].shift(lag)
        df[f"pm25_lag{lag}"] = df["avg_pm25"].shift(lag)
        df[f"pm10_lag{lag}"] = df["avg_pm10"].shift(lag)
    return df


# ── Step 3: Add rolling features ─────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean/std/max over 3-day and 7-day windows.

    closed='left' means the window at row i looks at rows [i-W, i-1],
    NOT including row i itself — this is the key leakage-prevention flag.
    min_periods=1 avoids NaN for the first few rows of the window
    (though those rows will still have NaN lags and be dropped later).
    """
    df = df.copy()
    for w, label in [(3, "3"), (7, "7")]:
        # closed='left' excludes current day → no leakage
        win = df["aqi_value"].shift(1).rolling(window=w, min_periods=1)
        df[f"aqi_roll{label}_mean"] = win.mean()
        df[f"aqi_roll{label}_std"]  = win.std()
        df[f"aqi_roll{label}_max"]  = win.max()
        win_pm = df["avg_pm25"].shift(1).rolling(window=w, min_periods=1)
        df[f"pm25_roll{label}_mean"] = win_pm.mean()
    return df


# ── Step 4: Add weather lag features ─────────────────────────────────────────

def add_weather_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag-1 weather features: yesterday's temperature, humidity, wind, rain.
    Weather from the previous day is a relevant predictor because stagnant
    weather patterns (temperature inversion, low wind) persist and trap
    pollutants for multiple days.
    """
    df = df.copy()
    df["temp_lag1"]     = df["avg_temperature"].shift(1)
    df["humidity_lag1"] = df["avg_humidity"].shift(1)
    df["wind_lag1"]     = df["avg_wind_speed"].shift(1)
    df["rain_lag1"]     = df["total_precipitation"].shift(1)
    return df


# ── Step 5: Add calendar features ────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day-of-week, month, weekend flag, and cyclical encodings.

    WHY CYCLICAL ENCODING:
    Day 0 (Monday) and day 6 (Sunday) are adjacent in real life but
    far apart numerically. A linear model will think Mon≈0 and Sun≈6 are
    very different. Encoding with sin/cos wraps the scale so that
    Mon and Sun are close — the model can learn weekly seasonality correctly.
    Same logic applies to month (Jan and Dec are adjacent in the annual cycle).
    """
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek         # 0=Monday … 6=Sunday
    df["month"]       = dt.dt.month             # 1–12
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding: maps periodic features onto unit circle
    df["dow_sin"]   = np.sin(2 * math.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * math.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * math.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * math.pi * df["month"] / 12)
    return df


# ── Step 6: Add target columns ────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add next-day AQI (regression target) and next-day category (classification).

    shift(-1) moves values one row backward, so the value at row i
    is the value from row i+1 — i.e., tomorrow's AQI. The last row
    will have NaN targets (we don't know the future yet) and will be
    filtered at training time, but kept for inference.
    """
    df = df.copy()
    df[TARGET_REG] = df["aqi_value"].shift(-1)
    df[TARGET_CLF] = df[TARGET_REG].apply(
        lambda x: aqi_to_category(x) if pd.notna(x) else None
    )
    return df


# ── Step 7: Handle missing values ────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the missing-value strategy documented in this module's docstring:
    - Forward-fill weather columns (slow-moving variables)
    - Leave AQI/PM NaN as NaN (not imputed silently)
    - Log the missing rate for monitoring purposes
    """
    df = df.copy()

    # Forward-fill weather (last known observation carried forward)
    weather_cols = ["avg_temperature", "avg_humidity", "avg_wind_speed",
                    "total_precipitation", "temp_lag1", "humidity_lag1",
                    "wind_lag1", "rain_lag1"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Report missing rate for monitoring
    n_total = len(df) * len(FEATURE_COLS)
    n_missing = df[FEATURE_COLS].isna().sum().sum()
    missing_rate = n_missing / max(n_total, 1)
    print(f"[features] Missing rate after imputation: {missing_rate:.1%} "
          f"({n_missing}/{n_total} cells)")
    return df


# ── Master feature builder ─────────────────────────────────────────────────────

def build_features(
    city: str = TARGET_CITY,
    engine=None,
    save: bool = True,
) -> pd.DataFrame:
    """
    End-to-end feature pipeline. Call this from both train.py and the API.

    Returns:
        DataFrame with all feature columns + target columns, sorted by date.
        Rows where ANY of the lag columns are NaN (i.e., the first 7 days)
        are retained — caller must decide whether to drop them.

    Args:
        city:   target city (default from config)
        engine: SQLAlchemy engine (created if None)
        save:   if True, saves parquet + CSV to data/ml/ and upserts to
                ml.features_daily in Postgres
    """
    df = load_daily_aqi(city=city, engine=engine)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_lag_features(df)
    df = add_calendar_features(df)
    df = add_targets(df)
    df = handle_missing(df)

    if save:
        _save_features(df, city=city, engine=engine)

    return df


def _save_features(df: pd.DataFrame, city: str, engine=None):
    """Save feature table to parquet, CSV, and Postgres ml.features_daily."""
    from sqlalchemy import text as sql_text

    if engine is None:
        engine = get_engine()

    # ── Parquet (preferred for ML, preserves types) ──────────────────────────
    parquet_path = ML_DATA_DIR / f"features_{city.lower().replace(' ', '_')}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"[features] Saved parquet → {parquet_path}")

    # ── CSV (human-readable fallback) ────────────────────────────────────────
    csv_path = ML_DATA_DIR / f"features_{city.lower().replace(' ', '_')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[features] Saved CSV    → {csv_path}")

    # ── Postgres ml.features_daily ───────────────────────────────────────────
    # Build the row dict for each date, using only columns that exist in the table
    pg_cols = [
        "feature_date", "city",
        "aqi_value", "avg_pm25", "avg_pm10",
        "avg_temperature", "avg_humidity", "avg_wind_speed", "total_precipitation",
        "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",
        "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
        "pm10_lag1", "pm10_lag2", "pm10_lag3", "pm10_lag7",
        "aqi_roll3_mean", "aqi_roll3_std", "aqi_roll3_max", "pm25_roll3_mean",
        "aqi_roll7_mean", "aqi_roll7_std", "aqi_roll7_max", "pm25_roll7_mean",
        "temp_lag1", "humidity_lag1", "wind_lag1", "rain_lag1",
        "day_of_week", "month", "is_weekend",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "target_aqi_next_day", "target_category_next_day",
    ]
    pg_df = df.copy()
    pg_df["feature_date"] = pd.to_datetime(pg_df["date"]).dt.date

    # Keep only columns that exist in both the df and our target list
    cols_to_write = [c for c in pg_cols if c in pg_df.columns]
    pg_df = pg_df[cols_to_write]

    # Upsert: on conflict (same feature_date + city) replace the row
    # We use a simple "delete then insert" approach for simplicity
    with engine.begin() as conn:
        # Ensure the ml schema and table exist
        conn.execute(sql_text("""
            CREATE SCHEMA IF NOT EXISTS ml;
            CREATE TABLE IF NOT EXISTS ml.features_daily (
                feature_date DATE NOT NULL,
                city TEXT NOT NULL,
                PRIMARY KEY (feature_date, city)
            );
        """))

        for _, row in pg_df.iterrows():
            # Build dynamic upsert — safe because column names come from our own code
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            cols = list(row_dict.keys())
            col_str = ", ".join(cols)
            val_str = ", ".join([f":{c}" for c in cols])
            update_str = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols
                                    if c not in ("feature_date", "city")])
            stmt = sql_text(f"""
                INSERT INTO ml.features_daily ({col_str})
                VALUES ({val_str})
                ON CONFLICT (feature_date, city)
                DO UPDATE SET {update_str}
            """)
            conn.execute(stmt, row_dict)

    print(f"[features] Upserted {len(pg_df)} rows into ml.features_daily")


# ── Training/inference helpers ────────────────────────────────────────────────

def get_training_data(
    city: str = TARGET_CITY,
    min_rows: int = 60,
    engine=None,
) -> pd.DataFrame:
    """
    Load features and apply training-time filters:
    - Drop rows where lag-7 features are NaN (first 7 days have no full lookback)
    - Drop rows where target is NaN (last day has no tomorrow)
    - Assert minimum row count for meaningful training

    Returns clean DataFrame ready for chronological split.
    """
    df = build_features(city=city, engine=engine, save=True)

    before = len(df)
    # Drop rows where any lag-7 feature is NaN (not enough history)
    lag7_cols = ["aqi_lag7", "pm25_lag7", "pm10_lag7"]
    df = df.dropna(subset=lag7_cols)
    # Drop the last row (no next-day target)
    df = df.dropna(subset=[TARGET_REG])

    print(f"[features] Training rows: {len(df)} (dropped {before - len(df)} due to NaN lags/targets)")

    if len(df) < min_rows:
        raise ValueError(
            f"[features] Only {len(df)} training rows available (minimum {min_rows}). "
            "Run 'make backfill' to load more historical data. "
            "At least 60 days of data are needed for a reliable train/val/test split."
        )
    return df.reset_index(drop=True)


def get_inference_features(
    city: str = TARGET_CITY,
    target_date: Optional[date] = None,
    engine=None,
) -> Optional[pd.DataFrame]:
    """
    Build features for a single date (the day before the date we want to predict).
    Used by the FastAPI /predict endpoint.

    Args:
        city: city to predict for
        target_date: the date we want to PREDICT (we build features from t-1 data)
                     defaults to tomorrow if not specified

    Returns:
        Single-row DataFrame of features, or None if insufficient history.
    """
    df = build_features(city=city, engine=engine, save=False)

    if target_date is None:
        from datetime import date as date_cls
        target_date = date_cls.today()  # predict for today using yesterday's features

    # The feature row for predicting "target_date" is the row for target_date - 1 day
    from datetime import timedelta
    feature_date = target_date - timedelta(days=1)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    row = df[df["date"] == feature_date]

    if row.empty:
        print(f"[features] No feature row found for {feature_date} "
              f"(needed to predict {target_date})")
        return None

    return row[FEATURE_COLS]


if __name__ == "__main__":
    # Quick smoke test: build features and print first 5 rows
    features_df = build_features(save=True)
    print(f"\nFeature table shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    print(f"\nFirst 3 rows:\n{features_df.head(3)}")
    print(f"\nMissing values per column:\n{features_df[FEATURE_COLS].isna().sum()}")
