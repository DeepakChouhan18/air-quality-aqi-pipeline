-- ==========================================================
-- Part 2 MLOps schema extensions
-- Adds a new `ml` schema alongside the existing raw/staging/core/analytics schemas.
-- Run AFTER sql/schemas.sql, sql/tables.sql, sql/analytics.sql.
-- ==========================================================

-- New schema for all ML layer tables
CREATE SCHEMA IF NOT EXISTS ml;

-- ──────────────────────────────────────────────────────────
-- ml.features_daily
-- One row per date per city — the feature table built by
-- src/ml/features.py. This is the input to model training
-- and inference. Saving it to Postgres enables:
--   (a) full reproducibility (exact features used are stored)
--   (b) fast inference without re-querying all raw tables
--   (c) drift monitoring (compare recent rows vs training)
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.features_daily (
    feature_date    DATE    NOT NULL,
    city            TEXT    NOT NULL,

    -- Raw aggregated values (from analytics.daily_aqi)
    aqi_value               NUMERIC,
    avg_pm25                NUMERIC,
    avg_pm10                NUMERIC,
    avg_temperature         NUMERIC,
    avg_humidity            NUMERIC,
    avg_wind_speed          NUMERIC,
    total_precipitation     NUMERIC,

    -- Lag features: AQI at t-1, t-2, t-3, t-7
    aqi_lag1        NUMERIC,
    aqi_lag2        NUMERIC,
    aqi_lag3        NUMERIC,
    aqi_lag7        NUMERIC,

    -- Lag features: PM2.5 at t-1, t-2, t-3, t-7
    pm25_lag1       NUMERIC,
    pm25_lag2       NUMERIC,
    pm25_lag3       NUMERIC,
    pm25_lag7       NUMERIC,

    -- Lag features: PM10 at t-1, t-2, t-3, t-7
    pm10_lag1       NUMERIC,
    pm10_lag2       NUMERIC,
    pm10_lag3       NUMERIC,
    pm10_lag7       NUMERIC,

    -- Rolling statistics: 3-day window
    aqi_roll3_mean  NUMERIC,
    aqi_roll3_std   NUMERIC,
    aqi_roll3_max   NUMERIC,
    pm25_roll3_mean NUMERIC,

    -- Rolling statistics: 7-day window
    aqi_roll7_mean  NUMERIC,
    aqi_roll7_std   NUMERIC,
    aqi_roll7_max   NUMERIC,
    pm25_roll7_mean NUMERIC,

    -- Lag features: weather at t-1
    temp_lag1       NUMERIC,
    humidity_lag1   NUMERIC,
    wind_lag1       NUMERIC,
    rain_lag1       NUMERIC,

    -- Calendar features
    day_of_week     INTEGER,          -- 0=Monday … 6=Sunday
    month           INTEGER,          -- 1–12
    is_weekend      INTEGER,          -- 0 or 1
    dow_sin         NUMERIC,          -- cyclical encoding sin(2π*dow/7)
    dow_cos         NUMERIC,          -- cyclical encoding cos(2π*dow/7)
    month_sin       NUMERIC,          -- cyclical encoding sin(2π*month/12)
    month_cos       NUMERIC,          -- cyclical encoding cos(2π*month/12)

    -- Targets (what we are predicting; NULL during inference)
    target_aqi_next_day      NUMERIC,  -- regression target
    target_category_next_day TEXT,     -- classification target

    created_at TIMESTAMPTZ DEFAULT now(),

    PRIMARY KEY (feature_date, city)
);

CREATE INDEX IF NOT EXISTS idx_features_city_date
    ON ml.features_daily (city, feature_date);

-- ──────────────────────────────────────────────────────────
-- ml.prediction_log
-- Every call to POST /predict writes one row here.
-- This powers:
--   (a) monitoring (compare predicted vs actual once actual arrives)
--   (b) operational metrics (latency, error rate)
--   (c) audit trail (who predicted what, when, with which model version)
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.prediction_log (
    pred_id         SERIAL      PRIMARY KEY,
    requested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    city            TEXT,
    target_date     DATE,               -- the date we are predicting FOR
    predicted_aqi   NUMERIC,            -- regression output
    predicted_category TEXT,            -- classification output
    actual_aqi      NUMERIC,            -- filled in later once the real value is known
    regressor_name  TEXT,
    regressor_version TEXT,
    classifier_name TEXT,
    classifier_version TEXT,
    latency_ms      NUMERIC,
    status          TEXT,               -- SUCCESS / ERROR
    error_msg       TEXT,
    input_features  JSONB               -- raw input payload (for audit/debugging)
);

CREATE INDEX IF NOT EXISTS idx_pred_log_city_date
    ON ml.prediction_log (city, target_date);

-- ──────────────────────────────────────────────────────────
-- ml.monitoring_metrics
-- Rolling monitoring results written by src/monitoring/monitor.py.
-- Stores one row per monitoring run with aggregated drift and
-- performance metrics so trends can be plotted over time.
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.monitoring_metrics (
    check_id            SERIAL      PRIMARY KEY,
    checked_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_start        DATE,
    window_end          DATE,
    city                TEXT,

    -- Data quality
    missing_rate        NUMERIC,    -- fraction of expected records missing
    stale_data_flag     BOOLEAN,    -- true if no new data in STALE_DATA_HOURS

    -- Feature drift
    n_drifted_features  INTEGER,    -- how many features flagged by PSI/KS
    drift_triggered     BOOLEAN,    -- true if above DRIFT_FEATURE_COUNT_THRESHOLD
    drift_details       JSONB,      -- per-feature PSI/KS results

    -- Model performance (regression)
    rolling_rmse        NUMERIC,
    rolling_mae         NUMERIC,
    baseline_rmse       NUMERIC,    -- test-set baseline from training run
    rmse_pct_change     NUMERIC,    -- (rolling - baseline) / baseline

    -- Model performance (classification)
    rolling_f1          NUMERIC,

    -- Retraining decision
    retrain_triggered   BOOLEAN,
    retrain_reason      TEXT        -- human-readable explanation
);
