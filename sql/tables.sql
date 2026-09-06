-- CORE LAYER TABLES
-- Clean, deduplicated, relational data — the "single source of truth"
-- that both analytics tables and future ML feature pipelines read from.

CREATE TABLE IF NOT EXISTS core.stations (
    station_id TEXT PRIMARY KEY,
    station_name TEXT,
    city TEXT NOT NULL,
    latitude NUMERIC,
    longitude NUMERIC
);

CREATE TABLE IF NOT EXISTS core.air_quality (
    measurement_id SERIAL PRIMARY KEY,
    -- station_id is a plain TEXT column (no FK to core.stations) because the
    -- pipeline does not yet populate core.stations. Adding a FK without a
    -- matching insert step causes every measurement INSERT to fail with a
    -- ForeignKeyViolation. The FK can be re-added once a station-loading task
    -- is implemented (planned for pipeline v2).
    station_id TEXT,
    parameter TEXT NOT NULL,
    value NUMERIC NOT NULL,
    unit TEXT,
    recorded_at TIMESTAMPTZ NOT NULL,
    loaded_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (station_id, parameter, recorded_at)
);

CREATE TABLE IF NOT EXISTS core.weather (
    weather_id SERIAL PRIMARY KEY,
    city TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    temperature NUMERIC,
    humidity NUMERIC,
    precipitation NUMERIC,
    wind_speed NUMERIC,
    loaded_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (city, recorded_at)
);

CREATE INDEX IF NOT EXISTS idx_air_quality_station_time
    ON core.air_quality (station_id, recorded_at);

CREATE INDEX IF NOT EXISTS idx_weather_city_time
    ON core.weather (city, recorded_at);
