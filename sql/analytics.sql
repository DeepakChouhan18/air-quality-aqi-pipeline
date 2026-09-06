-- ANALYTICS (GOLD) LAYER
-- These tables are what Tableau connects to directly. Pre-aggregating
-- here keeps the dashboard fast and keeps business logic (AQI
-- calculation) in one place instead of duplicated in Tableau calc fields.

CREATE TABLE IF NOT EXISTS analytics.daily_aqi (
    date DATE NOT NULL,
    city TEXT NOT NULL,
    avg_pm25 NUMERIC,
    avg_pm10 NUMERIC,
    aqi_value NUMERIC,
    aqi_category TEXT,
    avg_temperature NUMERIC,
    avg_humidity NUMERIC,
    avg_wind_speed NUMERIC,
    total_precipitation NUMERIC,
    PRIMARY KEY (date, city)
);

CREATE TABLE IF NOT EXISTS analytics.station_summary (
    station_id TEXT NOT NULL,
    avg_pm25 NUMERIC,
    max_pm25 NUMERIC,
    min_pm25 NUMERIC,
    avg_aqi NUMERIC,
    PRIMARY KEY (station_id)
);

CREATE TABLE IF NOT EXISTS analytics.pollutant_summary (
    date DATE NOT NULL,
    parameter TEXT NOT NULL,
    avg_value NUMERIC,
    max_value NUMERIC,
    PRIMARY KEY (date, parameter)
);
