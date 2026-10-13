from src.transformation.create_analytics import (
    pm25_to_aqi,
    aqi_to_category,
    build_daily_city_aqi,
    build_station_summary,
    build_pollutant_summary,
    filter_recent_records,
)
import pandas as pd


def test_pm25_to_aqi_good_range():
    # 6.0 ug/m3 sits mid-way in the "Good" breakpoint band (0-12 -> AQI 0-50)
    aqi = pm25_to_aqi(6.0)
    assert 0 <= aqi <= 50


def test_pm25_to_aqi_moderate_range():
    aqi = pm25_to_aqi(20.0)
    assert 51 <= aqi <= 100


def test_pm25_to_aqi_negative_returns_none():
    assert pm25_to_aqi(-5) is None


def test_aqi_to_category_boundaries():
    assert aqi_to_category(25) == "Good"
    assert aqi_to_category(75) == "Moderate"
    assert aqi_to_category(175) == "Unhealthy"
    assert aqi_to_category(400) == "Hazardous"


def test_build_daily_city_aqi_drops_stale_records():
    # Sensor 1 from 2018 (stale) vs Sensor 2 from 2026 (current)
    pollutant_df = pd.DataFrame([
        {"station_id": 1, "parameter": "pm25", "value": 20.0, "recorded_at": "2018-02-22 00:00:00+00:00"},
        {"station_id": 2, "parameter": "pm25", "value": 25.0, "recorded_at": "2026-09-06 00:00:00+00:00"},
    ])
    weather_df = pd.DataFrame([
        {"recorded_at": "2026-09-06 00:00:00", "temperature": 25.0, "humidity": 70, "wind_speed": 10, "precipitation": 0},
    ])
    daily = build_daily_city_aqi(pollutant_df, weather_df)
    assert len(daily) == 1
    assert str(daily.iloc[0]["date"]) == "2026-09-06"


def test_build_pollutant_summary_drops_stale_records():
    pollutant_df = pd.DataFrame([
        {"station_id": 1, "parameter": "co", "value": 330.0, "recorded_at": "2018-02-21 00:00:00+00:00"},
        {"station_id": 2, "parameter": "co", "value": 0.5, "recorded_at": "2026-09-06 00:00:00+00:00"},
    ])
    summary = build_pollutant_summary(pollutant_df)
    assert len(summary) == 1
    assert str(summary.iloc[0]["date"]) == "2026-09-06"
    assert summary.iloc[0]["parameter"] == "co"


def test_build_station_summary_drops_stale_records():
    pollutant_df = pd.DataFrame([
        {"station_id": 412, "parameter": "pm25", "value": 43.79, "recorded_at": "2018-02-22 00:00:00+00:00"},
        {"station_id": 5548, "parameter": "pm25", "value": 20.0, "recorded_at": "2026-09-06 00:00:00+00:00"},
    ])
    summary = build_station_summary(pollutant_df)
    assert len(summary) == 1
    assert summary.iloc[0]["station_id"] == 5548


def test_filter_recent_records_handles_empty():
    empty_df = pd.DataFrame()
    res = filter_recent_records(empty_df)
    assert res.empty


