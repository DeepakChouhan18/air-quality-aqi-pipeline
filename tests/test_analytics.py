"""Tests for the AQI calculation (US EPA breakpoint methodology)."""

from src.transformation.create_analytics import pm25_to_aqi, aqi_to_category


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
