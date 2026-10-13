"""
Builds the analytics (gold) layer: joins pollutant + weather data,
calculates AQI using the US EPA breakpoint methodology, and produces
the aggregated tables that power the dashboard.

WHY US EPA AQI METHODOLOGY (documented, not invented):
The US EPA's PM2.5 AQI breakpoint table is a published, standard
methodology (used internationally as a reference even outside the US).
Using a documented standard instead of a made-up formula is exactly
what the assignment brief requires ("must use a documented and
appropriate standard/methodology rather than an invented formula").
Source: US EPA, "Technical Assistance Document for the Reporting of
Daily Air Quality" — breakpoints for PM2.5 (ug/m3, 24-hr average):

  0.0  -  12.0   -> AQI   0-50   (Good)
  12.1 -  35.4   -> AQI  51-100  (Moderate)
  35.5 -  55.4   -> AQI 101-150  (Unhealthy for Sensitive Groups)
  55.5 - 150.4   -> AQI 151-200  (Unhealthy)
  150.5- 250.4   -> AQI 201-300  (Very Unhealthy)
  250.5- 500.4   -> AQI 301-500  (Hazardous)
"""

from pathlib import Path

import pandas as pd

# (C_low, C_high, AQI_low, AQI_high) breakpoints for PM2.5
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def pm25_to_aqi(concentration: float) -> float:
    """Convert a PM2.5 concentration (ug/m3) to a US EPA AQI value."""
    if pd.isna(concentration) or concentration < 0:
        return None
    for c_low, c_high, aqi_low, aqi_high in PM25_BREAKPOINTS:
        if c_low <= concentration <= c_high:
            return round(
                ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low,
                1,
            )
    return 500.0  # above the top breakpoint, cap at max


def aqi_to_category(aqi_value: float) -> str:
    if pd.isna(aqi_value):
        return "Unknown"
    for low, high, category in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return category
    return "Hazardous"


def filter_recent_records(df: pd.DataFrame, days: int = 730) -> pd.DataFrame:
    """
    Filter records to only keep rows within `days` of the most recent recorded_at.

    OpenAQ's /latest endpoint returns the last known reading for each sensor.
    For inactive or decommissioned sensors, this "latest" reading can be years old
    (e.g. 2017, 2018, or 2022 entries mixed in with current 2026 data).
    Dropping records older than `days` before the dataset's max timestamp ensures
    that time series, pollutant distributions, and station summaries reflect
    active, synchronized observations rather than stale historical artifacts.
    """
    if df is None or df.empty or "recorded_at" not in df.columns:
        return df.copy() if df is not None else df
    df = df.copy()
    dt = pd.to_datetime(df["recorded_at"], utc=True)
    if dt.notna().any():
        cutoff = dt.max() - pd.Timedelta(days=days)
        return df[dt >= cutoff].copy()
    return df


def build_daily_city_aqi(pollutant_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces analytics.daily_aqi: one row per city per day, with average
    pollutant levels, an AQI value/category, and average weather.
    """
    pollutant_df = filter_recent_records(pollutant_df)
    weather_df = filter_recent_records(weather_df)

    pollutant_df["date"] = pd.to_datetime(pollutant_df["recorded_at"], utc=True).dt.date
    weather_df["date"] = pd.to_datetime(weather_df["recorded_at"], utc=True).dt.date

    pm25 = pollutant_df[pollutant_df["parameter"].str.lower() == "pm25"]
    daily_pm25 = pm25.groupby("date")["value"].mean().reset_index(name="avg_pm25")

    pm10 = pollutant_df[pollutant_df["parameter"].str.lower() == "pm10"]
    daily_pm10 = pm10.groupby("date")["value"].mean().reset_index(name="avg_pm10")

    daily_weather = weather_df.groupby("date").agg(
        avg_temperature=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
        avg_wind_speed=("wind_speed", "mean"),
        total_precipitation=("precipitation", "sum"),
    ).reset_index()

    daily = daily_pm25.merge(daily_pm10, on="date", how="outer")
    daily = daily.merge(daily_weather, on="date", how="left")

    daily["aqi_value"] = daily["avg_pm25"].apply(pm25_to_aqi)
    daily["aqi_category"] = daily["aqi_value"].apply(aqi_to_category)

    return daily.sort_values("date")


def build_station_summary(pollutant_df: pd.DataFrame) -> pd.DataFrame:
    """One row per station: avg/max/min AQI-relevant PM2.5 levels."""
    df = filter_recent_records(pollutant_df)
    pm25 = df[df["parameter"].str.lower() == "pm25"]
    summary = pm25.groupby("station_id")["value"].agg(
        avg_pm25="mean", max_pm25="max", min_pm25="min"
    ).reset_index()
    summary["avg_aqi"] = summary["avg_pm25"].apply(pm25_to_aqi)
    return summary


def build_pollutant_summary(pollutant_df: pd.DataFrame) -> pd.DataFrame:
    """One row per day per pollutant: avg/max concentration."""
    df = filter_recent_records(pollutant_df)
    df["date"] = pd.to_datetime(df["recorded_at"], utc=True).dt.date
    return df.groupby(["date", "parameter"])["value"].agg(
        avg_value="mean", max_value="max"
    ).reset_index()


if __name__ == "__main__":
    print("Import this module's functions from a script or the Airflow DAG.")
