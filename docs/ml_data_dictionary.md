# ML Data Dictionary

Defines every feature in `ml.features_daily` and how it is computed.
This is the canonical reference for the feature pipeline (`src/ml/features.py`).

## Feature Version: v1

> **Important**: Changing any feature definition requires bumping `FEATURE_VERSION` in
> `src/ml/features.py` and retraining from scratch. Old models trained on v1 features
> are incompatible with v2 features.

## Raw Daily Values (from analytics.daily_aqi)

| Feature | Type | Source column | Description |
|---|---|---|---|
| `aqi_value` | float | `analytics.daily_aqi.aqi_value` | US EPA AQI computed from PM2.5 (0–500) |
| `avg_pm25` | float | `analytics.daily_aqi.avg_pm25` | Mean daily PM2.5 (µg/m³) |
| `avg_pm10` | float | `analytics.daily_aqi.avg_pm10` | Mean daily PM10 (µg/m³) |
| `avg_temperature` | float | `analytics.daily_aqi.avg_temperature` | Mean daily temperature at 2m (°C) |
| `avg_humidity` | float | `analytics.daily_aqi.avg_humidity` | Mean daily relative humidity at 2m (%) |
| `avg_wind_speed` | float | `analytics.daily_aqi.avg_wind_speed` | Mean daily wind speed at 10m (km/h) |
| `total_precipitation` | float | `analytics.daily_aqi.total_precipitation` | Total daily precipitation (mm) |

## Lag Features

Lag features are computed using `pandas.Series.shift(n)` where n ≥ 1.
This guarantees that at date t, the feature reflects data from date t-n,
never from t itself or any future date (no leakage).

### AQI Lags

| Feature | Formula | Description |
|---|---|---|
| `aqi_lag1` | `aqi_value.shift(1)` | Yesterday's AQI |
| `aqi_lag2` | `aqi_value.shift(2)` | AQI two days ago |
| `aqi_lag3` | `aqi_value.shift(3)` | AQI three days ago |
| `aqi_lag7` | `aqi_value.shift(7)` | AQI one week ago |

### PM2.5 Lags

| Feature | Formula | Description |
|---|---|---|
| `pm25_lag1` | `avg_pm25.shift(1)` | Yesterday's PM2.5 |
| `pm25_lag2` | `avg_pm25.shift(2)` | PM2.5 two days ago |
| `pm25_lag3` | `avg_pm25.shift(3)` | PM2.5 three days ago |
| `pm25_lag7` | `avg_pm25.shift(7)` | PM2.5 one week ago |

### PM10 Lags

| Feature | Formula | Description |
|---|---|---|
| `pm10_lag1` | `avg_pm10.shift(1)` | Yesterday's PM10 |
| `pm10_lag2` | `avg_pm10.shift(2)` | PM10 two days ago |
| `pm10_lag3` | `avg_pm10.shift(3)` | PM10 three days ago |
| `pm10_lag7` | `avg_pm10.shift(7)` | PM10 one week ago |

## Rolling Features

Computed using `shift(1).rolling(window=W)` — the `shift(1)` ensures the
window closes before the current day, preventing any look-ahead.

| Feature | Formula | Description |
|---|---|---|
| `aqi_roll3_mean` | `aqi.shift(1).rolling(3).mean()` | 3-day mean AQI |
| `aqi_roll3_std` | `aqi.shift(1).rolling(3).std()` | 3-day AQI standard deviation |
| `aqi_roll3_max` | `aqi.shift(1).rolling(3).max()` | 3-day maximum AQI |
| `pm25_roll3_mean` | `pm25.shift(1).rolling(3).mean()` | 3-day mean PM2.5 |
| `aqi_roll7_mean` | `aqi.shift(1).rolling(7).mean()` | 7-day mean AQI |
| `aqi_roll7_std` | `aqi.shift(1).rolling(7).std()` | 7-day AQI standard deviation |
| `aqi_roll7_max` | `aqi.shift(1).rolling(7).max()` | 7-day maximum AQI |
| `pm25_roll7_mean` | `pm25.shift(1).rolling(7).mean()` | 7-day mean PM2.5 |

## Weather Lag Features

Previous-day weather is a useful predictor because atmospheric conditions
(temperature inversions, low wind, high humidity) that trap pollutants tend
to persist for multiple days.

| Feature | Formula | Description |
|---|---|---|
| `temp_lag1` | `avg_temperature.shift(1)` | Yesterday's temperature (°C) |
| `humidity_lag1` | `avg_humidity.shift(1)` | Yesterday's humidity (%) |
| `wind_lag1` | `avg_wind_speed.shift(1)` | Yesterday's wind speed (km/h) |
| `rain_lag1` | `total_precipitation.shift(1)` | Yesterday's precipitation (mm) |

## Calendar Features

| Feature | Formula | Description |
|---|---|---|
| `day_of_week` | `date.dayofweek` | 0 = Monday … 6 = Sunday |
| `month` | `date.month` | 1–12 |
| `is_weekend` | `int(day_of_week >= 5)` | 1 if Saturday or Sunday |
| `dow_sin` | `sin(2π × day_of_week / 7)` | Cyclical encoding of day-of-week |
| `dow_cos` | `cos(2π × day_of_week / 7)` | Cyclical encoding of day-of-week |
| `month_sin` | `sin(2π × month / 12)` | Cyclical encoding of month |
| `month_cos` | `cos(2π × month / 12)` | Cyclical encoding of month |

> **Why cyclical encoding?** Day 0 (Monday) and Day 6 (Sunday) are numerically
> far apart but temporally adjacent. Cyclical encoding maps both ends of the
> weekly/annual cycle onto a unit circle so linear models learn the correct
> periodicity without needing polynomial expansion.

## Targets

| Column | Type | Description |
|---|---|---|
| `target_aqi_next_day` | float | AQI for the next calendar day — **regression target** |
| `target_category_next_day` | str | AQI category for the next day — **classification target** |

Both targets are computed as `aqi_value.shift(-1)`, meaning at date t the
target is the AQI observed at date t+1. The last row of the feature table
always has NULL targets (we cannot know the future).

### AQI Category Bins (US EPA, same as Part 1)

| AQI Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301–500 | Hazardous |

## Missing Value Strategy

| Case | Strategy | Reason |
|---|---|---|
| First 7 days (no lookback for lag-7) | Rows dropped at training time | Cannot fabricate historical data |
| Weather columns NaN | Forward-fill (ffill) | Weather doesn't jump wildly hour-to-hour |
| AQI/PM NaN | Left as NaN, imputed by model pipeline (median) | Imputing means would shift the distribution |
| Last row (no next-day target) | Dropped at training, kept at inference | We never know the future |
