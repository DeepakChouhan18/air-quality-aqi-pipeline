# Data Dictionary

## core.stations
| Column | Type | Description | Source | Nullable |
|---|---|---|---|---|
| station_id | TEXT | Unique OpenAQ location identifier | OpenAQ | No (PK) |
| station_name | TEXT | Human-readable station name | OpenAQ | Yes |
| city | TEXT | City the station belongs to | OpenAQ | No |
| latitude | NUMERIC | Station latitude | OpenAQ | Yes |
| longitude | NUMERIC | Station longitude | OpenAQ | Yes |

## core.air_quality
| Column | Type | Description | Source | Unit | Nullable |
|---|---|---|---|---|---|
| measurement_id | SERIAL | Surrogate key | Generated | — | No (PK) |
| station_id | TEXT | FK to core.stations | OpenAQ | — | No |
| parameter | TEXT | Pollutant name (pm25, pm10, no2, so2, co) | OpenAQ | — | No |
| value | NUMERIC | Measured concentration | OpenAQ | Varies by parameter | No |
| unit | TEXT | Unit of measurement | OpenAQ | — | Yes |
| recorded_at | TIMESTAMPTZ | Timestamp of the measurement (UTC) | OpenAQ | — | No |
| loaded_at | TIMESTAMPTZ | When this row was loaded into Postgres | Generated | — | No |

## core.weather
| Column | Type | Description | Source | Unit | Nullable |
|---|---|---|---|---|---|
| weather_id | SERIAL | Surrogate key | Generated | — | No (PK) |
| city | TEXT | City the reading applies to | Open-Meteo | — | No |
| recorded_at | TIMESTAMPTZ | Timestamp of the reading (UTC) | Open-Meteo | — | No |
| temperature | NUMERIC | Air temperature at 2m | Open-Meteo | °C | Yes |
| humidity | NUMERIC | Relative humidity at 2m | Open-Meteo | % | Yes |
| precipitation | NUMERIC | Precipitation | Open-Meteo | mm | Yes |
| wind_speed | NUMERIC | Wind speed at 10m | Open-Meteo | km/h | Yes |

## analytics.daily_aqi
| Column | Type | Description | Nullable |
|---|---|---|---|
| date | DATE | Calendar date | No (PK) |
| city | TEXT | City | No (PK) |
| avg_pm25 | NUMERIC | Mean daily PM2.5 concentration | Yes |
| avg_pm10 | NUMERIC | Mean daily PM10 concentration | Yes |
| aqi_value | NUMERIC | US EPA AQI computed from avg_pm25 | Yes |
| aqi_category | TEXT | AQI category label (Good/Moderate/etc.) | Yes |
| avg_temperature | NUMERIC | Mean daily temperature | Yes |
| avg_humidity | NUMERIC | Mean daily humidity | Yes |
| avg_wind_speed | NUMERIC | Mean daily wind speed | Yes |
| total_precipitation | NUMERIC | Total daily precipitation | Yes |

## staging.pipeline_runs (lineage/audit table)
| Column | Type | Description |
|---|---|---|
| run_id | UUID | Unique ID for this pipeline run stage |
| source | TEXT | Data source (openaq / open-meteo / analytics) |
| stage | TEXT | Pipeline stage (extract / transform / load) |
| started_at | TIMESTAMPTZ | Run start time |
| finished_at | TIMESTAMPTZ | Run end time |
| status | TEXT | RUNNING / SUCCESS / FAILED |
| rows_extracted | INTEGER | Rows pulled from source |
| rows_valid | INTEGER | Rows that passed validation |
| rows_rejected | INTEGER | Rows that failed validation |
| error_message | TEXT | Error detail if status = FAILED |
