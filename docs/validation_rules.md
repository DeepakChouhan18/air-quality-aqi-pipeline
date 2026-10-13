# Data Validation Rules

Every rule below is implemented in `src/transformation/validation.py`
and covered by tests in `tests/test_data_quality.py`. Nothing is
silently dropped — every rejected row is written to `data/rejected/`
(and `staging.rejected_records` in Postgres) with a `rejection_reason`.

## 1. Missing required fields
**Rule:** `station_id`, `parameter`, `value`, `unit`, `recorded_at` must all be present.
**Why:** These are the minimum fields needed to identify what was measured, where, and when. A row missing any of them can't be meaningfully analyzed.

## 2. Duplicate records
**Rule:** Reject duplicates on `(station_id, parameter, recorded_at)`, keeping the first occurrence.
**Why:** The same station can appear in more than one API response (e.g., overlapping pagination, re-running a failed job). Without deduplication, daily averages would be skewed by double-counted readings.

## 3. Value range checks
**Rule:** Reject negative values and values above a documented per-pollutant maximum (see `POLLUTANT_MAX_VALID` in `src/config.py`).
**Why:** Pollutant concentrations cannot physically be negative. Extremely high values are far more likely to be sensor faults or unit errors than real readings. Thresholds are set generously above realistic worst-case ambient air pollution levels — no arbitrary tight limits.

## 4. Timestamp validation
**Rule:** `recorded_at` must parse as a valid datetime.
**Why:** An unparseable timestamp makes a row impossible to place in the time series correctly; it can't be joined to weather data or aggregated by date.

## 5. Missing periods (documented, checked at the analytics stage)
**Approach:** When building `analytics.daily_aqi`, dates with no corresponding pollutant readings simply won't appear in the output — this is treated as a known data gap rather than back-filled or estimated, to avoid presenting synthetic data as real.

## 6. Station validation
**Rule:** Station metadata (`core.stations`) is loaded from OpenAQ's `/locations` endpoint, which provides `location_id`, `name`, `city`, and coordinates directly — no separate validation needed beyond the standard missing-field check.

## 7. Stale sensor recency filtering (OpenAQ /latest endpoint)
**Known Characteristic:** OpenAQ's `/latest` measurements endpoint returns the last recorded reading for every sensor associated with a location. Inactive or decommissioned monitoring stations still report their last historical measurement, which may be months or years old (e.g., readings from 2018 or 2022 appearing alongside current 2026 data).
**Mitigation Rule:** In `src/transformation/create_analytics.py` (`filter_recent_records`), before grouping or aggregating records by date, pollutant or weather measurements whose `recorded_at` timestamp is older than 30 days prior to the most recent `recorded_at` in the dataset are filtered out. This is applied at the entry of the analytics pipeline (in `src/export/export_for_tableau.py` and `src/database/load_postgres.py`) and inside each individual analytics builder (`build_daily_city_aqi`, `build_pollutant_summary`, and `build_station_summary`) for full consistency.
**Why:** Grouping multi-year historical artifacts into current daily time series skews longitudinal trends and misrepresents city-wide air quality indices. A 30-day rolling recency window guarantees that analytics reflect active, synchronized environmental observations across all views (Daily AQI, Pollutant Analysis, and Station Comparison).

