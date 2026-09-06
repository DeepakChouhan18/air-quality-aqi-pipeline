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
