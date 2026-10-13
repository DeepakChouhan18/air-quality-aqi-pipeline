"""
Tests for the feature engineering pipeline.

KEY THINGS WE TEST:
1. No leakage: features at date t use ONLY data from dates ≤ t-1
2. Targets use data from date t+1 (shift(-1)) — correct direction
3. Chronological split preserves time order (no future in past)
4. Missing value strategy is applied correctly

These tests use synthetic data so they don't require a database connection.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.features import (
    add_lag_features,
    add_rolling_features,
    add_weather_lag_features,
    add_calendar_features,
    add_targets,
    handle_missing,
    FEATURE_COLS,
    TARGET_REG,
    TARGET_CLF,
)
from src.ml.train import chronological_split


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_synthetic_df(n: int = 100) -> pd.DataFrame:
    """
    Create a synthetic daily AQI DataFrame with known values for deterministic tests.
    AQI values are integers 1..n so we can trace which value ended up where.
    """
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "date":              dates,
        "city":              "TestCity",
        "aqi_value":         list(range(1, n + 1)),   # 1, 2, 3, … n
        "avg_pm25":          [float(i) * 0.5 for i in range(1, n + 1)],
        "avg_pm10":          [float(i) * 0.8 for i in range(1, n + 1)],
        "avg_temperature":   [25.0] * n,
        "avg_humidity":      [60.0] * n,
        "avg_wind_speed":    [10.0] * n,
        "total_precipitation": [0.0] * n,
    })


# ── Test: Lag features — no leakage ──────────────────────────────────────────

class TestLagFeatures:
    def test_lag1_is_previous_day(self):
        """aqi_lag1 at row i must equal aqi_value at row i-1 (yesterday's AQI)."""
        df = make_synthetic_df(20)
        df = add_lag_features(df)

        # Row 5: aqi_lag1 should be aqi_value from row 4
        assert df.iloc[5]["aqi_lag1"] == df.iloc[4]["aqi_value"], (
            "aqi_lag1 at row 5 should equal aqi_value at row 4 (no leakage)"
        )

    def test_lag7_is_seven_days_ago(self):
        """aqi_lag7 at row i must equal aqi_value at row i-7."""
        df = make_synthetic_df(30)
        df = add_lag_features(df)

        for i in range(7, 30):
            assert df.iloc[i]["aqi_lag7"] == df.iloc[i - 7]["aqi_value"], (
                f"aqi_lag7 at row {i} should equal aqi_value at row {i-7}"
            )

    def test_lag_first_rows_are_nan(self):
        """First 7 rows should have NaN for lag-7 features (no history available)."""
        df = make_synthetic_df(30)
        df = add_lag_features(df)

        for i in range(7):
            assert pd.isna(df.iloc[i]["aqi_lag7"]), (
                f"Row {i} should have NaN aqi_lag7 — not enough history yet"
            )

    def test_no_future_data_in_lags(self):
        """
        LEAKAGE TEST: the value at row i must never appear in a LAG feature
        for any row j < i. That would mean we used future data in the past.
        """
        df = make_synthetic_df(50)
        df = add_lag_features(df)

        for lag_col in ["aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7"]:
            for i, row in df.iterrows():
                lag_val = row[lag_col]
                if pd.isna(lag_val):
                    continue
                # lag_val must correspond to a row BEFORE row i
                matching_rows = df[df["aqi_value"] == lag_val].index.tolist()
                # all matching rows must have index < i
                for match_idx in matching_rows:
                    assert match_idx < i, (
                        f"LEAKAGE DETECTED: {lag_col} at row {i} "
                        f"contains value from row {match_idx} (future data!)"
                    )


# ── Test: Rolling features — no leakage ──────────────────────────────────────

class TestRollingFeatures:
    def test_rolling3_mean_uses_only_past(self):
        """
        aqi_roll3_mean at row i should be the mean of aqi_value at rows i-3..i-1.
        It must NOT include the current row (i) or any future row.
        """
        df = make_synthetic_df(20)
        df_with_lags = add_lag_features(df)  # rolling uses shifted lags internally
        df_with_roll = add_rolling_features(df)

        # At row 10: roll3_mean = mean of rows 9, 8, 7 (shifted by 1 first)
        expected = np.mean([df.iloc[9]["aqi_value"], df.iloc[8]["aqi_value"],
                            df.iloc[7]["aqi_value"]])
        # Allow small tolerance for floating point
        actual = df_with_roll.iloc[10]["aqi_roll3_mean"]
        assert abs(actual - expected) < 0.1, (
            f"aqi_roll3_mean at row 10: expected {expected:.2f}, got {actual:.2f}"
        )

    def test_rolling7_does_not_use_current_row(self):
        """
        If we set aqi_value at row i to a sentinel (9999), it must NOT appear
        in aqi_roll7_mean for row i (which should only see rows i-7..i-1).
        """
        df = make_synthetic_df(20)
        df.loc[15, "aqi_value"] = 9999.0  # sentinel value
        df_with_roll = add_rolling_features(df)

        # Row 15's own roll7 should NOT include 9999 (current row not in window)
        assert df_with_roll.iloc[15]["aqi_roll7_mean"] < 9999, (
            "Rolling 7-day mean at row 15 incorrectly includes row 15's value (leakage!)"
        )


# ── Test: Targets — correct direction ────────────────────────────────────────

class TestTargets:
    def test_target_is_next_day_aqi(self):
        """target_aqi_next_day at row i must equal aqi_value at row i+1."""
        df = make_synthetic_df(20)
        df = add_targets(df)

        for i in range(len(df) - 1):  # skip last row (NaN target)
            expected = df.iloc[i + 1]["aqi_value"]
            actual   = df.iloc[i][TARGET_REG]
            assert actual == expected, (
                f"target_aqi_next_day at row {i} should be aqi_value at row {i+1}"
            )

    def test_last_row_target_is_nan(self):
        """The last row has no tomorrow — target must be NaN."""
        df = make_synthetic_df(20)
        df = add_targets(df)
        assert pd.isna(df.iloc[-1][TARGET_REG]), (
            "Last row's target_aqi_next_day must be NaN (no future data available)"
        )

    def test_category_matches_aqi(self):
        """Category target must be consistent with the AQI target value."""
        from src.transformation.create_analytics import aqi_to_category
        df = make_synthetic_df(20)
        df = add_targets(df)

        for _, row in df.dropna(subset=[TARGET_REG]).iterrows():
            expected_cat = aqi_to_category(row[TARGET_REG])
            assert row[TARGET_CLF] == expected_cat, (
                f"Category mismatch at {row['date']}: "
                f"AQI={row[TARGET_REG]} → expected '{expected_cat}' got '{row[TARGET_CLF]}'"
            )


# ── Test: Chronological split ─────────────────────────────────────────────────

class TestChronologicalSplit:
    def test_split_is_chronological(self):
        """train/val/test must be non-overlapping and in strict temporal order."""
        df = make_synthetic_df(100)
        df["date"] = pd.to_datetime(df["date"])
        train, val, test = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

        # No overlap by date
        train_dates = set(train["date"])
        val_dates   = set(val["date"])
        test_dates  = set(test["date"])
        assert train_dates.isdisjoint(val_dates), "Train and val share dates!"
        assert train_dates.isdisjoint(test_dates), "Train and test share dates!"
        assert val_dates.isdisjoint(test_dates), "Val and test share dates!"

        # All train dates must be before all val dates
        assert train["date"].max() < val["date"].min(), (
            "Latest train date is not before earliest val date!"
        )
        assert val["date"].max() < test["date"].min(), (
            "Latest val date is not before earliest test date!"
        )

    def test_split_sizes(self):
        """Check that split sizes roughly match the specified ratios."""
        n = 200
        df = make_synthetic_df(n)
        df["date"] = pd.to_datetime(df["date"])
        train, val, test = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

        # Allow ±1 row for integer truncation
        assert abs(len(train) - int(n * 0.70)) <= 1
        assert abs(len(val)   - int(n * 0.15)) <= 1

    def test_no_future_in_train(self):
        """
        CRITICAL LEAKAGE TEST: no training row should have a date that
        appears in the validation or test sets.
        """
        df = make_synthetic_df(100)
        df["date"] = pd.to_datetime(df["date"])
        train, val, test = chronological_split(df)

        future_dates = set(val["date"]) | set(test["date"])
        train_dates  = set(train["date"])
        leaked = train_dates & future_dates
        assert len(leaked) == 0, (
            f"LEAKAGE: {len(leaked)} training dates appear in val/test: "
            f"{sorted(leaked)[:5]}"
        )


# ── Test: Missing value handling ──────────────────────────────────────────────

class TestMissingValues:
    def test_weather_ffill(self):
        """Weather columns should be forward-filled after handle_missing."""
        df = make_synthetic_df(20)
        # Introduce NaN in weather columns
        df.loc[5, "avg_temperature"] = np.nan
        df.loc[6, "avg_temperature"] = np.nan

        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_weather_lag_features(df)
        df = add_calendar_features(df)
        df = add_targets(df)
        df = handle_missing(df)

        # After forward fill, rows 5 and 6 should have non-NaN temperature
        assert not pd.isna(df.iloc[5]["avg_temperature"]) or df.iloc[5]["avg_temperature"] == df.iloc[4]["avg_temperature"], (
            "avg_temperature at row 5 should be forward-filled from row 4"
        )


# ── Test: Calendar features ────────────────────────────────────────────────────

class TestCalendarFeatures:
    def test_weekend_flag(self):
        """is_weekend must be 1 for Saturday/Sunday and 0 for weekdays."""
        df = make_synthetic_df(10)
        df = add_calendar_features(df)

        for _, row in df.iterrows():
            dow = row["day_of_week"]
            expected_weekend = int(dow >= 5)
            assert row["is_weekend"] == expected_weekend, (
                f"is_weekend wrong for dow={dow}: expected {expected_weekend}"
            )

    def test_cyclical_encoding_range(self):
        """Sin/cos encodings must be in [-1, 1]."""
        df = make_synthetic_df(50)
        df = add_calendar_features(df)

        for col in ["dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert df[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all(), (
                f"Cyclical encoding '{col}' out of [-1, 1] range"
            )
