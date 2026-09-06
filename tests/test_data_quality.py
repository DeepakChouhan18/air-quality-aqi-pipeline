"""
Tests for validation.py — these are the CORE tests your assignment
rubric expects. Run with: pytest tests/test_data_quality.py -v
"""

import pandas as pd
import pytest

from src.transformation.validation import (
    check_missing_required_fields,
    check_duplicates,
    check_value_ranges,
    check_timestamps,
    run_all_validations,
)


def make_df(rows):
    return pd.DataFrame(rows)


def test_missing_required_fields_are_rejected():
    df = make_df([
        {"station_id": "S1", "parameter": "pm25", "value": 50, "unit": "ug/m3", "recorded_at": "2026-09-01T00:00:00Z"},
        {"station_id": None, "parameter": "pm25", "value": 50, "unit": "ug/m3", "recorded_at": "2026-09-01T01:00:00Z"},
    ])
    valid, rejected = check_missing_required_fields(df)
    assert len(valid) == 1
    assert len(rejected) == 1
    assert rejected.iloc[0]["rejection_reason"] == "missing_required_field"


def test_duplicates_are_rejected():
    df = make_df([
        {"station_id": "S1", "parameter": "pm25", "value": 50, "unit": "ug/m3", "recorded_at": "2026-09-01T00:00:00Z"},
        {"station_id": "S1", "parameter": "pm25", "value": 55, "unit": "ug/m3", "recorded_at": "2026-09-01T00:00:00Z"},
    ])
    valid, rejected = check_duplicates(df)
    assert len(valid) == 1
    assert len(rejected) == 1


def test_negative_values_are_rejected():
    df = make_df([
        {"station_id": "S1", "parameter": "pm25", "value": -5, "unit": "ug/m3", "recorded_at": "2026-09-01T00:00:00Z"},
    ])
    valid, rejected = check_value_ranges(df)
    assert len(valid) == 0
    assert rejected.iloc[0]["rejection_reason"] == "value_out_of_range"


def test_extreme_values_above_pollutant_max_are_rejected():
    df = make_df([
        {"station_id": "S1", "parameter": "pm25", "value": 5000, "unit": "ug/m3", "recorded_at": "2026-09-01T00:00:00Z"},
    ])
    valid, rejected = check_value_ranges(df)
    assert len(valid) == 0


def test_invalid_timestamp_is_rejected():
    df = make_df([
        {"station_id": "S1", "parameter": "pm25", "value": 50, "unit": "ug/m3", "recorded_at": "not-a-date"},
    ])
    valid, rejected = check_timestamps(df)
    assert len(valid) == 0
    assert rejected.iloc[0]["rejection_reason"] == "invalid_timestamp"


def test_run_all_validations_end_to_end():
    df = make_df([
        {"station_id": "S1", "parameter": "pm25", "value": 50, "unit": "ug/m3", "recorded_at": "2026-09-01T00:00:00Z"},
        {"station_id": "S1", "parameter": "pm25", "value": 50, "unit": "ug/m3", "recorded_at": "2026-09-01T00:00:00Z"},  # dup
        {"station_id": None, "parameter": "pm25", "value": 50, "unit": "ug/m3", "recorded_at": "2026-09-01T01:00:00Z"},  # missing
        {"station_id": "S2", "parameter": "pm25", "value": -1, "unit": "ug/m3", "recorded_at": "2026-09-01T02:00:00Z"},  # negative
    ])
    valid, rejected = run_all_validations(df)
    assert len(valid) == 1
    assert len(rejected) == 3
