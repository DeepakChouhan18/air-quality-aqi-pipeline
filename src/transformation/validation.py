"""
Data-quality validation for cleaned air-quality records.

Rules implemented here are documented in docs/validation_rules.md —
never invent a threshold without writing down why it exists.

Each function returns (valid_df, rejected_df) so nothing is silently
dropped: rejected rows always carry a reason and get written to
data/rejected/ by the calling script.
"""

import pandas as pd

from src.config import POLLUTANT_MAX_VALID

REQUIRED_COLUMNS = ["station_id", "parameter", "value", "unit", "recorded_at"]


def check_missing_required_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject rows missing any required field."""
    mask_valid = df[REQUIRED_COLUMNS].notna().all(axis=1)
    valid = df[mask_valid].copy()
    rejected = df[~mask_valid].copy()
    rejected["rejection_reason"] = "missing_required_field"
    return valid, rejected


def check_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reject duplicate records for the same station + parameter + timestamp.
    Keeps the first occurrence, flags the rest.
    """
    key_cols = ["station_id", "parameter", "recorded_at"]
    is_dup = df.duplicated(subset=key_cols, keep="first")
    valid = df[~is_dup].copy()
    rejected = df[is_dup].copy()
    rejected["rejection_reason"] = "duplicate_record"
    return valid, rejected


def check_value_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reject physically implausible pollutant values (negative, or above
    the documented max for that pollutant). Thresholds come from
    src/config.py::POLLUTANT_MAX_VALID — see docs/validation_rules.md
    for justification.
    """
    def is_valid_row(row):
        if row["value"] < 0:
            return False
        max_valid = POLLUTANT_MAX_VALID.get(str(row["parameter"]).lower())
        if max_valid is not None and row["value"] > max_valid:
            return False
        return True

    mask_valid = df.apply(is_valid_row, axis=1)
    valid = df[mask_valid].copy()
    rejected = df[~mask_valid].copy()
    rejected["rejection_reason"] = "value_out_of_range"
    return valid, rejected


def check_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject rows whose recorded_at cannot be parsed as a valid datetime."""
    parsed = pd.to_datetime(df["recorded_at"], errors="coerce", utc=True)
    mask_valid = parsed.notna()
    valid = df[mask_valid].copy()
    valid["recorded_at"] = parsed[mask_valid]
    rejected = df[~mask_valid].copy()
    rejected["rejection_reason"] = "invalid_timestamp"
    return valid, rejected


def run_all_validations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run every validation rule in sequence. A row is rejected the moment
    it fails ANY rule; all rejected rows (with reasons) are collected
    and returned alongside the final valid set.
    """
    all_rejected = []

    df, rejected = check_missing_required_fields(df)
    all_rejected.append(rejected)

    df, rejected = check_timestamps(df)
    all_rejected.append(rejected)

    df, rejected = check_duplicates(df)
    all_rejected.append(rejected)

    df, rejected = check_value_ranges(df)
    all_rejected.append(rejected)

    rejected_df = pd.concat(all_rejected, ignore_index=True) if all_rejected else pd.DataFrame()
    return df, rejected_df
