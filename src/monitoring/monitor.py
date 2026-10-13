"""
Monitoring: data quality, feature drift, and model performance checks.

WHY MONITORING IS ESSENTIAL:
Production ML systems degrade silently. The input distribution shifts
(weather patterns change seasonally), the sensors drift, and the model's
accuracy erodes — all without raising any exceptions. Active monitoring
catches these issues before users notice bad predictions.

THREE MONITORING LAYERS:
1. Data quality: are inputs arriving and passing range checks?
2. Feature drift: has the input distribution shifted from training?
3. Model performance: has prediction accuracy degraded?

DRIFT DETECTION METHODS:
- PSI (Population Stability Index): measures shift in feature distributions.
  PSI < 0.1: no significant shift
  PSI 0.1–0.2: moderate shift (monitor closely)
  PSI > 0.2: significant drift (threshold in config → retrain trigger)

- KS test (Kolmogorov-Smirnov): non-parametric test comparing two
  distributions. p < 0.05 means the distributions are significantly
  different. Complements PSI (PSI is better for binned distributions,
  KS is better for small samples).

All thresholds are in src/config.py — never hardcoded here.
"""

import json
import logging
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from src.config import (
    TARGET_CITY, ML_DATA_DIR, REPORTS_DIR,
    PSI_DRIFT_THRESHOLD, KS_PVALUE_THRESHOLD, DRIFT_FEATURE_COUNT_THRESHOLD,
    MISSING_RATE_THRESHOLD, STALE_DATA_HOURS,
    RMSE_DEGRADATION_THRESHOLD,
)
from src.database.connection import get_engine
from src.ml.features import FEATURE_COLS
from src.ml.evaluate import evaluate_regression, evaluate_classification

logger = logging.getLogger(__name__)


# ── 1. Data Quality ──────────────────────────────────────────────────────────

def check_data_quality(
    city: str = TARGET_CITY,
    lookback_days: int = 7,
) -> dict:
    """
    Check data quality for the most recent `lookback_days` of ml.features_daily.

    Returns a dict with:
      - missing_rate: fraction of expected feature values that are NaN
      - stale_data_flag: True if no new data in the last STALE_DATA_HOURS hours
      - out_of_range_rows: count of rows with AQI < 0 or > 600 (physically implausible)
    """
    engine = get_engine()
    cutoff = date.today() - timedelta(days=lookback_days)

    query = text("""
        SELECT feature_date, aqi_value, avg_pm25, avg_pm10,
               avg_temperature, avg_humidity, avg_wind_speed
        FROM ml.features_daily
        WHERE city = :city AND feature_date >= :cutoff
        ORDER BY feature_date DESC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"city": city, "cutoff": cutoff})

    if df.empty:
        logger.warning(f"[monitor] No feature rows found for {city} since {cutoff}")
        return {
            "missing_rate": 1.0,
            "stale_data_flag": True,
            "out_of_range_rows": 0,
            "n_rows_checked": 0,
        }

    # Missing rate: fraction of feature cells that are NaN
    check_cols = ["aqi_value", "avg_pm25", "avg_pm10",
                  "avg_temperature", "avg_humidity", "avg_wind_speed"]
    n_total = len(df) * len(check_cols)
    n_missing = df[check_cols].isna().sum().sum()
    missing_rate = float(n_missing / max(n_total, 1))

    # Stale data check: is the most recent row within STALE_DATA_HOURS?
    most_recent = pd.to_datetime(df["feature_date"].max())
    hours_since = (datetime.now() - most_recent.to_pydatetime()).total_seconds() / 3600
    stale_flag = hours_since > STALE_DATA_HOURS

    # Out-of-range AQI (physically implausible)
    oor = int((df["aqi_value"] < 0).sum() + (df["aqi_value"] > 600).sum())

    result = {
        "missing_rate": missing_rate,
        "stale_data_flag": stale_flag,
        "hours_since_latest_data": round(hours_since, 1),
        "out_of_range_rows": oor,
        "n_rows_checked": len(df),
    }
    logger.info(f"[monitor] Data quality: {result}")
    return result


# ── 2. Feature Drift ──────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between reference and production arrays.

    PSI = sum((P_actual - P_expected) * ln(P_actual / P_expected))

    Uses the reference distribution's quantiles to define bins — this is the
    standard approach because the bins represent the expected distribution.
    Adding 1e-4 avoids log(0) when a bin has zero counts.
    """
    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]

    if len(reference) < 5 or len(production) < 5:
        return 0.0  # not enough data to compute PSI reliably

    # Define bins from the reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # collapse duplicate edges

    ref_counts  = np.histogram(reference,  bins=breakpoints)[0]
    prod_counts = np.histogram(production, bins=breakpoints)[0]

    ref_pct  = (ref_counts  + 1e-4) / len(reference)
    prod_pct = (prod_counts + 1e-4) / len(production)

    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    return psi


def check_feature_drift(
    city: str = TARGET_CITY,
    recent_days: int = 14,
) -> dict:
    """
    Compare the distribution of each feature over the last `recent_days`
    against the training reference distribution (from ml.features_daily
    covering the training date range).

    We use two methods per feature:
    - PSI: sensitive to magnitude of shift
    - KS test: p-value sensitive to shape differences

    A feature is flagged as drifted if EITHER PSI > threshold OR KS p < threshold.
    """
    engine = get_engine()

    # ── Load training reference (first 70% by date) ───────────────────────
    ref_query = text("""
        SELECT feature_date, {cols}
        FROM ml.features_daily
        WHERE city = :city
        ORDER BY feature_date ASC
    """.format(cols=", ".join(FEATURE_COLS[:15])))  # use numeric feature cols only

    numeric_cols = [c for c in FEATURE_COLS if not c.startswith("is_")]

    ref_query = text("""
        SELECT feature_date, aqi_value, avg_pm25, avg_pm10,
               avg_temperature, avg_humidity, avg_wind_speed, total_precipitation,
               aqi_lag1, aqi_lag7, pm25_lag1, pm25_lag7,
               aqi_roll7_mean, aqi_roll7_std
        FROM ml.features_daily
        WHERE city = :city
        ORDER BY feature_date ASC
    """)

    with engine.connect() as conn:
        all_df = pd.read_sql(ref_query, conn, params={"city": city})

    if len(all_df) < 30:
        logger.warning("[monitor] Fewer than 30 training rows — skipping drift check.")
        return {"drift_details": {}, "n_drifted_features": 0, "drift_triggered": False}

    # Split training reference (first 70% of historical data)
    split_idx = int(len(all_df) * 0.70)
    reference_df = all_df.iloc[:split_idx]

    # Recent production window
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
    recent_df = all_df[pd.to_datetime(all_df["feature_date"]) >= cutoff]

    if recent_df.empty:
        logger.warning("[monitor] No recent data in the last %d days.", recent_days)
        return {"drift_details": {}, "n_drifted_features": 0, "drift_triggered": False}

    check_cols = [
        "aqi_value", "avg_pm25", "avg_pm10",
        "avg_temperature", "avg_humidity", "avg_wind_speed", "total_precipitation",
        "aqi_lag1", "aqi_lag7", "pm25_lag1", "aqi_roll7_mean",
    ]
    check_cols = [c for c in check_cols if c in reference_df.columns and c in recent_df.columns]

    drift_details = {}
    n_drifted = 0

    for col in check_cols:
        ref_vals  = reference_df[col].dropna().values
        prod_vals = recent_df[col].dropna().values

        psi   = compute_psi(ref_vals, prod_vals)
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, prod_vals) if (len(ref_vals) > 5 and len(prod_vals) > 5) else (0, 1)

        is_drifted = (psi > PSI_DRIFT_THRESHOLD) or (ks_pval < KS_PVALUE_THRESHOLD)
        if is_drifted:
            n_drifted += 1
            logger.warning(f"[monitor] DRIFT detected: {col} | PSI={psi:.3f} | KS p={ks_pval:.4f}")

        drift_details[col] = {
            "psi":        round(psi, 4),
            "ks_pvalue":  round(float(ks_pval), 4),
            "drifted":    is_drifted,
        }

    drift_triggered = n_drifted >= DRIFT_FEATURE_COUNT_THRESHOLD
    logger.info(f"[monitor] Drift: {n_drifted} features drifted (threshold={DRIFT_FEATURE_COUNT_THRESHOLD}, triggered={drift_triggered})")

    return {
        "drift_details": drift_details,
        "n_drifted_features": n_drifted,
        "drift_triggered": drift_triggered,
        "recent_days": recent_days,
        "reference_rows": len(reference_df),
        "recent_rows": len(recent_df),
    }


# ── 3. Model Performance ──────────────────────────────────────────────────────

def check_model_performance(
    city: str = TARGET_CITY,
    lookback_days: int = 30,
) -> dict:
    """
    Join ml.prediction_log with actual AQI from analytics.daily_aqi to
    compute rolling RMSE/MAE and classification F1 over the last N days.

    WHY JOIN INSTEAD OF STORING ACTUAL IN prediction_log UPFRONT:
    At prediction time (day T), we don't know the actual AQI for T+1.
    We store the prediction and fill in the actual when it arrives (day T+1+
    pipeline lag). This join does that retroactively.
    """
    engine = get_engine()
    cutoff = date.today() - timedelta(days=lookback_days)

    query = text("""
        SELECT
            pl.pred_id,
            pl.target_date,
            pl.predicted_aqi,
            pl.predicted_category,
            da.aqi_value AS actual_aqi,
            da.aqi_category AS actual_category
        FROM ml.prediction_log pl
        LEFT JOIN analytics.daily_aqi da
            ON da.date = pl.target_date AND da.city = pl.city
        WHERE pl.city = :city
          AND pl.target_date >= :cutoff
          AND pl.status = 'SUCCESS'
        ORDER BY pl.target_date ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"city": city, "cutoff": cutoff})

    # Drop rows where we don't yet have actuals (future dates)
    df = df.dropna(subset=["actual_aqi"])

    if df.empty:
        logger.info("[monitor] No matched prediction+actual rows yet.")
        return {
            "rolling_rmse": None, "rolling_mae": None,
            "rolling_f1": None, "n_predictions_evaluated": 0,
        }

    reg_metrics = evaluate_regression(
        df["actual_aqi"].values, df["predicted_aqi"].values,
        label="rolling-regression"
    )

    clf_metrics = {}
    if df["actual_category"].notna().any():
        clf_metrics = evaluate_classification(
            df["actual_category"].dropna().values,
            df["predicted_category"][df["actual_category"].notna()].values,
            label="rolling-classification",
        )

    # Load baseline RMSE from the model_comparison CSV if available
    baseline_rmse = None
    comp_path = REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        test_rows = comp_df[
            (comp_df["type"] == "regression") &
            (comp_df["split"] == "test") &
            (comp_df["model"].str.contains("Persistence|MA7", case=False, na=False) == False)
        ]
        if not test_rows.empty:
            baseline_rmse = float(test_rows["rmse"].min())

    rmse_pct_change = None
    if baseline_rmse and reg_metrics.get("rmse"):
        rmse_pct_change = (reg_metrics["rmse"] - baseline_rmse) / baseline_rmse

    return {
        "rolling_rmse":           reg_metrics.get("rmse"),
        "rolling_mae":            reg_metrics.get("mae"),
        "rolling_r2":             reg_metrics.get("r2"),
        "rolling_f1":             clf_metrics.get("macro_f1"),
        "baseline_rmse":          baseline_rmse,
        "rmse_pct_change":        rmse_pct_change,
        "n_predictions_evaluated": len(df),
    }


# ── 4. Save to Postgres + report ──────────────────────────────────────────────

def save_monitoring_result(result: dict, city: str = TARGET_CITY):
    """Write one row to ml.monitoring_metrics."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ml.monitoring_metrics (
                checked_at, city,
                missing_rate, stale_data_flag,
                n_drifted_features, drift_triggered, drift_details,
                rolling_rmse, rolling_mae, baseline_rmse, rmse_pct_change,
                rolling_f1,
                retrain_triggered, retrain_reason
            ) VALUES (
                now(), :city,
                :missing_rate, :stale_data_flag,
                :n_drifted_features, :drift_triggered, :drift_details,
                :rolling_rmse, :rolling_mae, :baseline_rmse, :rmse_pct_change,
                :rolling_f1,
                :retrain_triggered, :retrain_reason
            )
        """), {
            "city":               city,
            "missing_rate":       result.get("missing_rate"),
            "stale_data_flag":    result.get("stale_data_flag"),
            "n_drifted_features": result.get("n_drifted_features", 0),
            "drift_triggered":    result.get("drift_triggered", False),
            "drift_details":      json.dumps(result.get("drift_details", {})),
            "rolling_rmse":       result.get("rolling_rmse"),
            "rolling_mae":        result.get("rolling_mae"),
            "baseline_rmse":      result.get("baseline_rmse"),
            "rmse_pct_change":    result.get("rmse_pct_change"),
            "rolling_f1":         result.get("rolling_f1"),
            "retrain_triggered":  result.get("retrain_triggered", False),
            "retrain_reason":     result.get("retrain_reason"),
        })
    logger.info("[monitor] Saved monitoring result to ml.monitoring_metrics")


def generate_monitoring_report(result: dict):
    """Write a human-readable monitoring report to reports/monitoring_report.md."""
    report_path = REPORTS_DIR / "monitoring_report.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# AQI Monitoring Report",
        f"Generated: {now_str}",
        "",
        "## 1. Data Quality",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Missing rate | {result.get('missing_rate', 'N/A'):.1%} |",
        f"| Stale data flag | {result.get('stale_data_flag', 'N/A')} |",
        f"| Hours since latest data | {result.get('hours_since_latest_data', 'N/A')} |",
        f"| Out-of-range rows | {result.get('out_of_range_rows', 'N/A')} |",
        "",
        "## 2. Feature Drift",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Features drifted | {result.get('n_drifted_features', 0)} |",
        f"| Drift triggered | {result.get('drift_triggered', False)} |",
        "",
    ]

    drift_details = result.get("drift_details", {})
    if drift_details:
        lines += [
            "### Per-Feature Drift",
            "| Feature | PSI | KS p-value | Drifted |",
            "|---|---|---|---|",
        ]
        for feat, d in drift_details.items():
            flag = "DRIFT DETECTED" if d.get("drifted") else "STABLE"
            lines.append(f"| {feat} | {d.get('psi', 'N/A')} | {d.get('ks_pvalue', 'N/A')} | {flag} |")
        lines.append("")

    lines += [
        "## 3. Model Performance",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Rolling RMSE | {result.get('rolling_rmse', 'N/A')} |",
        f"| Rolling MAE  | {result.get('rolling_mae',  'N/A')} |",
        f"| Baseline RMSE| {result.get('baseline_rmse', 'N/A')} |",
        f"| RMSE % change| {result.get('rmse_pct_change', 'N/A')} |",
        f"| Rolling F1   | {result.get('rolling_f1',   'N/A')} |",
        f"| N predictions evaluated | {result.get('n_predictions_evaluated', 0)} |",
        "",
        "## 4. Retraining Decision",
        f"| Triggered | {result.get('retrain_triggered', False)} |",
        f"| Reason | {result.get('retrain_reason', 'N/A')} |",
        "",
        "---",
        "_Thresholds: PSI > 0.2, KS p < 0.05, missing rate > 20%, RMSE degradation > 20%, retrain every 30 days._",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[monitor] Report saved → {report_path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_monitoring(city: str = TARGET_CITY) -> dict:
    """
    Run all monitoring checks and return a consolidated result dict.
    Called by: make monitor, the Airflow ML DAG, and src/monitoring/retrain_trigger.py
    """
    print("\n" + "="*60)
    print(f"[monitor] Running monitoring for city={city}")

    dq = check_data_quality(city=city)
    drift = check_feature_drift(city=city)
    perf = check_model_performance(city=city)

    result = {**dq, **drift, **perf}

    # ── Determine if retraining should trigger ────────────────────────────
    from src.monitoring.retrain_trigger import should_retrain
    retrain, reason = should_retrain(result)
    result["retrain_triggered"] = retrain
    result["retrain_reason"]    = reason

    save_monitoring_result(result, city=city)
    generate_monitoring_report(result)

    print(f"\n[monitor] Summary:")
    print(f"  Missing rate:  {result.get('missing_rate', 'N/A'):.1%}")
    print(f"  Stale data:    {result.get('stale_data_flag')}")
    print(f"  Drifted feats: {result.get('n_drifted_features', 0)}")
    print(f"  Rolling RMSE:  {result.get('rolling_rmse')}")
    print(f"  Retrain:       {retrain} — {reason}")

    return result


if __name__ == "__main__":
    run_monitoring()
