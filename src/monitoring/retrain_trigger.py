"""
Retraining trigger policy: decides whether to retrain the model based on
monitoring signals.

WHY DOCUMENT THE POLICY IN CODE (not just docs):
A policy that only lives in documentation is ignored. Writing it as code
forces it to be precise (exact thresholds, exact logic) and testable
(see tests/test_retrain_trigger.py). The docs reference this file.

RETRAINING TRIGGERS (all configurable in config.py / .env):
1. Feature drift: PSI > PSI_DRIFT_THRESHOLD on ≥ DRIFT_FEATURE_COUNT_THRESHOLD features
2. Performance degradation: rolling RMSE > baseline_rmse * (1 + RMSE_DEGRADATION_THRESHOLD)
3. Data quality: missing_rate > MISSING_RATE_THRESHOLD
4. Scheduled: last retrain was more than RETRAIN_EVERY_DAYS days ago

PROMOTION GATE (in register_best.py):
Even after retraining, the new model is only promoted to production if it
beats the current production model on the validation metric. This prevents
accidental downgrades from a bad training run.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.config import (
    PSI_DRIFT_THRESHOLD,
    DRIFT_FEATURE_COUNT_THRESHOLD,
    MISSING_RATE_THRESHOLD,
    RMSE_DEGRADATION_THRESHOLD,
    RETRAIN_EVERY_DAYS,
)

logger = logging.getLogger(__name__)


def should_retrain(monitoring_result: dict) -> tuple[bool, str]:
    """
    Evaluate monitoring signals and return (should_retrain: bool, reason: str).

    Args:
        monitoring_result: dict returned by src/monitoring/monitor.py::run_monitoring()

    Returns:
        (True, "reason string") if any trigger is met
        (False, "no trigger") if all checks pass
    """
    # ── Trigger 1: Feature drift ───────────────────────────────────────────
    n_drifted = monitoring_result.get("n_drifted_features", 0)
    if n_drifted >= DRIFT_FEATURE_COUNT_THRESHOLD:
        reason = (
            f"Feature drift: {n_drifted} features exceeded drift threshold "
            f"(PSI > {PSI_DRIFT_THRESHOLD} or KS p < 0.05). "
            f"Threshold is {DRIFT_FEATURE_COUNT_THRESHOLD} features."
        )
        logger.warning(f"[retrain] TRIGGER: {reason}")
        return True, reason

    # ── Trigger 2: Performance degradation ────────────────────────────────
    rolling_rmse  = monitoring_result.get("rolling_rmse")
    baseline_rmse = monitoring_result.get("baseline_rmse")
    if rolling_rmse is not None and baseline_rmse is not None and baseline_rmse > 0:
        degradation = (rolling_rmse - baseline_rmse) / baseline_rmse
        if degradation > RMSE_DEGRADATION_THRESHOLD:
            reason = (
                f"Performance degradation: rolling RMSE={rolling_rmse:.2f} is "
                f"{degradation:.1%} above baseline RMSE={baseline_rmse:.2f}. "
                f"Threshold is {RMSE_DEGRADATION_THRESHOLD:.0%}."
            )
            logger.warning(f"[retrain] TRIGGER: {reason}")
            return True, reason

    # ── Trigger 3: Data quality ────────────────────────────────────────────
    missing_rate = monitoring_result.get("missing_rate", 0.0)
    if missing_rate is not None and missing_rate > MISSING_RATE_THRESHOLD:
        reason = (
            f"Data quality: missing rate={missing_rate:.1%} exceeds "
            f"threshold={MISSING_RATE_THRESHOLD:.0%}."
        )
        logger.warning(f"[retrain] TRIGGER: {reason}")
        return True, reason

    # ── Trigger 4: Scheduled retrain ──────────────────────────────────────
    if _days_since_last_retrain() >= RETRAIN_EVERY_DAYS:
        reason = (
            f"Scheduled retrain: last retrain was ≥ {RETRAIN_EVERY_DAYS} days ago."
        )
        logger.info(f"[retrain] TRIGGER: {reason}")
        return True, reason

    # ── No triggers ────────────────────────────────────────────────────────
    reason = (
        f"No triggers: drift={n_drifted}/{DRIFT_FEATURE_COUNT_THRESHOLD}, "
        f"missing_rate={missing_rate:.1%}/{MISSING_RATE_THRESHOLD:.0%}, "
        f"RMSE {'N/A' if rolling_rmse is None else f'{rolling_rmse:.2f}'}."
    )
    logger.info(f"[retrain] No trigger. {reason}")
    return False, reason


def _days_since_last_retrain() -> int:
    """
    Read the last successful retrain timestamp from ml.monitoring_metrics.
    Returns a large number (999) if no retrain record exists so the first
    run always triggers a retrain check.
    """
    try:
        from src.database.connection import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MAX(checked_at)
                FROM ml.monitoring_metrics
                WHERE retrain_triggered = true
            """)).scalar()

        if result is None:
            return 999  # never retrained → always trigger scheduled check

        days = (datetime.now(timezone.utc) - result.astimezone(timezone.utc)).days
        return days

    except Exception as e:
        logger.warning(f"[retrain] Could not query last retrain time: {e}")
        return 999  # fail-safe: assume retrain needed


if __name__ == "__main__":
    # Quick test with dummy monitoring result
    dummy = {
        "n_drifted_features": 0,
        "rolling_rmse": 12.5,
        "baseline_rmse": 10.0,
        "missing_rate": 0.05,
    }
    triggered, reason = should_retrain(dummy)
    print(f"Retrain: {triggered}\nReason: {reason}")
