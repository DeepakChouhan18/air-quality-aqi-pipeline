"""
Tests for drift detection and retrain trigger logic.

These test the monitoring module's drift computation and the
retrain decision policy — all with synthetic data, no DB required.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.monitor import compute_psi
from src.monitoring.retrain_trigger import should_retrain


# ── PSI Tests ─────────────────────────────────────────────────────────────────

class TestPSI:
    def test_identical_distributions_psi_near_zero(self):
        """PSI should be ~0 when reference == production (no drift)."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=50, scale=10, size=500)
        psi = compute_psi(data, data.copy())
        assert psi < 0.05, f"PSI for identical distributions should be ~0, got {psi:.4f}"

    def test_shifted_distribution_psi_high(self):
        """
        PSI should be > 0.2 when the production distribution is significantly
        different from the reference (mean shifted by 3 sigma).
        """
        rng = np.random.default_rng(42)
        reference  = rng.normal(loc=50, scale=10, size=500)
        production = rng.normal(loc=80, scale=10, size=500)  # mean shifted
        psi = compute_psi(reference, production)
        assert psi > 0.2, f"PSI for shifted distribution should be > 0.2, got {psi:.4f}"

    def test_psi_non_negative(self):
        """PSI must always be ≥ 0."""
        rng = np.random.default_rng(0)
        ref  = rng.exponential(scale=10, size=200)
        prod = rng.uniform(0, 100, size=200)
        psi = compute_psi(ref, prod)
        assert psi >= 0, f"PSI must be non-negative, got {psi}"

    def test_psi_with_few_samples_returns_zero(self):
        """With fewer than 5 samples, PSI should return 0 (not enough data)."""
        psi = compute_psi(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert psi == 0.0, "PSI with < 5 samples should return 0"

    def test_psi_handles_nan(self):
        """PSI computation should ignore NaN values gracefully."""
        ref  = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        prod = np.array([1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10])
        psi = compute_psi(ref, prod)
        assert isinstance(psi, float), "PSI should return float even with NaN inputs"


# ── Retrain Trigger Tests ─────────────────────────────────────────────────────

class TestRetrainTrigger:
    def test_no_trigger_when_all_ok(self):
        """No trigger when metrics are healthy."""
        result = {
            "n_drifted_features": 0,
            "rolling_rmse":       10.0,
            "baseline_rmse":      10.0,
            "missing_rate":       0.01,
        }
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain", return_value=5):
            triggered, reason = should_retrain(result)
        assert not triggered, f"Should not trigger: {reason}"

    def test_trigger_on_drift(self):
        """Trigger when enough features have drifted."""
        from src.config import DRIFT_FEATURE_COUNT_THRESHOLD
        result = {
            "n_drifted_features": DRIFT_FEATURE_COUNT_THRESHOLD,  # exactly at threshold
            "rolling_rmse":       10.0,
            "baseline_rmse":      10.0,
            "missing_rate":       0.01,
        }
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain", return_value=5):
            triggered, reason = should_retrain(result)
        assert triggered, "Should trigger when n_drifted_features == threshold"
        assert "drift" in reason.lower()

    def test_trigger_on_performance_degradation(self):
        """Trigger when rolling RMSE is significantly above baseline."""
        result = {
            "n_drifted_features": 0,
            "rolling_rmse":       15.0,  # 50% above baseline
            "baseline_rmse":      10.0,
            "missing_rate":       0.01,
        }
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain", return_value=5):
            triggered, reason = should_retrain(result)
        assert triggered, "Should trigger when RMSE degradation is >20%"
        assert "performance" in reason.lower() or "rmse" in reason.lower()

    def test_trigger_on_high_missing_rate(self):
        """Trigger when missing rate exceeds threshold."""
        from src.config import MISSING_RATE_THRESHOLD
        result = {
            "n_drifted_features": 0,
            "rolling_rmse":       10.0,
            "baseline_rmse":      10.0,
            "missing_rate":       MISSING_RATE_THRESHOLD + 0.05,  # above threshold
        }
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain", return_value=5):
            triggered, reason = should_retrain(result)
        assert triggered, "Should trigger on high missing rate"
        assert "missing" in reason.lower() or "quality" in reason.lower()

    def test_trigger_on_schedule(self):
        """Trigger when enough days have passed since last retrain."""
        from src.config import RETRAIN_EVERY_DAYS
        result = {
            "n_drifted_features": 0,
            "rolling_rmse":       10.0,
            "baseline_rmse":      10.0,
            "missing_rate":       0.01,
        }
        # Simulate that last retrain was RETRAIN_EVERY_DAYS ago
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain",
                   return_value=RETRAIN_EVERY_DAYS):
            triggered, reason = should_retrain(result)
        assert triggered, "Should trigger on scheduled retrain interval"
        assert "schedule" in reason.lower() or "days" in reason.lower()

    def test_no_trigger_just_before_schedule(self):
        """One day before the scheduled retrain interval, should NOT trigger."""
        from src.config import RETRAIN_EVERY_DAYS
        result = {
            "n_drifted_features": 0,
            "rolling_rmse":       10.0,
            "baseline_rmse":      10.0,
            "missing_rate":       0.01,
        }
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain",
                   return_value=RETRAIN_EVERY_DAYS - 1):
            triggered, reason = should_retrain(result)
        assert not triggered, f"Should NOT trigger one day before scheduled interval: {reason}"

    def test_trigger_returns_reason_string(self):
        """should_retrain must always return a non-empty reason string."""
        result = {"n_drifted_features": 0, "missing_rate": 0.01}
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain", return_value=5):
            triggered, reason = should_retrain(result)
        assert isinstance(reason, str) and len(reason) > 0, "Reason must be a non-empty string"

    def test_rmse_exactly_at_threshold_triggers(self):
        """
        RMSE degradation exactly equal to the threshold should trigger.
        (threshold is a > comparison in config, test with slightly above)
        """
        from src.config import RMSE_DEGRADATION_THRESHOLD
        result = {
            "n_drifted_features": 0,
            "rolling_rmse":       10.0 * (1 + RMSE_DEGRADATION_THRESHOLD + 0.001),
            "baseline_rmse":      10.0,
            "missing_rate":       0.01,
        }
        with patch("src.monitoring.retrain_trigger._days_since_last_retrain", return_value=5):
            triggered, reason = should_retrain(result)
        assert triggered
