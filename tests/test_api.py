"""
Tests for the FastAPI inference service.

Uses httpx TestClient (no live server needed) with a monkey-patched
model so these tests run without MLflow or a database connection.

WHY MOCK THE MODEL AND DB:
Unit tests should test the API logic (routing, validation, response schema)
without being flaky due to external service availability. We patch:
  - _regressor and _classifier globals with simple lambdas
  - get_inference_features to return a synthetic feature row
  - _log_prediction to no-op (no DB writes in tests)
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.features import FEATURE_COLS


# ── Dummy model class ─────────────────────────────────────────────────────────

class DummyRegressor:
    """Fake sklearn pipeline that always predicts AQI=75.0."""
    def predict(self, X):
        return np.array([75.0] * len(X))


class DummyClassifier:
    """Fake sklearn pipeline that always predicts 'Moderate'."""
    def predict(self, X):
        return np.array(["Moderate"] * len(X))


def make_dummy_features():
    """Return a synthetic single-row feature DataFrame."""
    row = {col: 10.0 for col in FEATURE_COLS}
    return pd.DataFrame([row])


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Create a TestClient with models monkey-patched in.
    We patch BEFORE importing the app to ensure the globals are set at load time.
    """
    import src.api.main as api_module

    # Inject fake models
    api_module._regressor   = DummyRegressor()
    api_module._classifier  = DummyClassifier()
    api_module._reg_version = "test-v1"
    api_module._clf_version = "test-v1"

    from httpx import AsyncClient
    from httpx._transports.asgi import ASGITransport
    from fastapi.testclient import TestClient

    return TestClient(api_module.app)


# ── Tests: /health ─────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        with patch("src.api.main.get_engine") as mock_eng:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__  = MagicMock(return_value=False)
            mock_conn.execute   = MagicMock(return_value=None)
            mock_eng.return_value.connect.return_value = mock_conn

            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        with patch("src.api.main.get_engine") as mock_eng:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__  = MagicMock(return_value=False)
            mock_conn.execute   = MagicMock(return_value=None)
            mock_eng.return_value.connect.return_value = mock_conn

            resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "db_reachable" in data
        assert "errors" in data

    def test_models_loaded_is_true(self, client):
        with patch("src.api.main.get_engine") as mock_eng:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__  = MagicMock(return_value=False)
            mock_conn.execute   = MagicMock(return_value=None)
            mock_eng.return_value.connect.return_value = mock_conn

            resp = client.get("/health")
        assert resp.json()["models_loaded"] is True


# ── Tests: /model-info ─────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_model_info_has_feature_cols(self, client):
        resp = client.get("/model-info")
        data = resp.json()
        assert "feature_cols" in data
        assert len(data["feature_cols"]) > 0
        assert "aqi_lag1" in data["feature_cols"]


# ── Tests: POST /predict ───────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_200(self, client):
        """Happy path: valid city + date → 200 with predicted_aqi."""
        with patch("src.api.main.get_inference_features", return_value=make_dummy_features()), \
             patch("src.api.main._log_prediction"):
            resp = client.post("/predict", json={"city": "Bengaluru", "target_date": "2025-06-15"})

        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_aqi" in data
        assert data["predicted_aqi"] == 75.0
        assert data["predicted_category"] == "Moderate"

    def test_predict_invalid_date_format(self, client):
        """Invalid date format must return 422 (Pydantic validation)."""
        resp = client.post("/predict", json={"city": "Bengaluru", "target_date": "15-06-2025"})
        assert resp.status_code == 422

    def test_predict_no_features_returns_404(self, client):
        """If features are unavailable (no DB history), return 404."""
        with patch("src.api.main.get_inference_features", return_value=None):
            resp = client.post("/predict", json={"city": "Bengaluru", "target_date": "2025-06-15"})
        assert resp.status_code == 404

    def test_predict_response_schema(self, client):
        """Response must have all required fields."""
        with patch("src.api.main.get_inference_features", return_value=make_dummy_features()), \
             patch("src.api.main._log_prediction"):
            resp = client.post("/predict", json={"city": "Bengaluru"})

        assert resp.status_code == 200
        data = resp.json()
        required_fields = ["city", "target_date", "predicted_aqi",
                           "predicted_category", "aqi_confidence",
                           "regressor_version", "classifier_version", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_models_not_loaded(self):
        """If models are not loaded, /predict must return 503."""
        import src.api.main as api_module
        from fastapi.testclient import TestClient

        # Temporarily unload models
        orig_reg, orig_clf = api_module._regressor, api_module._classifier
        api_module._regressor  = None
        api_module._classifier = None

        try:
            test_client = TestClient(api_module.app)
            resp = test_client.post("/predict", json={"city": "Bengaluru"})
            assert resp.status_code == 503
        finally:
            api_module._regressor  = orig_reg
            api_module._classifier = orig_clf


# ── Tests: /metrics ────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_has_required_keys(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        assert "request_count" in data
        assert "error_count" in data
        assert "latency_p50_ms" in data
        assert "latency_p95_ms" in data

    def test_request_count_increments(self, client):
        """Each call to /predict should increment request_count."""
        import src.api.main as api_module
        initial = api_module._metrics_store["request_count"]

        with patch("src.api.main.get_inference_features", return_value=make_dummy_features()), \
             patch("src.api.main._log_prediction"):
            client.post("/predict", json={"city": "Bengaluru"})

        resp = client.get("/metrics")
        assert resp.json()["request_count"] >= initial + 1
