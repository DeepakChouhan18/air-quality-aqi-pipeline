"""
FastAPI inference service for next-day AQI prediction.

WHY FASTAPI (not Flask):
- Built-in Pydantic validation → automatic 422 errors with clear messages
- Async-native → handles multiple concurrent requests without blocking
- Auto-generated OpenAPI docs at /docs
- Type-safe → fewer runtime surprises in production

DESIGN DECISIONS:
1. Models are loaded ONCE at startup (not per-request) using MLflow
   registry by alias — changing the 'production' model in the registry
   takes effect on the next server restart without touching this file.
2. The API builds features itself from the DB using src/ml/features.py —
   callers don't need to compute features client-side. This is the
   "feature serving" pattern in MLOps.
3. Every prediction is logged to ml.prediction_log in Postgres for
   monitoring and audit. This is fire-and-forget (doesn't slow response).
4. No hardcoded file paths — all config comes from environment variables.
"""

import time
import json
import logging
from datetime import date, datetime, timezone
from typing import Optional

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from src.database.connection import get_engine
from src.ml.features import get_inference_features, FEATURE_COLS, FEATURE_VERSION

# ── Application setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aqi-api")

app = FastAPI(
    title="AQI Prediction API",
    description=(
        "Next-day AQI regression and category classification for Part 2 MLOps. "
        "Loads models from MLflow registry by alias 'production'."
    ),
    version="2.0.0",
)

# ── Global state (loaded once at startup) ─────────────────────────────────────
_regressor   = None   # sklearn pipeline for AQI regression
_classifier  = None   # sklearn pipeline for AQI category classification
_reg_version = None   # version string from registry
_clf_version = None   # version string from registry
_load_errors: list[str] = []

# ── Operational counters ──────────────────────────────────────────────────────
_metrics_store = {
    "request_count": 0,
    "error_count": 0,
    "latencies_ms": [],      # sliding window — keep last 1000
    "last_prediction_time": None,
}


# ── Startup: load models from MLflow registry ─────────────────────────────────

@app.on_event("startup")
async def load_models():
    """
    Load production models from MLflow registry at startup.
    Failures are logged but don't crash the server so /health can still
    report the error clearly.
    """
    global _regressor, _classifier, _reg_version, _clf_version

    from src.config import MLFLOW_TRACKING_URI, MLFLOW_REGISTERED_REGRESSOR, \
        MLFLOW_REGISTERED_CLASSIFIER, MLFLOW_PRODUCTION_ALIAS

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    for reg_name, is_reg in [
        (MLFLOW_REGISTERED_REGRESSOR, True),
        (MLFLOW_REGISTERED_CLASSIFIER, False),
    ]:
        try:
            mv = client.get_model_version_by_alias(reg_name, MLFLOW_PRODUCTION_ALIAS)
            model_uri = f"models:/{reg_name}@{MLFLOW_PRODUCTION_ALIAS}"
            model = mlflow.sklearn.load_model(model_uri)
            if is_reg:
                _regressor   = model
                _reg_version = mv.version
            else:
                _classifier  = model
                _clf_version = mv.version
            logger.info(f"Loaded '{reg_name}' v{mv.version} from registry.")
        except Exception as e:
            msg = f"Failed to load '{reg_name}': {e}"
            logger.warning(msg)
            _load_errors.append(msg)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input: either city + date (API fetches features from DB)
           OR city + target_date + raw feature values (bypass DB lookup).
    """
    city: str = Field(
        default="Bengaluru",
        description="Target city (must match analytics.daily_aqi city column)",
        examples=["Bengaluru"],
    )
    target_date: Optional[str] = Field(
        default=None,
        description=(
            "Date to PREDICT for in YYYY-MM-DD format. "
            "If None, predicts for tomorrow using today's features."
        ),
        examples=["2025-06-15"],
    )

    @field_validator("target_date")
    @classmethod
    def validate_date(cls, v):
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                raise ValueError("target_date must be in YYYY-MM-DD format")
        return v


class PredictResponse(BaseModel):
    city: str
    target_date: str
    predicted_aqi: Optional[float]
    predicted_category: Optional[str]
    aqi_confidence: str              # "high" / "low" based on feature completeness
    regressor_version: Optional[str]
    classifier_version: Optional[str]
    timestamp: str
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                      # "ok" / "degraded" / "error"
    models_loaded: bool
    regressor_version: Optional[str]
    classifier_version: Optional[str]
    db_reachable: bool
    errors: list[str]


class ModelInfoResponse(BaseModel):
    regressor_name: str
    regressor_version: Optional[str]
    classifier_name: str
    classifier_version: Optional[str]
    feature_version: str
    feature_cols: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    """
    Check model load status and database connectivity.
    Returns 200 with status='degraded' if models aren't loaded (rather than 503)
    so monitoring tools can distinguish between "server down" and "model not loaded".
    """
    from src.config import MLFLOW_REGISTERED_REGRESSOR, MLFLOW_REGISTERED_CLASSIFIER

    db_ok = False
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        _load_errors.append(f"DB connectivity check failed: {e}")

    models_loaded = _regressor is not None and _classifier is not None
    status = "ok" if (models_loaded and db_ok) else "degraded"

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        regressor_version=_reg_version,
        classifier_version=_clf_version,
        db_reachable=db_ok,
        errors=_load_errors[-5:],   # last 5 errors
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Operations"])
def model_info():
    """Return model names, versions, and the feature schema."""
    from src.config import MLFLOW_REGISTERED_REGRESSOR, MLFLOW_REGISTERED_CLASSIFIER

    return ModelInfoResponse(
        regressor_name=MLFLOW_REGISTERED_REGRESSOR,
        regressor_version=_reg_version,
        classifier_name=MLFLOW_REGISTERED_CLASSIFIER,
        classifier_version=_clf_version,
        feature_version=FEATURE_VERSION,
        feature_cols=FEATURE_COLS,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict next-day AQI and category for a city/date.

    Flow:
    1. Parse target_date (default: tomorrow)
    2. Build features for (target_date - 1 day) from the DB via features.py
    3. Run regression model → predicted AQI
    4. Run classification model → predicted category
    5. Log prediction to ml.prediction_log
    6. Return structured response

    Returns 503 if models aren't loaded, 422 if input is invalid,
    200 with warning if features are incomplete.
    """
    from src.config import MLFLOW_REGISTERED_REGRESSOR, MLFLOW_REGISTERED_CLASSIFIER

    t_start = time.time()
    _metrics_store["request_count"] += 1

    # ── Validate models loaded ────────────────────────────────────────────
    if _regressor is None or _classifier is None:
        _metrics_store["error_count"] += 1
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check /health for details.",
        )

    # ── Parse target date ─────────────────────────────────────────────────
    if req.target_date:
        target_dt = date.fromisoformat(req.target_date)
    else:
        from datetime import timedelta
        target_dt = date.today() + timedelta(days=1)

    target_date_str = target_dt.isoformat()

    # ── Build features from DB ────────────────────────────────────────────
    warning = None
    try:
        feature_row = get_inference_features(
            city=req.city, target_date=target_dt
        )
    except Exception as e:
        logger.error(f"Feature build failed for {req.city} / {target_dt}: {e}")
        _metrics_store["error_count"] += 1
        raise HTTPException(
            status_code=500,
            detail=f"Feature engineering failed: {e}",
        )

    if feature_row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical data available for city='{req.city}' "
                f"on {target_dt - __import__('datetime').timedelta(days=1)}. "
                "Run 'make backfill' to load history."
            ),
        )

    # Check for NaN features — predict anyway but flag confidence
    import numpy as np
    X = feature_row[FEATURE_COLS].values
    n_nan = int(np.isnan(X).sum())
    confidence = "low" if n_nan > 0 else "high"
    if n_nan > 0:
        warning = f"{n_nan} feature(s) are NaN — prediction may be less accurate."
        logger.warning(warning)

    # ── Run models ────────────────────────────────────────────────────────
    try:
        pred_aqi = float(_regressor.predict(X)[0])
        pred_cat = str(_classifier.predict(X)[0])
    except Exception as e:
        _metrics_store["error_count"] += 1
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    latency_ms = (time.time() - t_start) * 1000
    _metrics_store["latencies_ms"].append(latency_ms)
    if len(_metrics_store["latencies_ms"]) > 1000:
        _metrics_store["latencies_ms"] = _metrics_store["latencies_ms"][-1000:]
    _metrics_store["last_prediction_time"] = datetime.now(timezone.utc).isoformat()

    # ── Log to Postgres (fire-and-forget) ─────────────────────────────────
    try:
        _log_prediction(
            city=req.city,
            target_date=target_dt,
            predicted_aqi=pred_aqi,
            predicted_category=pred_cat,
            latency_ms=latency_ms,
            input_features=feature_row[FEATURE_COLS].to_dict(orient="records")[0],
            reg_version=_reg_version,
            clf_version=_clf_version,
        )
    except Exception as e:
        logger.warning(f"Prediction logging failed (non-fatal): {e}")

    logger.info(
        f"PREDICT city={req.city} date={target_date_str} "
        f"aqi={pred_aqi:.1f} cat={pred_cat} latency={latency_ms:.1f}ms"
    )

    return PredictResponse(
        city=req.city,
        target_date=target_date_str,
        predicted_aqi=round(pred_aqi, 1),
        predicted_category=pred_cat,
        aqi_confidence=confidence,
        regressor_version=_reg_version,
        classifier_version=_clf_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        warning=warning,
    )


@app.get("/metrics", tags=["Operations"])
def operational_metrics():
    """
    Prometheus-style operational metrics in JSON.
    Real Prometheus scraping would need the prometheus_client library;
    this JSON endpoint is sufficient for the Streamlit dashboard.
    """
    lats = _metrics_store["latencies_ms"]
    if lats:
        import numpy as np
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
    else:
        p50 = p95 = 0.0

    error_rate = (
        _metrics_store["error_count"] / max(_metrics_store["request_count"], 1)
    )

    return {
        "request_count":       _metrics_store["request_count"],
        "error_count":         _metrics_store["error_count"],
        "error_rate":          round(error_rate, 4),
        "latency_p50_ms":      round(p50, 2),
        "latency_p95_ms":      round(p95, 2),
        "last_prediction_time": _metrics_store["last_prediction_time"],
    }


# ── Prediction logging ────────────────────────────────────────────────────────

def _log_prediction(
    city, target_date, predicted_aqi, predicted_category,
    latency_ms, input_features, reg_version, clf_version,
):
    """Insert a row into ml.prediction_log (async-safe fire-and-forget)."""
    from src.config import MLFLOW_REGISTERED_REGRESSOR, MLFLOW_REGISTERED_CLASSIFIER

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ml.prediction_log (
                requested_at, city, target_date, predicted_aqi, predicted_category,
                regressor_name, regressor_version, classifier_name, classifier_version,
                latency_ms, status, input_features
            ) VALUES (
                now(), :city, :target_date, :predicted_aqi, :predicted_category,
                :reg_name, :reg_ver, :clf_name, :clf_ver,
                :latency_ms, 'SUCCESS', :features
            )
        """), {
            "city":              city,
            "target_date":       target_date,
            "predicted_aqi":     predicted_aqi,
            "predicted_category": predicted_category,
            "reg_name":          MLFLOW_REGISTERED_REGRESSOR,
            "reg_ver":           str(reg_version),
            "clf_name":          MLFLOW_REGISTERED_CLASSIFIER,
            "clf_ver":           str(clf_version),
            "latency_ms":        latency_ms,
            "features":          json.dumps(
                {k: (None if v != v else v) for k, v in input_features.items()}
            ),
        })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=False)
