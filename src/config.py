"""
Central configuration for the whole pipeline.

WHY THIS FILE EXISTS:
Instead of scattering `os.environ["X"]` calls and magic numbers across
every script, every setting lives here. If a URL, table name, or
threshold ever changes, you edit ONE file instead of hunting through
ten scripts. This is standard practice in real codebases.

Part 2 adds: backfill settings, ML paths, MLflow config, monitoring
thresholds, and API config — all controlled by environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a local .env file (never committed to git)
load_dotenv()

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
REJECTED_DIR = DATA_DIR / "rejected"
LOGS_DIR = DATA_DIR / "logs"

# Part 2: ML-specific output directories
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ML_DATA_DIR = DATA_DIR / "ml"

for d in [RAW_DIR, STAGING_DIR, REJECTED_DIR, LOGS_DIR,
          REPORTS_DIR, FIGURES_DIR, ML_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
# Part 2: historical archive endpoints (no API key required)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Study area: change this to whichever city/coordinates you're targeting
TARGET_CITY = os.environ.get("TARGET_CITY", "Bengaluru")
TARGET_LAT = float(os.environ.get("TARGET_LAT", "12.9716"))
TARGET_LON = float(os.environ.get("TARGET_LON", "77.5946"))

# Part 2: backfill configuration — how many months of history to pull
BACKFILL_MONTHS = int(os.environ.get("BACKFILL_MONTHS", "14"))  # ≥12 for enough ML data

# ---------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "aqi_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---------------------------------------------------------------------
# Data quality thresholds
# (documented in docs/validation_rules.md — do not invent values silently)
# ---------------------------------------------------------------------
POLLUTANT_MAX_VALID = {
    "pm25": 1000.0,   # ug/m3 — WHO/EPA scale caps around here for ambient air
    "pm10": 1000.0,
    "no2": 1000.0,    # ppb
    "so2": 1000.0,
    "co": 50000.0,    # ppb (CO has a much wider valid range than PM)
}

# ---------------------------------------------------------------------
# Part 2: MLflow configuration
# ---------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://localhost:5001"
)
MLFLOW_EXPERIMENT_REGRESSION = "aqi-regression"
MLFLOW_EXPERIMENT_CLASSIFICATION = "aqi-classification"
MLFLOW_REGISTERED_REGRESSOR = "aqi_regressor"
MLFLOW_REGISTERED_CLASSIFIER = "aqi_classifier"
MLFLOW_PRODUCTION_ALIAS = "production"   # alias tag used for the "live" model

# Fixed random seed — same seed everywhere ensures reproducibility
RANDOM_SEED = 42

# Train/val/test chronological split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------
# Part 2: Monitoring / retraining thresholds
# These live in config (not hardcoded in monitoring scripts) so you can
# tune them without touching logic code.
# ---------------------------------------------------------------------
# Data quality
STALE_DATA_HOURS = int(os.environ.get("STALE_DATA_HOURS", "48"))
MISSING_RATE_THRESHOLD = float(os.environ.get("MISSING_RATE_THRESHOLD", "0.20"))

# Feature drift (Population Stability Index and Kolmogorov-Smirnov)
PSI_DRIFT_THRESHOLD = float(os.environ.get("PSI_DRIFT_THRESHOLD", "0.2"))
KS_PVALUE_THRESHOLD = float(os.environ.get("KS_PVALUE_THRESHOLD", "0.05"))
DRIFT_FEATURE_COUNT_THRESHOLD = int(os.environ.get("DRIFT_FEATURE_COUNT_THRESHOLD", "3"))

# Model performance degradation: retrain if rolling RMSE > baseline * (1 + factor)
RMSE_DEGRADATION_THRESHOLD = float(os.environ.get("RMSE_DEGRADATION_THRESHOLD", "0.20"))

# Retrain every N days regardless of other signals
RETRAIN_EVERY_DAYS = int(os.environ.get("RETRAIN_EVERY_DAYS", "30"))

# ---------------------------------------------------------------------
# Part 2: FastAPI service configuration
# ---------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

