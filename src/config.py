"""
Central configuration for the whole pipeline.

WHY THIS FILE EXISTS:
Instead of scattering `os.environ["X"]` calls and magic numbers across
every script, every setting lives here. If a URL, table name, or
threshold ever changes, you edit ONE file instead of hunting through
ten scripts. This is standard practice in real codebases.
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

for d in [RAW_DIR, STAGING_DIR, REJECTED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Study area: change this to whichever city/coordinates you're targeting
TARGET_CITY = os.environ.get("TARGET_CITY", "Bengaluru")
TARGET_LAT = float(os.environ.get("TARGET_LAT", "12.9716"))
TARGET_LON = float(os.environ.get("TARGET_LON", "77.5946"))

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
