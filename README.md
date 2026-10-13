# Air Quality Data Engineering Pipeline

An end-to-end data engineering pipeline that ingests air-pollution and
weather data, validates and transforms it through a layered warehouse
(raw → staging → core → analytics), and powers a Tableau Public
dashboard. Built as Part 1 of a Data Engineering + MLOps course project.

## 1. Objective
Track and analyze air quality (AQI) trends for a target city by
combining pollutant measurements with weather context, producing a
reliable, auditable data pipeline suitable for downstream ML forecasting.

## 2. Problem Statement
Air quality data from public APIs is fragmented, unvalidated, and not
analysis-ready out of the box. This project builds the missing layer:
automated ingestion, quality-checked transformation, and a queryable
analytical warehouse.

## 3. Architecture
See `docs/pipeline_flow.md` for the full diagram. In short:

```
OpenAQ + Open-Meteo APIs -> raw landing zone -> Airflow ETL
  -> validated staging -> PostgreSQL (core) -> analytics (gold)
  -> Tableau Public dashboard
```

## 4. Data Sources
| Source | Data | Access |
|---|---|---|
| [OpenAQ v3](https://docs.openaq.org) | Pollutant concentrations (PM2.5, PM10, NO2, SO2, CO), station metadata | Free API key required — register at explore.openaq.org |
| [Open-Meteo](https://open-meteo.com) | Temperature, humidity, precipitation, wind speed | No key required (non-commercial use) |

## 5. Technology Stack
Python, Requests, Pandas, Apache Airflow, PostgreSQL, Tableau Public, Docker, pytest, GitHub Actions.

## 6. Folder Structure
See the project tree — key folders: `src/ingestion`, `src/transformation`, `src/database`, `dags/`, `sql/`, `tests/`, `docs/`.

## 7. Setup Instructions
```bash
git clone <your-repo-url>
cd air-quality-data-engineering
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill in your OpenAQ API key
```

## 8. Environment Variables
See `.env.example`. Required: `OPENAQ_API_KEY`. Optional: target city/coordinates, DB credentials.

## 9. Database Setup
```bash
docker compose up -d           # starts Postgres, auto-runs sql/*.sql on first boot
```
Or run the SQL files manually against your own Postgres instance in order:
`sql/schemas.sql` → `sql/tables.sql` → `sql/analytics.sql`.

## 10. Airflow Setup
Use the [official Airflow Docker Compose quick-start](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html),
mounting this repo's `dags/` and `src/` folders into the Airflow containers.
Then enable the `air_quality_pipeline` DAG from the Airflow UI.

## 11. How to Run the Pipeline
**Manually (without Airflow), for local testing:**
```bash
python -m src.ingestion.openaq_api
python -m src.ingestion.weather_api
python -m src.transformation.clean_air_quality data/raw/openaq/<date>/measurements.json
python -m src.transformation.clean_weather data/raw/weather/<date>/weather.json
```
**Via Airflow:** trigger the `air_quality_pipeline` DAG from the UI or CLI (`airflow dags trigger air_quality_pipeline`).

## 12. Data Quality Checks
See `docs/validation_rules.md`. Covers missing fields, duplicates, value-range checks (US EPA-informed thresholds), and timestamp validation. Rejected records are never deleted — see `data/rejected/` and `staging.rejected_records`.

## 13. Analytics
`analytics.daily_aqi`, `analytics.station_summary`, `analytics.pollutant_summary` — see `docs/data_dictionary.md` for full column definitions. AQI is calculated using the **US EPA PM2.5 breakpoint methodology** (documented in `src/transformation/create_analytics.py`).

## 14. Tableau Dashboard
Connect Tableau Public to the `analytics` schema (or export CSVs from those tables) and build the 6 views: AQI KPI, AQI Trend, Station Comparison, Pollutant Analysis, Hourly/Monthly Pattern, Weather Impact.

## 15. Results
Pipeline run on **2026-09-06** for **Bengaluru** (12.97°N, 77.59°E):

| Metric | Value |
|---|---|
| Monitoring stations discovered | 26 |
| Raw measurements extracted | 266 |
| Measurements after validation | 266 (0 rejected) |
| Weather records (hourly, 14 days) | 336 |
| Pollutants tracked | CO, NO, NO₂, NOₓ, O₃, PM1, PM2.5, PM10, SO₂ |
| Days with AQI data | 7 |
| Average PM2.5 AQI | ~68 — **Moderate** |
| Worst observed day | 2026-07-25, AQI 101.9 — **Unhealthy for Sensitive Groups** |
| Avg temperature | 25.0 °C |
| Avg humidity | 67.7 % |
| Total precipitation | 20.5 mm |

All pipeline stages (extract → validate → transform → analytics) completed successfully with full lineage recorded in `data/logs/pipeline_runs.csv`.

## 16. Limitations
- Single-city scope for this phase.
- AQI calculated from PM2.5 only (not the full multi-pollutant EPA composite index).
- No backfill/estimation for missing time periods — gaps are reported, not guessed.

## 17. Future MLOps Extension (Part 2)
This warehouse's `core` and `analytics` tables are the feature source for Part 2: lag/rolling features → scikit-learn model → MLflow tracking → FastAPI prediction service → Docker → monitoring for drift and forecast error.

## Pipeline Lineage / Audit Trail
Every ingestion and transform run writes a record to `staging.pipeline_runs` (and `data/logs/pipeline_runs.csv` locally) with: run ID, source, stage, status, rows extracted/valid/rejected, and any error message — so any run's outcome can be traced without re-executing it.

## Testing
```bash
pytest tests/ -v
```
CI runs these automatically on every push via `.github/workflows/ci.yml`.

---

## Part 2 — MLOps Extension

### Overview

Part 2 adds a full MLOps layer on top of the Part 1 data engineering pipeline:
historical data backfill → feature pipeline → model training (Ridge/RF/XGBoost) →
MLflow experiment tracking + model registry → FastAPI inference service → Docker →
monitoring (drift + performance) → automated Airflow-based retraining.

### New Data Sources

| Source | Data | Auth |
|---|---|---|
| [Open-Meteo Air Quality API](https://air-quality-api.open-meteo.com) | PM2.5, PM10, NO2, SO2, O3, CO, US AQI (historical hourly) | No key required |
| [Open-Meteo Archive API](https://archive-api.open-meteo.com) | Temperature, humidity, wind, precipitation (historical hourly) | No key required |

### Part 2 Setup (from a fresh clone)

```bash
# 1. Clone and install Part 1 dependencies
git clone <your-repo-url>
cd air-quality-data-engineering
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Install Part 2 ML/API dependencies
pip install -r requirements-ml.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set OPENAQ_API_KEY and DB credentials

# 4. Start Postgres (and MLflow + API)
make docker-up
# OR for just Postgres:
docker compose up -d postgres

# 5. Run the SQL to create the ml schema
psql -h localhost -U postgres -d aqi_db -f sql/ml_schema.sql

# 6. Backfill ≥12 months of historical data
make backfill

# 7. Build feature table
make features

# 8. Train models (logs to MLflow at http://localhost:5001)
make train

# 9. Register best models
make register

# 10. Start inference API
make serve
# Test: curl http://localhost:8000/health

# 11. Run monitoring
make monitor
```

### Verify End-to-End

```bash
# Health check
curl http://localhost:8000/health

# Predict next-day AQI for Bengaluru
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Bengaluru", "target_date": "2025-06-15"}'

# Run monitoring report
make monitor
cat reports/monitoring_report.md

# All tests
make test
```

### Docker (all services)

```bash
make docker-up
# Services:
#   Postgres:   localhost:5432
#   MLflow UI:  http://localhost:5001
#   API docs:   http://localhost:8000/docs
#   Streamlit:  http://localhost:8501
```

### Key Docs (Part 2)

| Document | Purpose |
|---|---|
| `docs/part2_architecture.md` | Mermaid architecture diagram |
| `docs/ml_data_dictionary.md` | Every feature, formula, leakage-prevention |
| `docs/monitoring_and_retraining.md` | PSI/KS drift detection, retrain policy |
| `docs/model_card.md` | Intended use, limitations, ethical considerations |
| `reports/model_comparison.md` | All model metrics (generated after `make train`) |
| `reports/monitoring_report.md` | Latest monitoring run (generated after `make monitor`) |
| `reports/part2_report.md` | Full project report |

