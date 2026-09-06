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
_(Fill in once you've run the pipeline against real data — e.g., "Average AQI over the observed period was X, categorized as Moderate.")_

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
