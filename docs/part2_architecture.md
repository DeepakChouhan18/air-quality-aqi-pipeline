# Part 2 Architecture — MLOps Layer

## Overview

Part 2 extends the Part 1 data engineering pipeline with a complete MLOps system:
feature engineering → experiment tracking → model registry → inference API →
monitoring → automated retraining.

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Part1["Part 1 — Data Engineering"]
        A1["OpenAQ API"] -->|hourly pollutants| R["data/raw/ (landing zone)"]
        A2["Open-Meteo Forecast"] -->|weather| R
        A3["Open-Meteo AQ Archive<br/>(NEW: backfill)"] -->|12+ months history| R
        A4["Open-Meteo Archive<br/>(NEW: backfill)"] -->|weather history| R
        R --> ET["Airflow: air_quality_pipeline DAG<br/>extract → validate → transform → load"]
        ET --> PG[("PostgreSQL\ncore.air_quality\ncore.weather\nanalytics.daily_aqi")]
        ET --> LOG["staging.pipeline_runs\ndata/logs/pipeline_runs.csv"]
    end

    subgraph Part2["Part 2 — MLOps"]
        PG -->|analytics.daily_aqi| FP["src/ml/features.py\nFeature Pipeline\n(lag, rolling, calendar, targets)"]
        FP -->|ml.features_daily| FS[("ml schema\nml.features_daily\nml.prediction_log\nml.monitoring_metrics")]
        FP -->|parquet| DISK["data/ml/*.parquet"]

        FP --> TR["src/ml/train.py\nChronological split 70/15/15\nRidge / RF / XGBoost\nTimeSeriesSplit CV"]
        TR -->|experiments| MLFLOW[("MLflow Tracking\naqi-regression\naqi-classification")]
        MLFLOW --> REG["src/ml/register_best.py\nModel Registry\naqi_regressor @ production\naqi_classifier @ production"]

        REG -->|load by alias| API["FastAPI\nsrc/api/main.py\nGET /health\nGET /model-info\nPOST /predict\nGET /metrics"]
        API -->|prediction_log| FS
        API -->|serving| STRM["Streamlit Dashboard\napp/streamlit_app.py"]

        FS --> MON["src/monitoring/monitor.py\nData quality\nFeature drift (PSI+KS)\nModel performance (rolling RMSE)"]
        MON -->|monitoring_metrics| FS
        MON -->|report| RPT["reports/monitoring_report.md"]
        MON --> RT["src/monitoring/retrain_trigger.py\nRetrain decision policy"]

        RT -->|trigger| MDAG["Airflow: aqi_ml_pipeline DAG\nrefresh_features → monitor\n→ branch → train + register"]
        MDAG --> TR
    end

    subgraph Infra["Infrastructure (Docker)"]
        D1["postgres:16"]
        D2["mlflow server"]
        D3["FastAPI container\n(non-root, healthcheck)"]
        D4["streamlit container"]
    end

    PG --- D1
    MLFLOW --- D2
    API --- D3
    STRM --- D4
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Chronological split only | Time series: random split leaks future into training |
| Shift(n) for lags, closed='left' for rolling | Guarantees no look-ahead at features or rolling stats |
| Same features.py at training and inference | One code path → no train-serve skew |
| MLflow alias 'production', not file path | Model version changes without code changes |
| Separate requirements-ml.txt | Airflow and ML packages have conflicting pins |
| Non-root Docker user | Limits blast radius of container escape |
| PSI + KS for drift | PSI for magnitude, KS for shape — complementary signals |

## File Map

```
src/
├── config.py                  # all settings (Part 1 + Part 2)
├── ingestion/
│   ├── backfill_history.py    # Step 1: historical AQ + weather backfill
│   ├── openaq_api.py          # Part 1: current measurements
│   └── weather_api.py         # Part 1: current weather
├── ml/
│   ├── features.py            # Step 2: feature pipeline (lag, rolling, calendar)
│   ├── train.py               # Step 3: training (chronological split, CV, baselines)
│   ├── evaluate.py            # Step 3: metrics, plots, comparison table
│   └── register_best.py       # Step 4: MLflow model registration + promotion gate
├── api/
│   └── main.py                # Step 5: FastAPI service
└── monitoring/
    ├── monitor.py             # Step 7: data quality + drift + performance
    └── retrain_trigger.py     # Step 7: retrain policy

dags/
├── air_quality_pipeline.py    # Part 1: daily data ingestion (unchanged)
└── aqi_ml_pipeline.py         # Step 8: ML pipeline with conditional retraining

app/
└── streamlit_app.py           # Step 9: 5-page dashboard

sql/
├── schemas.sql                # Part 1: schemas
├── tables.sql                 # Part 1: core tables
├── analytics.sql              # Part 1: gold layer
└── ml_schema.sql              # Part 2: ml schema (features_daily, prediction_log, monitoring_metrics)
```
