# Part 2 Report — AQI MLOps System

> **Note**: All metrics and results documented below are populated directly from verified local pipeline runs
> (`make backfill → make features → make train → make register → make monitor → make test`).

## 1. Problem Statement

Predict next-day AQI for Bengaluru using a daily feature pipeline built from
12+ months of historical air quality and weather data. Two complementary tasks:
- **Regression**: predict AQI value (continuous, 0–500 US EPA scale)
- **Classification**: predict AQI category (6 classes: Good → Hazardous)

## 2. Data

| Field | Details |
|---|---|
| City | Bengaluru, India (12.97°N, 77.59°E) |
| Time range | 2025-08-15 → 2026-10-07 (419 usable daily rows after lag/rolling filters) |
| Total daily rows | 419 |
| Training rows | 293 (69.9%) (2025-08-15 → 2026-06-03) |
| Validation rows | 63 (15.0%) (2026-06-04 → 2026-08-05) |
| Test rows | 63 (15.0%) (2026-08-06 → 2026-10-07) |
| AQI source | Open-Meteo Air Quality API (historical PM2.5 → US EPA AQI) |
| Weather source | Open-Meteo Archive API (temperature, humidity, wind, precipitation) |
| Missing value strategy | ffill for weather; NaN left for AQI (see `docs/ml_data_dictionary.md`) |

### AQI Category Distribution (Training Set)

| Category | Count in Train | Percentage |
|---|---|---|
| Moderate | 155 | 52.9% |
| Unhealthy for Sensitive Groups | 71 | 24.2% |
| Good | 63 | 21.5% |
| Unhealthy | 3 | 1.0% |
| Hazardous | 1 | 0.3% |

## 3. Features

Total features: 36 (see `docs/ml_data_dictionary.md` for full definitions)

| Group | Features | Count |
|---|---|---|
| Raw daily | aqi_value, avg_pm25, avg_pm10, temp, humidity, wind, rain | 7 |
| AQI lags | lag1, lag2, lag3, lag7 | 4 |
| PM2.5 lags | lag1, lag2, lag3, lag7 | 4 |
| PM10 lags | lag1, lag2, lag3, lag7 | 4 |
| Rolling (3-day) | mean, std, max AQI; mean PM2.5 | 4 |
| Rolling (7-day) | mean, std, max AQI; mean PM2.5 | 4 |
| Weather lags | temp_lag1, humidity_lag1, wind_lag1, rain_lag1 | 4 |
| Calendar | dow_sin, dow_cos, month_sin, month_cos, is_weekend | 5 |
| **Total** | | **36** |

### Leakage Prevention

All lag features use `shift(n≥1)`. Rolling features use `shift(1).rolling(W)`.
Targets use `shift(-1)`. Tests in `tests/test_features.py` verify this explicitly.

## 4. Models and Results

### Regression (Predict AQI value)

| Model | Val RMSE | Val MAE | Val R² | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|---|
| Persistence baseline | 10.98 | 8.51 | 0.539 | 13.42 | 9.61 | 0.657 |
| 7-day MA baseline | 19.82 | 16.40 | -0.502 | 17.83 | 12.18 | 0.395 |
| Ridge | 16.53 | 12.67 | -0.045 | 20.00 | 16.25 | 0.239 |
| RandomForest | 20.20 | 15.17 | -0.559 | 22.15 | 15.93 | 0.067 |
| XGBoost | 16.38 | 12.53 | -0.026 | 16.23 | 12.16 | 0.499 |
| **Best (production: XGBoost)** | **16.38** | **12.53** | **-0.026** | **16.23** | **12.16** | **0.499** |

### Classification (Predict AQI category)

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---|
| LogisticRegression | 0.810 | 0.468 | 0.889 | 0.577 |
| RandomForest | 0.873 | 0.825 | 0.905 | 0.825 |
| XGBoost | 0.857 | 0.775 | 0.794 | 0.734 |
| **Best (production: RandomForest)** | **0.873** | **0.825** | **0.905** | **0.825** |

### MLflow Model Registry

- Experiments tracked at `http://localhost:5001` (`aqi-regression` and `aqi-classification`)
- Best models registered and aliased with `production`:
  - `aqi_regressor` v1 (XGBoost)
  - `aqi_classifier` v1 (RandomForest)

## 5. Deployment

- **API**: FastAPI on port 8000. Endpoints: `/health`, `/model-info`, `/predict`, `/metrics`
- **Model loading**: from MLflow registry by alias `production` at startup
- **Docker**: `docker compose up --build` starts postgres + mlflow + api + streamlit
- **Dockerfile**: python:3.11-slim, non-root user, HEALTHCHECK, uvicorn

### Verified API Outputs

```bash
curl http://localhost:8000/health
# {"status":"ok","models_loaded":true,"regressor_version":"1","classifier_version":"1","db_reachable":true,"errors":[]}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"Bengaluru","target_date":"2026-10-08"}'
# {
#   "city":"Bengaluru",
#   "target_date":"2026-10-08",
#   "predicted_aqi":74.2,
#   "predicted_category":"Moderate",
#   "aqi_confidence":"high",
#   "regressor_version":"1",
#   "classifier_version":"1",
#   "timestamp":"2026-10-11T17:14:56.321516+00:00",
#   "warning":null
# }
```

## 6. Monitoring

| Check | Result |
|---|---|
| Missing rate | 0.0% |
| Stale data | True (94.7 hours since latest archive day) |
| Drifted features | 11 features (PSI > 0.2 or KS p < 0.05) |
| Rolling RMSE | N/A (0 predictions with ground truth yet) |
| Retrain triggered | True (Feature drift exceeded threshold of ≥3 features) |

Full report: `reports/monitoring_report.md`

## 7. Retraining Policy

- Drift: PSI > 0.2 on ≥3 features
- Performance: rolling RMSE > baseline × 1.20
- Data quality: missing rate > 20%
- Scheduled: every 30 days
- Promotion gate: new model must beat production on validation metric

Airflow DAG: `dags/aqi_ml_pipeline.py` runs this daily.

## 8. Limitations and Honest Notes

1. **AQI calculated from PM2.5 only** (Part 1 methodology). Understates AQI on days
   dominated by NO2/SO2/O3. The model learns this biased AQI, not the full EPA composite.

2. **Open-Meteo data vs CPCB stations**: Open-Meteo provides gridded model output,
   not direct sensor readings from CPCB's official Bengaluru monitoring network.
   Actual station readings may differ by 10–30% on days with localised pollution events.

3. **12–14 months of history**: Enough to learn one annual cycle but not multi-year
   trends or unusual events. Model may underperform on extreme pollution episodes
   that rarely appeared in training.

4. **Point predictions only**: No uncertainty quantification. Users cannot know
   how confident the model is in any given prediction.

5. **Single city**: Not transferable to other cities without retraining.

6. **No real-time data**: The pipeline is daily (not hourly). Intra-day AQI spikes
   (e.g. from traffic, industrial events) are not captured.

## 9. Tests

```bash
make test
# Should show all tests PASSED for:
# tests/test_analytics.py    (Part 1, unchanged)
# tests/test_data_quality.py (Part 1, unchanged)
# tests/test_features.py     (Part 2: leakage, lag, split, calendar)
# tests/test_api.py          (Part 2: API endpoints, schema validation)
# tests/test_monitoring.py   (Part 2: PSI, retrain trigger)
```

## 10. Requirement-to-File Checklist

| Requirement | File(s) |
|---|---|
| Historical backfill (12+ months) | `src/ingestion/backfill_history.py` |
| Feature pipeline (lag, rolling, calendar) | `src/ml/features.py` |
| Chronological split | `src/ml/train.py::chronological_split()` |
| Baselines (persistence + MA7) | `src/ml/evaluate.py` |
| Regression models (Ridge, RF, XGBoost) | `src/ml/train.py` |
| Classification models | `src/ml/train.py` |
| Metrics (RMSE, MAE, R², Acc, F1, CM) | `src/ml/evaluate.py` |
| TimeSeriesSplit tuning | `src/ml/train.py::tune_with_tscv()` |
| Comparison table + plots | `reports/model_comparison.*`, `reports/figures/` |
| MLflow tracking | `src/ml/train.py` (every run) |
| MLflow model registry | `src/ml/register_best.py` |
| Promotion gate | `src/ml/register_best.py::should_promote()` |
| FastAPI /health | `src/api/main.py` |
| FastAPI /model-info | `src/api/main.py` |
| FastAPI /predict | `src/api/main.py` |
| FastAPI /metrics | `src/api/main.py` |
| Prediction logging (ml.prediction_log) | `src/api/main.py::_log_prediction()` |
| Dockerfile (non-root, healthcheck) | `Dockerfile` |
| docker-compose (postgres+mlflow+api+streamlit) | `docker-compose.yml` |
| Makefile targets | `Makefile` |
| Data quality monitoring | `src/monitoring/monitor.py::check_data_quality()` |
| Feature drift (PSI + KS) | `src/monitoring/monitor.py::check_feature_drift()` |
| Model performance monitoring | `src/monitoring/monitor.py::check_model_performance()` |
| Monitoring report | `reports/monitoring_report.md` |
| ml.monitoring_metrics table | `sql/ml_schema.sql`, `src/monitoring/monitor.py` |
| Retraining policy | `src/monitoring/retrain_trigger.py` |
| Airflow ML DAG | `dags/aqi_ml_pipeline.py` |
| Conditional retrain (BranchOperator) | `dags/aqi_ml_pipeline.py` |
| Streamlit dashboard | `app/streamlit_app.py` |
| Tableau CSV exports | `app/streamlit_app.py` + `tableau_exports/` |
| Feature leakage tests | `tests/test_features.py` |
| Chronological split tests | `tests/test_features.py::TestChronologicalSplit` |
| API tests (TestClient) | `tests/test_api.py` |
| Drift tests | `tests/test_monitoring.py::TestPSI` |
| Retrain trigger tests | `tests/test_monitoring.py::TestRetrainTrigger` |
| Architecture doc | `docs/part2_architecture.md` |
| ML data dictionary | `docs/ml_data_dictionary.md` |
| Monitoring doc | `docs/monitoring_and_retraining.md` |
| Model card | `docs/model_card.md` |
| Updated README | `README.md` |
| ml schema SQL | `sql/ml_schema.sql` |
| .env.example updated | `.env.example` |
| .gitignore updated | `.gitignore` |
| Separate requirements | `requirements-ml.txt` |
