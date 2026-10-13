# Model Card — AQI Next-Day Predictor

## Model Details

| Field | Value |
|---|---|
| Model names | `aqi_regressor`, `aqi_classifier` (MLflow registry) |
| Task | Regression: predict next-day AQI (0–500 scale) |
|  | Classification: predict next-day AQI category (6 classes) |
| Algorithm | Ridge/RandomForest/XGBoost/LightGBM (best chosen by val metric) |
| Training framework | scikit-learn, XGBoost, LightGBM |
| Experiment tracking | MLflow (experiments: aqi-regression, aqi-classification) |
| Feature version | v1 (see `docs/ml_data_dictionary.md`) |
| Language | Python 3.11 |

## Intended Use

**In scope:**
- Daily next-day AQI forecasting for Bengaluru (configurable via TARGET_CITY)
- Academic/research use, public health awareness
- Demonstration of MLOps practices (Part 2 of Data Engineering + MLOps course)

**Out of scope:**
- Real-time alerts for health emergencies (no guaranteed SLA)
- Policy decisions without human review
- Generalisation to other cities without retraining on city-specific data

## Training Data

| Field | Details |
|---|---|
| Source | `analytics.daily_aqi` (gold layer, Part 1 pipeline) |
| Underlying raw data | Open-Meteo Air Quality API + Open-Meteo Archive API (historical backfill) |
| Time range | ≥12 months ending ~3 days before the training date |
| City | Bengaluru, India (12.97°N, 77.59°E) |
| AQI calculation | US EPA PM2.5 breakpoint methodology (documented in `create_analytics.py`) |
| Note | AQI is computed from PM2.5 only — the full multi-pollutant EPA composite index uses NO2, SO2, CO, O3 and PM10 in addition to PM2.5. Using PM2.5 alone may underestimate AQI on days when other pollutants are dominant. |

## Metrics

*Note: fill these in from actual training run results after `make train`.*

| Model | Split | RMSE | MAE | R² |
|---|---|---|---|---|
| Persistence (baseline) | val | TBD | TBD | TBD |
| Ridge | val | TBD | TBD | TBD |
| RandomForest | val | TBD | TBD | TBD |
| XGBoost | val | TBD | TBD | TBD |
| **Best (production)** | test | TBD | TBD | TBD |

| Model | Split | Accuracy | Macro-F1 |
|---|---|---|---|
| LogisticRegression | val | TBD | TBD |
| RandomForest | val | TBD | TBD |
| **Best (production)** | test | TBD | TBD |

> Numbers are honest: they come from the actual chronological test split,
> not a randomly shuffled test set. A model evaluated on a random split
> would appear to perform better than it actually does in production.

## Limitations

1. **Small data (mitigated)**: Part 1 had only 4 days of data. Part 2 backfills 12–14 months
   via Open-Meteo historical APIs. Still, 12 months is relatively short for capturing
   multi-year seasonal patterns.

2. **PM2.5-only AQI**: AQI in this system is calculated from PM2.5 only (following
   Part 1's methodology). On days where NO2, SO2, or O3 dominates, the AQI may be
   understated. The model learns from and predicts this PM2.5-based AQI, not the full
   multi-pollutant EPA index.

3. **Single city**: The model is trained on Bengaluru data only. Applying it to other
   cities without retraining will likely produce poor predictions (different pollution
   sources, different weather patterns).

4. **No uncertainty quantification**: The model returns a point estimate, not a
   confidence interval. A prediction of AQI=75 could have high variance on days
   with unusual weather conditions.

5. **API rate limits during backfill**: Open-Meteo's free tier has rate limits.
   The backfill script includes retries but will fail clearly if limits are hit.

6. **Spatial granularity**: Open-Meteo provides grid-cell level data (≈1km resolution),
   not individual station measurements. Actual AQI at a specific station may differ
   from the city-average predictions.

## Ethical Considerations

- AQI forecasts can influence decisions about outdoor activities, particularly for
  people with respiratory conditions. Inaccurate predictions could cause harm.
- The model does not predict health outcomes directly — it predicts an AQI index.
- Users should treat predictions as estimates, not ground truth.
- The data source (Open-Meteo) covers Bengaluru but may have data gaps or
  calibration differences compared to official CPCB monitoring stations.

## Monitoring and Update Policy

- Monitored daily via `dags/aqi_ml_pipeline.py`
- Retraining triggered by: drift (PSI > 0.2 on ≥3 features), performance
  degradation (RMSE > 20% above baseline), data quality, or every 30 days
- New models only promoted if they beat the current production model
- Full monitoring policy: `docs/monitoring_and_retraining.md`
