# Monitoring and Retraining Strategy

## Why Monitor in Production?

ML models degrade silently. Unlike software bugs that raise exceptions,
a model that was 85% accurate in training may silently drift to 60% accuracy
six months later because:
- The input distribution shifts (seasonal weather changes, new pollution sources)
- Sensor calibration drifts
- The relationship between features and AQI changes (e.g. new regulations)

Active monitoring catches these issues before users notice bad predictions.

## Three Monitoring Layers

### 1. Data Quality (`src/monitoring/monitor.py::check_data_quality`)

Checks the most recent N days of `ml.features_daily`:

| Check | Metric | Threshold | Alert |
|---|---|---|---|
| Freshness | Hours since latest data | > `STALE_DATA_HOURS` (default 48h) | Stale data flag |
| Completeness | Missing rate across feature columns | > `MISSING_RATE_THRESHOLD` (20%) | Retrain trigger |
| Range | Rows with AQI < 0 or > 600 | Any | Warning |

### 2. Feature Drift (`src/monitoring/monitor.py::check_feature_drift`)

Compares the distribution of each feature over the last 14 days against the
reference distribution from the training set (first 70% of historical data).

**Methods:**

#### PSI (Population Stability Index)
```
PSI = Σ (Pactual - Pexpected) × ln(Pactual / Pexpected)
```
| PSI | Interpretation |
|---|---|
| < 0.1 | No significant shift |
| 0.1–0.2 | Moderate shift — monitor closely |
| > 0.2 | **Significant drift → retrain** |

#### KS Test (Kolmogorov-Smirnov)
A non-parametric test comparing two cumulative distribution functions.
- `p < 0.05` → distributions are significantly different → flag as drifted

**Trigger**: retrain when ≥ `DRIFT_FEATURE_COUNT_THRESHOLD` (default 3) features
are drifted by EITHER method.

### 3. Model Performance (`src/monitoring/monitor.py::check_model_performance`)

Joins `ml.prediction_log` with `analytics.daily_aqi` to compute:
- Rolling RMSE and MAE (regression)
- Rolling macro-F1 (classification)

**Trigger**: retrain when rolling RMSE > baseline RMSE × (1 + `RMSE_DEGRADATION_THRESHOLD`)
(default 20% degradation).

## Retraining Policy (`src/monitoring/retrain_trigger.py`)

### Trigger Conditions (any one is sufficient)

| Priority | Condition | Threshold | Notes |
|---|---|---|---|
| 1 | Feature drift | ≥ 3 features drifted | PSI > 0.2 OR KS p < 0.05 |
| 2 | Performance degradation | Rolling RMSE > baseline × 1.20 | |
| 3 | Data quality | Missing rate > 20% | Usually indicates pipeline failure |
| 4 | Scheduled | Every 30 days | Regardless of other signals |

### Promotion Gate

Even after retraining, the new model is NOT automatically promoted to production.
`src/ml/register_best.py` compares the new model's validation metric against the
current production model:
- New model wins → set `production` alias to new version
- New model loses → keep current production model, log the comparison

This prevents a bad training run (caused by, say, temporarily missing data) from
silently downgrading production.

### Airflow Automation

`dags/aqi_ml_pipeline.py` runs daily and implements this logic as:
```
refresh_features → run_monitoring → branch:
    if retrain_triggered → train_models → register_if_better
    else                 → no_retrain (end)
```

## Changing Thresholds

All thresholds are environment variables. To tighten monitoring:
```bash
# In .env
PSI_DRIFT_THRESHOLD=0.1      # flag drift sooner
KS_PVALUE_THRESHOLD=0.10     # less stringent KS test
RMSE_DEGRADATION_THRESHOLD=0.10  # flag 10% degradation instead of 20%
RETRAIN_EVERY_DAYS=14        # retrain every 2 weeks instead of monthly
```

No code changes required.
