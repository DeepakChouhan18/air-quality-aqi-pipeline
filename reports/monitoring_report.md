# AQI Monitoring Report
Generated: 2026-10-11 23:23 UTC

## 1. Data Quality
| Metric | Value |
|---|---|
| Missing rate | 0.0% |
| Stale data flag | True |
| Hours since latest data | 95.4 |
| Out-of-range rows | 0 |

## 2. Feature Drift
| Metric | Value |
|---|---|
| Features drifted | 11 |
| Drift triggered | True |

### Per-Feature Drift
| Feature | PSI | KS p-value | Drifted |
|---|---|---|---|
| aqi_value | 4.1751 | 0.1065 | DRIFT DETECTED |
| avg_pm25 | 4.1411 | 0.1008 | DRIFT DETECTED |
| avg_pm10 | 5.1599 | 0.1008 | DRIFT DETECTED |
| avg_temperature | 9.2104 | 0.0003 | DRIFT DETECTED |
| avg_humidity | 5.2165 | 0.0122 | DRIFT DETECTED |
| avg_wind_speed | 7.653 | 0.0 | DRIFT DETECTED |
| total_precipitation | 2.0921 | 0.0289 | DRIFT DETECTED |
| aqi_lag1 | 5.2506 | 0.0267 | DRIFT DETECTED |
| aqi_lag7 | 4.0507 | 0.1842 | DRIFT DETECTED |
| pm25_lag1 | 5.2165 | 0.0249 | DRIFT DETECTED |
| aqi_roll7_mean | 5.335 | 0.15 | DRIFT DETECTED |

## 3. Model Performance
| Metric | Value |
|---|---|
| Rolling RMSE | 10.932795715332034 |
| Rolling MAE  | 10.932795715332034 |
| Baseline RMSE| 16.22985036917309 |
| RMSE % change| -0.32637729451297093 |
| Rolling F1   | 1.0 |
| N predictions evaluated | 4 |

## 4. Retraining Decision
| Triggered | True |
| Reason | Feature drift: 11 features exceeded drift threshold (PSI > 0.2 or KS p < 0.05). Threshold is 3 features. |

---
_Thresholds: PSI > 0.2, KS p < 0.05, missing rate > 20%, RMSE degradation > 20%, retrain every 30 days._