# Pipeline Flow

```
OpenAQ API          Open-Meteo API
     │                    │
     ▼                    ▼
extract_openaq       extract_weather        (Airflow tasks, run in parallel)
     │                    │
     ▼                    ▼
data/raw/openaq/*    data/raw/weather/*     (raw landing zone, JSON, untouched)
     │                    │
     ▼                    ▼
clean_openaq          clean_weather         (parse + validate)
     │                    │
     ▼                    ▼
data/staging/*.csv   data/staging/*.csv     (+ data/rejected/*.csv for failures)
     │                    │
     └─────────┬──────────┘
               ▼
        load_analytics                       (join, aggregate, calculate AQI)
               │
               ▼
        core.air_quality / core.weather       (PostgreSQL)
        analytics.daily_aqi
        analytics.station_summary
        analytics.pollutant_summary
               │
               ▼
        sync_lineage                          (writes run metadata to staging.pipeline_runs)
               │
               ▼
        Tableau Public dashboard
```

Every stage above is a separate Airflow task (see `dags/air_quality_pipeline.py`)
with its own retry policy and its own lineage record, so a failure in
one stage doesn't require re-running the whole pipeline from scratch.
