.PHONY: install install-ml test backfill features train register serve monitor \
        docker-up docker-down clean ingest

# ── Part 1: original targets (unchanged) ─────────────────────────────────────

install:
	pip install -r requirements.txt

ingest:
	python -m src.ingestion.openaq_api
	python -m src.ingestion.weather_api

up:
	docker compose up -d

down:
	docker compose down

# ── Part 2: ML pipeline targets ───────────────────────────────────────────────

install-ml:
	# Install ML/API dependencies (separate from Airflow requirements)
	pip install -r requirements-ml.txt

backfill:
	# Pull ≥12 months of historical AQ + weather data from Open-Meteo APIs
	# and load into core.air_quality + core.weather (idempotent, safe to re-run)
	python -m src.ingestion.backfill_history

features:
	# Build ml.features_daily from analytics.daily_aqi (lag/rolling/calendar features)
	python -m src.ml.features

train:
	# Train all regression + classification models, log to MLflow, save plots
	python -m src.ml.train

register:
	# Promote best model from each experiment to 'production' alias in registry
	python -m src.ml.register_best

serve:
	# Run FastAPI inference server locally (models loaded from MLflow registry)
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

monitor:
	# Run data quality + drift + performance checks, save report
	python -m src.monitoring.monitor

streamlit:
	# Launch the Streamlit dashboard locally
	streamlit run app/streamlit_app.py --server.port 8501

# ── Docker ───────────────────────────────────────────────────────────────────

docker-up:
	# Build images and start all services (postgres, mlflow, api, streamlit)
	docker compose up --build -d
	@echo "Services starting …"
	@echo "  Postgres:   localhost:5432"
	@echo "  MLflow UI:  http://localhost:5001"
	@echo "  API:        http://localhost:8000/docs"
	@echo "  Streamlit:  http://localhost:8501"
	@echo "Use 'docker compose logs -f api' to watch the API startup."

docker-down:
	docker compose down

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	# Run all tests including Part 2 feature/API/monitoring tests
	pytest tests/ -v --tb=short

# ── Full ML pipeline (convenience) ───────────────────────────────────────────

ml-pipeline: backfill features train register
	@echo "Full ML pipeline complete. Run 'make serve' or 'make docker-up' to start the API."

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
