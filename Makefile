.PHONY: install test ingest clean-air clean-weather up down

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

ingest:
	python -m src.ingestion.openaq_api
	python -m src.ingestion.weather_api

up:
	docker compose up -d

down:
	docker compose down
