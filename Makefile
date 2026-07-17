.PHONY: install install-dev fetch-data lint format test train evaluate monitor serve docker-build docker-run compose-up clean

install:
	pip install -r requirements.txt

fetch-data:
	python -m src.data.fetch_data

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

test:
	pytest tests/ --cov=src --cov-report=term-missing

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

monitor:
	python -m src.monitoring.drift_check

serve:
	uvicorn src.api.main:app --reload --port 8000

docker-build:
	docker build -t fraud-detection-api .

docker-run:
	docker run -p 8000:8000 fraud-detection-api

compose-up:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache coverage.xml
