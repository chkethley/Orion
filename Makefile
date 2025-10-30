.PHONY: help install install-dev clean test lint format docker-build docker-up docker-down jupyter train serve

help:
	@echo "Orion AGI Development Hub - Available Commands"
	@echo "================================================"
	@echo "install          : Install production dependencies"
	@echo "install-dev      : Install development dependencies"
	@echo "clean            : Clean build artifacts and cache"
	@echo "test             : Run tests with coverage"
	@echo "lint             : Run linting checks"
	@echo "format           : Format code with black and isort"
	@echo "docker-build     : Build Docker images"
	@echo "docker-up        : Start all Docker services"
	@echo "docker-down      : Stop all Docker services"
	@echo "jupyter          : Start Jupyter lab"
	@echo "train            : Run training pipeline"
	@echo "serve            : Start API server"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/
	pylint src/

format:
	black src/ tests/
	isort src/ tests/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

jupyter:
	docker-compose up jupyter

train:
	python -m src.core.train

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
