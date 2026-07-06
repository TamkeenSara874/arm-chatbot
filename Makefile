.PHONY: install dev run stop migrate test test-unit lint format audit clean seed build-backend build-frontend build

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
	pre-commit install

# ── Development ───────────────────────────────────────────────────────────────
run:
	docker compose up

run-detached:
	docker compose up -d

stop:
	docker compose down

stop-clean:
	docker compose down -v

migrate:
	alembic upgrade head

migrate-new:
	alembic revision --autogenerate -m "$(name)"

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest -v

test-unit:
	pytest tests/unit -v --no-header

test-integration:
	pytest tests/integration -v -m integration

lint:
	ruff check src tests

format:
	ruff format src tests

audit:
	pip-audit
	bandit -r src -ll

smoke-test:
	python scripts/smoke_test.py

# ── Seed & demo ───────────────────────────────────────────────────────────────
seed:
	python scripts/seed.py

# ── Deployment (see docs/runbook.md) ──────────────────────────────────────────
build-backend:
	docker build -f infra/docker/backend/Dockerfile -t arm-chatbot-backend:local .

# VITE_API_URL defaults to empty (relative /api/v1/... paths); override with
# `make build-frontend VITE_API_URL=https://your-backend-host` for a real deploy.
build-frontend:
	docker build -f infra/docker/frontend/Dockerfile -t arm-chatbot-frontend:local \
		--build-arg VITE_API_URL=$(VITE_API_URL) .

build: build-backend build-frontend

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
