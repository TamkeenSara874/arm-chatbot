.PHONY: install dev run stop migrate test test-unit lint format audit clean seed

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

# ── Evaluation (fill in evaluation_report.md) ─────────────────────────────────
eval:
	python scripts/eval_retrieval.py --k 5

load-test:
	python scripts/load_test.py --n 50

smoke-test:
	python scripts/smoke_test.py

# ── Seed & demo ───────────────────────────────────────────────────────────────
seed:
	python scripts/seed.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
