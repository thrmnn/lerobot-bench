.PHONY: help install install-dev dev-setup lint format typecheck test test-fast test-cov clean all pre-commit hooks

PYTHON ?= python
PIP ?= pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with all extras
	$(PIP) install -e ".[all]"

install-dev:  ## Editable install with dev extras only
	$(PIP) install -e ".[dev]"

dev-setup: install hooks  ## One-shot contributor setup: install[all] + pre-commit hooks

lint:  ## Run ruff check
	$(PYTHON) -m ruff check src tests scripts

format:  ## Run ruff format
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck:  ## Run mypy
	$(PYTHON) -m mypy src

test:  ## Run all tests
	$(PYTHON) -m pytest

test-fast:  ## Run only fast tests (skip sim and gpu marks)
	$(PYTHON) -m pytest -m "not slow and not gpu and not sim"

test-cov:  ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=lerobot_bench --cov-report=term-missing --cov-report=html

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true

hooks:  ## Install pre-commit hooks
	$(PYTHON) -m pre_commit install

pre-commit:  ## Run pre-commit on all files
	$(PYTHON) -m pre_commit run --all-files

all: lint typecheck test  ## Lint + typecheck + test

# --- bench-specific ---

.PHONY: calibrate sweep-mini sweep-full publish space-deploy dashboard

calibrate:  ## Day 0b: per-policy step latency probe
	$(PYTHON) scripts/calibrate.py

run-one:  ## Single-cell debug: pass `ARGS="--policy ... --env ... --seed N"`
	$(PYTHON) scripts/run_one.py $(ARGS)

reproduce:  ## Verify one published cell: pass `CELL=policy/env/seed` (e.g. act/pusht/0)
	@test -n "$(CELL)" || { echo "usage: make reproduce CELL=policy/env/seed"; exit 2; }
	$(PYTHON) scripts/reproduce_cell.py \
		--policy $(word 1,$(subst /, ,$(CELL))) \
		--env    $(word 2,$(subst /, ,$(CELL))) \
		--seed   $(word 3,$(subst /, ,$(CELL)))

sweep:  ## Generic sweep dispatch: pass `ARGS="--config ... [--max-cells N] [--shuffle SEED]"`
	$(PYTHON) scripts/run_sweep.py $(ARGS)

sweep-mini:  ## Smoke sweep: 2 baselines x 2 envs x 2 seeds x 25 episodes
	$(PYTHON) scripts/run_sweep.py --config configs/sweep_mini.yaml

sweep-full:  ## Full benchmark sweep (overnight)
	$(PYTHON) scripts/run_sweep.py --config configs/sweep_full.yaml

review-results:  ## Sanity-check the partial sweep results.parquet for anomalies
	$(PYTHON) scripts/review_results.py

publish:  ## Push results to HF Hub dataset: pass `ARGS="--results-path ... --manifest-path ... --videos-dir ..."`
	$(PYTHON) scripts/publish_results.py $(ARGS)

space-deploy:  ## Push the Spaces app to HF Spaces git remote
	cd space && git push hf-space main

dashboard:  ## Launch the local-first operator sweep dashboard (Gradio)
	$(PYTHON) dashboard/app.py
