.PHONY: help install dev test lint fmt typecheck check run serve docker docker-demo docker-stop clean

PY ?= python3

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the core package
	$(PY) -m pip install -e .

dev: ## Install with dev tooling
	$(PY) -m pip install -e ".[dev]"

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint
	$(PY) -m ruff check src tests

fmt: ## Autoformat + autofix
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## Static types
	$(PY) -m mypy

check: lint typecheck test ## Everything CI runs

run: ## Whole system: set up, build a corpus, train, evaluate, serve
	./run.sh

serve: ## Run the API + browser client on :8000
	$(PY) -m signsync.cli serve

docker: ## Build the image and serve in a container
	./infrastructure/run-docker.sh

docker-demo: ## Same, but train a synthetic demo model on first boot
	./infrastructure/run-docker.sh --demo --detach

docker-stop: ## Stop and remove the container
	docker rm -f signsync 2>/dev/null || true

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
