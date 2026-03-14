.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Paths ────────────────────────────────────────────────────────────────────
COMPOSE_FILE := docker/compose-dev.yml
VENV         := .venv
PYTHON       := $(VENV)/bin/python
PYTEST       := $(PYTHON) -m pytest
STREAMLIT    := $(VENV)/bin/streamlit
UV           := uv

# ── Colours (disable with NO_COLOR=1) ────────────────────────────────────────
ifndef NO_COLOR
  GREEN  := \033[0;32m
  YELLOW := \033[1;33m
  BLUE   := \033[0;34m
  RED    := \033[0;31m
  NC     := \033[0m
else
  GREEN  :=
  YELLOW :=
  BLUE   :=
  RED    :=
  NC     :=
endif

# ── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@printf "$(BLUE)Codebase RAG — available targets$(NC)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ── Virtual environment ──────────────────────────────────────────────────────
$(VENV)/bin/activate:
	$(UV) venv --python 3.12 --no-config

.PHONY: venv
venv: $(VENV)/bin/activate ## Create venv and install all deps
	$(UV) sync --no-config --extra dev

# ── Docker services ──────────────────────────────────────────────────────────
.PHONY: services-start
services-start: ## Start Docker services (Qdrant, Langfuse, Ollama)
	@printf "$(BLUE)Starting services…$(NC)\n"
	@set -a; [ -f .env ] && . ./.env; set +a; \
		docker compose -f $(COMPOSE_FILE) up -d
	@printf "$(BLUE)Pulling LLM model (this may take a while on first run)…$(NC)\n"
	@set -a; [ -f .env ] && . ./.env; set +a; \
		MODEL=$${LLM_MODEL_NAME:-sam860/LFM2:350m}; \
		printf "$(BLUE)Model: $$MODEL$(NC)\n"; \
		docker exec codebase-rag-ollama ollama pull "$$MODEL"
	@printf "$(GREEN)Services started.$(NC)\n"

.PHONY: services-stop
services-stop: ## Stop Docker services
	docker compose -f $(COMPOSE_FILE) down

.PHONY: services-restart
services-restart: services-stop services-start ## Restart Docker services

.PHONY: services-status
services-status: ## Show Docker service status
	docker compose -f $(COMPOSE_FILE) ps

.PHONY: services-logs
services-logs: ## Tail Docker service logs
	docker compose -f $(COMPOSE_FILE) logs -f

.PHONY: services-clean
services-clean: ## Remove all containers and volumes (destructive)
	@printf "$(YELLOW)This will remove all containers and volumes. Data will be lost.$(NC)\n"
	@read -p "Are you sure? (y/n) " -n 1 -r && echo && \
		if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
			docker compose -f $(COMPOSE_FILE) down -v; \
			printf "$(GREEN)Environment cleaned.$(NC)\n"; \
		fi

# ── Setup ────────────────────────────────────────────────────────────────────
.PHONY: setup
setup: venv ## Initial dev setup: venv + .env file
	@mkdir -p docker
	@if [ ! -f .env ]; then \
		printf "$(BLUE)Creating .env from .env.example…$(NC)\n"; \
		cp .env.example .env; \
	fi
	@printf "$(GREEN)Setup complete.$(NC)\n"

# ── Application ──────────────────────────────────────────────────────────────
.PHONY: app
app: venv ## Start the Streamlit app
	$(STREAMLIT) run src/codebase_rag/app/main.py

.PHONY: ingest
ingest: venv ## Run data ingestion (use REPO= to specify a repo URL)
ifdef REPO
	$(PYTHON) scripts/ingest.py --repo $(REPO) --no-cache
else
	$(PYTHON) scripts/ingest.py
endif

# ── Testing ──────────────────────────────────────────────────────────────────
PYTEST_COV := --cov=src/codebase_rag --cov-report=term --cov-report=xml:coverage.xml

.PHONY: test
test: venv ## Run unit + integration + e2e tests with coverage
	$(PYTEST) tests/unit/ tests/integration/ -m "not performance and not evaluation" $(PYTEST_COV)

.PHONY: test-unit
test-unit: venv ## Run unit tests only
	$(PYTEST) tests/unit/ $(PYTEST_COV)

.PHONY: test-integration
test-integration: venv ## Run integration tests only
	$(PYTEST) -m integration $(PYTEST_COV)

.PHONY: test-e2e
test-e2e: venv ## Run end-to-end tests only
	$(PYTEST) -m e2e $(PYTEST_COV)

.PHONY: test-performance
test-performance: venv ## Run performance tests
	$(PYTEST) -m performance $(PYTEST_COV)

.PHONY: test-evaluation
test-evaluation: venv ## Run evaluation tests
	$(PYTEST) -m evaluation $(PYTEST_COV)

.PHONY: test-all
test-all: venv ## Run all tests (including performance + evaluation)
	$(PYTEST) $(PYTEST_COV)

# ── Linting / type-checking ──────────────────────────────────────────────────
.PHONY: lint
lint: venv ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/ scripts/

.PHONY: format
format: venv ## Auto-format with ruff
	$(PYTHON) -m ruff format src/ tests/ scripts/

.PHONY: typecheck
typecheck: venv ## Run mypy
	$(PYTHON) -m mypy src/

# ── SonarQube ────────────────────────────────────────────────────────────────
SONAR_TOKEN ?= $(shell cat .sonar-token 2>/dev/null)

.PHONY: sonar-start
sonar-start: ## Start SonarQube, create project, generate token
	@bash scripts/sonar_setup.sh

.PHONY: sonar-scan
sonar-scan: ## Run sonar-scanner (reads token from .sonar-token)
	docker run --rm --network host --platform linux/amd64 \
		-e SONAR_HOST_URL="http://localhost:9000" \
		-e SONAR_TOKEN="$(SONAR_TOKEN)" \
		-v "$$(pwd):/usr/src" \
		sonarsource/sonar-scanner-cli:5

.PHONY: sonar-report
sonar-report: ## Fetch results from SonarQube into sonar-report.md
	@bash scripts/sonar_report.sh

.PHONY: sonar-stop
sonar-stop: ## Stop and remove SonarQube container
	docker stop sonarqube && docker rm sonarqube
	@printf "$(GREEN)SonarQube stopped and removed.$(NC)\n"

# ── Cleanup ──────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artifacts, caches, coverage files
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov coverage.xml test_results
	rm -f data/cache/bm25_retriever.pkl data/cache/ingest_stats.json data/cache/processed_documents.pkl
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
