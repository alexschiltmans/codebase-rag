.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Paths ────────────────────────────────────────────────────────────────────
COMPOSE_FILE := docker/compose-dev.yml
# --env-file makes compose read the root .env for ${VAR:-default} interpolation
# regardless of cwd or the compose file's own directory (docker/, which has no
# .env). Unlike --project-directory, this only affects variable interpolation:
# it leaves the project name and relative-path resolution for build.context
# and friends untouched, so it doesn't require compose-dev.yml's own paths to
# be rewritten around it. The services-* targets below also `set -a; . ./.env`
# the root file into the shell first, so real env vars are already set before
# compose even looks at --env-file; --env-file matters for a bare
# `docker compose -f $(COMPOSE_FILE)` run by hand outside these targets, and
# for docs/getting-started.md's manual command.
ENV_FILE_FLAG := $(if $(wildcard .env),--env-file .env,)
# PROFILE selects which compose profile to bring up (e.g. PROFILE=full for the
# fully containerized stack, PROFILE=ollama for infrastructure + the LLM
# container). Empty by default: infrastructure only.
PROFILE_FLAG := $(if $(PROFILE),--profile $(PROFILE),)
COMPOSE      := docker compose -f $(COMPOSE_FILE) $(ENV_FILE_FLAG) $(PROFILE_FLAG)
# Teardown targets must reach profiled containers regardless of which PROFILE (if
# any) started them, so they name every profile explicitly instead of inheriting
# PROFILE_FLAG through COMPOSE.
COMPOSE_ALL_PROFILES := docker compose -f $(COMPOSE_FILE) $(ENV_FILE_FLAG) --profile ollama --profile full
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
services-start: ## Start Docker services (Qdrant, Langfuse). PROFILE=ollama adds the LLM container; PROFILE=full adds the app and api containers too.
	@printf "$(BLUE)Starting services…$(NC)\n"
	@set -a; [ -f .env ] && . ./.env; set +a; \
		$(COMPOSE) up -d
	@set -a; [ -f .env ] && . ./.env; set +a; \
		if [ "$(PROFILE)" = "ollama" ] || [ "$(PROFILE)" = "full" ]; then \
			if [ "$${LLM_PROVIDER:-ollama}" = "ollama" ]; then \
				printf "$(BLUE)Waiting for Ollama container to be healthy…$(NC)\n"; \
				attempt=0; \
				until [ "$$(docker inspect -f '{{.State.Health.Status}}' codebase-rag-ollama 2>/dev/null)" = "healthy" ]; do \
					attempt=$$((attempt + 1)); \
					if [ $$attempt -ge 30 ]; then \
						printf "$(RED)Timed out waiting for codebase-rag-ollama to become healthy.$(NC)\n"; \
						exit 1; \
					fi; \
					sleep 2; \
				done; \
				printf "$(BLUE)Pulling LLM model (this may take a while on first run)…$(NC)\n"; \
				MODEL=$${LLM_MODEL_NAME:-sam860/LFM2:350m}; \
				printf "$(BLUE)Model: $$MODEL$(NC)\n"; \
				docker exec codebase-rag-ollama ollama pull "$$MODEL"; \
			else \
				printf "$(BLUE)LLM_PROVIDER=$${LLM_PROVIDER}: skipping Ollama model pull.$(NC)\n"; \
			fi; \
		fi
	@if [ "$(PROFILE)" = "full" ]; then \
		printf "$(GREEN)Services started: Qdrant, Langfuse, Ollama, app, api.$(NC)\n"; \
	elif [ "$(PROFILE)" = "ollama" ]; then \
		printf "$(GREEN)Services started: Qdrant, Langfuse, Ollama.$(NC)\n"; \
	else \
		printf "$(GREEN)Services started: Qdrant, Langfuse.$(NC)\n"; \
		printf "$(BLUE)Run 'make app' to start the Streamlit app on the host, or 'make services-start PROFILE=full' for the fully containerized stack.$(NC)\n"; \
	fi

.PHONY: services-stop
services-stop: ## Stop Docker services
	$(COMPOSE_ALL_PROFILES) down

.PHONY: services-restart
services-restart: services-stop services-start ## Restart Docker services

.PHONY: services-status
services-status: ## Show Docker service status
	$(COMPOSE) ps

.PHONY: services-logs
services-logs: ## Tail Docker service logs
	$(COMPOSE) logs -f

.PHONY: services-clean
services-clean: ## Remove all containers and volumes (destructive)
	@printf "$(YELLOW)This will remove all containers and volumes. Data will be lost.$(NC)\n"
	@read -p "Are you sure? (y/n) " -n 1 -r && echo && \
		if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
			$(COMPOSE_ALL_PROFILES) down -v; \
			printf "$(GREEN)Environment cleaned.$(NC)\n"; \
		fi

# ── Setup ────────────────────────────────────────────────────────────────────
.PHONY: setup
setup: venv pre-commit-install ## Initial dev setup: venv + .env file + pre-commit hooks
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

.PHONY: api
api: venv ## Start the retrieval HTTP API (localhost only by default)
	$(PYTHON) -m codebase_rag.api

.PHONY: ingest
ingest: venv ## Run data ingestion (use REPO= to specify a repo URL)
ifdef REPO
	$(PYTHON) scripts/ingest.py --repo $(REPO) --no-cache
else
	$(PYTHON) scripts/ingest.py
endif

# EVAL_ARGS comes before the literal --judge-model: the flag resolver takes the
# first match in argv, so a caller-supplied judge model must lead to win.
.PHONY: eval
eval: venv ## Run the judged retrieval eval (SLOW, ~2 h for the three arms, costs model time; ask before running)
	$(PYTHON) evals/run_eval.py $(EVAL_ARGS) --judge-model qwen3.5:9b

# ── Testing ──────────────────────────────────────────────────────────────────
PYTEST_COV := --cov=src/codebase_rag --cov-report=term --cov-report=xml:coverage.xml

.PHONY: test
test: venv ## Run unit + integration + e2e tests (needs make services-start)
	$(PYTEST) tests/unit/ tests/integration/ tests/e2e/ -m "not performance and not evaluation" $(PYTEST_COV)

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
	$(PYTHON) -m ruff check src/ tests/ scripts/ evals/

.PHONY: format
format: venv ## Auto-format with ruff
	$(PYTHON) -m ruff format src/ tests/ scripts/ evals/

.PHONY: format-check
format-check: venv ## Check formatting without rewriting files (matches CI)
	$(PYTHON) -m ruff format --check src/ tests/ scripts/ evals/

.PHONY: typecheck
typecheck: venv ## Run mypy
	$(PYTHON) -m mypy src/ tests/ evals/ scripts/

.PHONY: check
check: lint format-check typecheck test-unit ## Fast gate: lint, format, types, unit tests

# ── Static analysis / SARIF ──────────────────────────────────────────────────
# Every analyser that can name a file, a line, and a rule writes SARIF into .review/, and
# merge_sarif.py collapses them into one file. That merged file is what a review reads alongside
# a change's spec delta: located findings on one side, stated intent on the other, instead of a
# raw diff and a guess about which parts were deliberate.
REVIEW_DIR := .review
OPENGREP   := .tools/opengrep

.PHONY: opengrep-install
opengrep-install: ## Fetch the pinned Opengrep binary into .tools/
	@bash scripts/install_opengrep.sh

.PHONY: scan
scan: venv opengrep-install ## Run ruff and Opengrep into .review/*.sarif, then merge
	@mkdir -p $(REVIEW_DIR)
	@printf "$(BLUE)ruff → $(REVIEW_DIR)/ruff.sarif$(NC)\n"
	@$(PYTHON) -m ruff check --output-format sarif --output-file $(REVIEW_DIR)/ruff.sarif \
		src/ tests/ scripts/ evals/ || true
	@printf "$(BLUE)opengrep → $(REVIEW_DIR)/opengrep.sarif$(NC)\n"
	@$(OPENGREP) scan --config semgrep-rules/ --sarif --output $(REVIEW_DIR)/opengrep.sarif . >/dev/null || true
	@$(PYTHON) scripts/merge_sarif.py --review-dir $(REVIEW_DIR)

.PHONY: scan-strict
scan-strict: scan ## Same as scan, but exit non-zero if anything was found
	@$(PYTHON) scripts/merge_sarif.py --review-dir $(REVIEW_DIR) --fail-on-findings

# Run deliberately, not part of the gate: it queries an advisory database and can't run offline.
# `-r` + `--disable-pip` avoids pip-audit building its own resolver venv, whose `ensurepip` aborts
# with SIGABRT under uv-managed interpreters; `--no-emit-project` and `--require-hashes` keep the
# unhashed `-e .` entry out of the export so every dependency in it carries a hash.
.PHONY: audit
audit: venv ## Audit the locked dependency set for known vulnerabilities
	@tmpfile=$$(mktemp) && \
	trap 'rm -f "$$tmpfile"' EXIT && \
	uv export --format requirements-txt --all-extras --no-emit-project -o "$$tmpfile" && \
	$(VENV)/bin/pip-audit -r "$$tmpfile" --disable-pip --require-hashes

# The gate to run before review, commit, and push. Covers every tier that
# works without live services, so it is safe to run anywhere and in a hook.
# scan-strict is in here because it is a CI job: the static analysis job sat red
# for four commits over one finding nobody could see locally, since no local
# target failed on it. It needs the network only the first time, to fetch the
# pinned Opengrep binary into .tools/.
.PHONY: verify
verify: lint format-check typecheck scan-strict ## Full offline gate: check + performance/evaluation tiers + SARIF scan + OpenSpec validation
	$(PYTHON) scripts/check_tracked_imports.py
	$(PYTEST) tests/unit/ tests/performance/ tests/evaluation/ $(PYTEST_COV)
	openspec validate --changes
	openspec validate --specs

# ── Pre-commit ───────────────────────────────────────────────────────────────
.PHONY: pre-commit-install
pre-commit-install: venv ## Install pre-commit hooks (commit, commit-msg, pre-push)
	$(VENV)/bin/pre-commit install
	$(VENV)/bin/pre-commit install --hook-type commit-msg
	$(VENV)/bin/pre-commit install --hook-type pre-push

.PHONY: pre-commit-run
pre-commit-run: venv ## Run all pre-commit hooks on all files
	$(VENV)/bin/pre-commit run --all-files

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
	rm -f data/cache/bm25_retriever.json data/cache/ingest_stats.json data/cache/processed_documents.pkl
	rm -rf data/cache/bm25_corpus
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
