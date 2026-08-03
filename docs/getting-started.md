# Getting Started

## Option A: Local development (recommended)

Recommended because a natively installed inference server reaches your GPU. On macOS a containerized one does not. Option B has the numbers.

Prerequisites: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) installed natively.

```bash
# Install Ollama (macOS/Linux) and pull the default model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull sam860/LFM2:350m
```

```bash
git clone <your-repo-url>
cd codebase-rag

# Create venv, install all deps, and copy .env.example → .env
make setup

# Start Qdrant and Langfuse via Docker
make services-start

# Ingest a repository
make ingest REPO=https://github.com/<owner>/<repo>

# Start the app
make app
```

Open http://localhost:8501 once the app starts.

Or manually, without the Makefile:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv --python 3.12 && uv sync --extra dev
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.17.0
ollama pull sam860/LFM2:350m
python scripts/ingest.py --repo https://github.com/<owner>/<repo>
streamlit run src/codebase_rag/app/main.py
```

**Useful services:**

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit app | http://localhost:8501 | Chat interface |
| Qdrant dashboard | http://localhost:6333/dashboard | Vector DB inspection |
| Langfuse | http://localhost:3000 | LLM tracing (if enabled) |
| Ollama (native) | http://127.0.0.1:11434 | LLM API |

## Option B: Fully containerized (CPU-only inference on macOS)

Qdrant, Ollama, Langfuse, and the Streamlit app all start in one command, with no native Ollama install.

**Check your platform first.** On Linux with the NVIDIA container runtime the containerized Ollama should have GPU access and this option costs you little, though that was not measured. On macOS it is CPU-only, and that was: Docker Desktop runs containers inside a Linux VM, and there is no Metal passthrough into that VM. On the same model and quantization, the containerized endpoint generated roughly 73 tok/s against 380 for a native install, and evaluated prompts at roughly 716 tok/s against 2378. The two endpoints ran different Ollama versions as well as different devices, so read the gap as approximate. [The Model Runner investigation](../docker-model-runner-findings.md) has the full method and caveats.

Use this option for a disposable or reproducible environment, in CI, or on a Linux host with GPU passthrough. On a Mac you actually work on, install Ollama natively and use Option A.

If you want a Docker-shaped setup on macOS without giving up the GPU, Docker Model Runner runs the inference engine as a host process rather than inside the VM. See the [configuration reference](configuration.md#docker-model-runner) for the settings.

```bash
git clone <your-repo-url>
cd codebase-rag
make services-start PROFILE=full
```

`make services-start PROFILE=full` starts every Docker service and pulls the configured LLM model into the Ollama container automatically. Open http://localhost:8501 once the app container is healthy.

If you only want the containerized LLM without the app or api containers, use `make services-start PROFILE=ollama`.

> **Manual alternative:** `docker compose -f docker/compose-dev.yml --env-file .env --profile full up -d` starts the containers but does not pull the model, so you'll need to run `docker exec codebase-rag-ollama ollama pull sam860/LFM2:350m` separately. `--env-file .env` matters here because, unlike `make services-start`, this command doesn't source `.env` into the shell first: without it, compose looks for `.env` next to the compose file (`docker/.env`, which doesn't exist) instead of the repo root, and every configured variable falls back to its default.

**Useful services:**

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit app | http://localhost:8501 | Chat interface |
| Qdrant dashboard | http://localhost:6333/dashboard | Vector DB inspection |
| Langfuse | http://localhost:3000 | LLM tracing (if enabled) |
| Ollama (container) | http://127.0.0.1:11435 | LLM API |

## Ingesting repositories

**From the UI:** Use the sidebar to add a repository URL and click "Ingest". The ingestion runs in the background, so you can continue chatting while it processes.

**From the CLI:**

```bash
# Single repository
python scripts/ingest.py --repo https://github.com/owner/repo

# Multiple repositories
python scripts/ingest.py --repo https://github.com/owner/repo1 --repo https://github.com/owner/repo2

# All repositories from REPO_URLS config
python scripts/ingest.py --all-repos

# Force re-index (drops the existing collection before ingesting)
python scripts/ingest.py --repo https://github.com/owner/repo --force
```

Ingestion is idempotent but not incremental: deterministic chunk IDs mean re-running an ingest never creates duplicates, and stale chunks are removed via delete-by-repo. By default the CLI caches processed chunks per repo HEAD SHA, so re-ingesting an unchanged repo skips re-processing (pass `--no-cache` to force it); either way, every run re-embeds all chunks, since embeddings are never cached.

## Developer commands

`make check` (lint, format, types, unit tests) and `make verify` (`make check` plus the performance/evaluation tiers and OpenSpec validation) are the gates described in `CLAUDE.md`; both run fully offline.

`make audit` audits the locked dependency set against published vulnerability advisories via `pip-audit` and exits nonzero when it finds one. It is not part of `make check`, `make verify`, or the pre-commit hooks: those gates must stay runnable offline and must not fail because of an advisory published upstream with no local code change. Run `make audit` deliberately, when you want to know what's outstanding in the dependency set.

## Example Queries

After ingesting a repository:

- "What does this project do?"
- "How is the codebase structured?"
- "What are the main classes and modules?"
- "How do I get started contributing?"
