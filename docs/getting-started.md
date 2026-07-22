# Getting Started

## Option A: Docker (recommended)

Qdrant, Ollama, Langfuse, and the Streamlit app all start in one command.

```bash
git clone <your-repo-url>
cd codebase-rag
make services-start
```

`make services-start` starts all Docker services and pulls the configured LLM model into Ollama automatically. Open http://localhost:8501 once the app container is healthy.

> **Manual alternative:** `docker compose -f docker/compose-dev.yml up -d` starts the containers but does not pull the model, so you'll need to run `ollama pull sam860/LFM2:350m` separately.

**Useful services:**

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit app | http://localhost:8501 | Chat interface |
| Qdrant dashboard | http://localhost:6333/dashboard | Vector DB inspection |
| Langfuse | http://localhost:3000 | LLM tracing (if enabled) |
| Ollama | http://localhost:11434 | LLM API |

## Option B: Local development

Prerequisites: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), a running Qdrant instance, a running Ollama instance.

```bash
# Create venv, install all deps, and copy .env.example → .env
make setup

# Start Qdrant and Ollama via Docker (skip if already running)
make services-start

# Ingest a repository
make ingest REPO=https://github.com/<owner>/<repo>

# Start the app
make app
```

Or manually:

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

# Force re-index (ignores content hashes)
python scripts/ingest.py --repo https://github.com/owner/repo --force
```

Ingestion is idempotent by default: unchanged chunks are skipped, modified chunks are updated, and no duplicates are created.

## Example Queries

After ingesting a repository:

- "What does this project do?"
- "How is the codebase structured?"
- "What are the main classes and modules?"
- "How do I get started contributing?"
