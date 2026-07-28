<p align="center">
  <img src="src/codebase_rag/app/logo.png" alt="Codebase RAG" width="200">
</p>

<h1 align="center">Codebase RAG</h1>

<p align="center">
  <strong>Ask questions about any codebase. Runs entirely on your machine.</strong><br>
  Built with LangChain · Qdrant · Local LLMs · Streamlit
</p>

<p align="center">
  <a href="https://github.com/aschiltmansnavara/codebase-rag/actions/workflows/ci.yml"><img src="https://github.com/aschiltmansnavara/codebase-rag/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/coverage-83%25-brightgreen.svg" alt="Coverage: 83%">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"></a>
</p>

## Why This Project?

- **Fully local.** Runs entirely on your machine. Your code never leaves your hardware.
- **BM25 retrieval by default.** The eval framework ran the same test set through vector-only, BM25-only, and hybrid RRF retrieval and measured recall for each, rather than assuming hybrid would win; BM25-only came out ahead, so that's what the app queries with. See the [retrieval ablation](evals/ablation.md).
- **Evaluated, not just vibes.** Ships with a reproducible evaluation framework; the current test set is 30 questions. See [Evaluation Results](docs/evaluation-results.md) for the retriever ablation and the model-size comparison.
- **Observable.** Optional Langfuse integration traces every retrieval and generation step, so you can debug quality issues instead of guessing.
- **Documented decisions.** Architecture Decision Records explain *why* each technology was chosen, not just *what* was used. See the [ADR index](docs/adr-index.md).
- **Batteries included.** `make services-start` gives you the app, vector DB, LLM server, and tracing dashboard. No manual setup.

## Features

**Retrieval Design**
- **BM25 search.** The app queries with BM25 keyword retrieval by default. `VectorRetriever` and `HybridRetriever` (Reciprocal Rank Fusion, weighted 0.7/0.3) remain in the codebase and are exercised by the eval ablation, but no user query goes through them.
- **Language-aware chunking.** Naively splitting code by token count breaks at arbitrary lines, destroying context. Python and Markdown/RST get structure-aware splitting; `.ipynb` notebooks get their own strategy that splits code cells as Python and markdown cells as Markdown; everything else uses generic recursive splitting.
- **Source citations.** Every answer includes the source files and repositories it drew from, so answers are verifiable.

**Infrastructure Choices**
- **Fully local stack.** Ollama, LM Studio, llama.cpp, vLLM, or Jan for inference; Qdrant for vectors; SQLite for chat history. No external API calls, no data egress.
- **Multi-repo ingestion.** Clone and index any public GitHub repository from the UI or CLI.
- **Idempotent ingestion.** Deterministic chunk IDs make re-ingestion safe and duplicate-free, so it's safe to run repeatedly in scheduled jobs or CI.

**Developer Experience**
- **Local LLM inference.** Choose your backend: Ollama, LM Studio, llama.cpp, vLLM, or Jan. Supports any model these platforms can run.
- **Conversation memory.** Multi-turn conversations with persistent SQLite-backed chat history.
- **LLM observability.** Optional Langfuse integration for tracing retrieval and generation with per-span metrics.

## Architecture

```mermaid
graph TD
    UI["Streamlit UI<br/><i>chat, repo management</i>"]
    RAG["RAG Chain<br/><i>LangChain pipeline</i>"]
    HS["BM25 Retrieval<br/><i>keyword search</i>"]
    LLM["Local LLM Backend<br/><i>Ollama, LM Studio, etc</i>"]
    QD["Qdrant<br/><i>Vector Database</i>"]
    LF["Langfuse<br/><i>LLM Observability</i>"]

    UI --> RAG
    RAG --> HS
    RAG --> LLM
    UI -.->|ingest| QD
    UI -.->|ingest| HS
    RAG -.->|optional| LF

    style LF stroke-dasharray: 5 5
```

**Data flow:**

1. **Ingest.** `GitLoader` clones a repo → `DocumentProcessor` splits files into chunks using language-specific strategies → chunks are embedded with `sentence-transformers/all-mpnet-base-v2` and stored in Qdrant, with a parallel BM25 index built for keyword search.
2. **Retrieve.** User query hits the BM25 retriever, which returns the top-k matching documents.
3. **Generate.** Retrieved documents are formatted into a context prompt and sent to the configured LLM backend (Ollama, LM Studio, llama.cpp, vLLM, or Jan). The `RAGChain` handles conversation memory, prompt construction, and Langfuse tracing.
4. **Persist.** Chat history is stored in SQLite. Vector data lives in Qdrant. Both survive container restarts via Docker volumes.

## Project Structure

```
codebase-rag/
├── src/codebase_rag/
│   ├── app/              # Streamlit UI (main.py, runtime.py, state.py, ui_chat.py, ui_sidebar.py)
│   ├── config.py         # Environment-based configuration (singleton)
│   ├── data_ingestion/   # Git cloning, document processing, chunking
│   ├── database/         # Qdrant store, SQLite chat storage, embeddings
│   ├── llm/              # Local LLM backends (Ollama, OpenAI-compatible), RAG chain with Langfuse tracing
│   ├── retrieval/        # Vector search, BM25 search, hybrid retriever
│   └── services/         # Native folder picker
├── Makefile              # Development task runner (make help)
├── scripts/
│   └── ingest.py         # Repository ingestion pipeline (single and multi-repo)
├── docker/
│   ├── compose-dev.yml   # Full-stack Docker Compose
│   ├── Dockerfile        # App container (Python 3.12)
│   └── entrypoint.sh     # Auto-ingest + model pull on first boot
├── evals/                # Evaluation framework and results
├── tests/                # Unit, integration, e2e, performance tests
└── docs/                 # ADRs and design documentation
```

## Getting Started

Quick start: `make services-start` → open http://localhost:8501.

The default model is small enough to run on any machine, and answers accordingly; set `LLM_MODEL_NAME` (env or compose) to a larger Ollama model for better answers. See [Evaluation Results](docs/evaluation-results.md) for the numbers behind that tradeoff.

See the [setup guide](docs/getting-started.md) for Docker and local installation, repository ingestion, and example queries.

## Configuration

All settings are configured via environment variables or `.env`. See the full [configuration reference](docs/configuration.md).

### LLM Backends

By default, the app uses Ollama for inference. To use a different backend, set `LLM_PROVIDER`:

**Ollama** (default)
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

**LM Studio**
```bash
LLM_PROVIDER=openai-compat
LLM_BASE_URL=http://localhost:1234/v1
# LLM_API_KEY is optional
```

**llama.cpp server**
```bash
LLM_PROVIDER=openai-compat
LLM_BASE_URL=http://localhost:8000/v1
```

**vLLM**
```bash
LLM_PROVIDER=openai-compat
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=token-abc123  # if authentication is enabled
```

**Jan**
```bash
LLM_PROVIDER=openai-compat
LLM_BASE_URL=http://localhost:1337/v1
```

All OpenAI-compatible backends use the same interface, so you can switch between them by just changing `LLM_BASE_URL`.

## Command-Line Interface

Use `codebase-rag` to query and explore codebases from shell scripts, git hooks, and CI pipelines. All output goes to stdout (clean for piping), while diagnostics go to stderr.

### Query Command

Search for code snippets:

```bash
codebase-rag query "where is error handling implemented?"
```

Output in compact format (path, score, snippet):

```
src/app/handlers.py (0.95)
def handle_error(error):
    logger.error("Error occurred: %s", error)
    return {"status": "error", "message": str(error)}
```

Output as JSON for programmatic use:

```bash
codebase-rag query "retry logic" --format json | jq '.[] | .path'
```

Limit results and filter by repository:

```bash
codebase-rag query "database query" --k 3 --repo my-repo
```

### Ask Command

Get a full natural-language answer grounded in the codebase:

```bash
codebase-rag ask "explain the ingestion pipeline"
```

### Piping into LLM CLIs

Compose codebase retrieval with other tools. Example with `claude-cli`:

```bash
codebase-rag query "where is authentication?" | claude "Explain this code and suggest security improvements"
```

### Using in Shell Scripts and Hooks

Pre-commit hook that checks for debugging statements:

```bash
codebase-rag query "debugger" --format compact | grep -q pdb && echo "Debugger found!"
```

Git hook to add context to commit messages. Exit code 2 means "no results", not
a failure, so a `set -e` hook has to tolerate it explicitly or it aborts the
commit on the common case of a query that legitimately matches nothing:

```bash
context=$(codebase-rag query "$(cat /tmp/commit-msg)" --k 2 --format compact) || [ $? -eq 2 ]
if [ -n "$context" ]; then
    echo -e "\nContext:\n$context" >> /tmp/commit-msg
fi
```

## HTTP API

`make api` starts a FastAPI server (`uvicorn`) alongside Streamlit, giving coding agents (Copilot, Claude Code, OpenCode, Cursor, aider) direct, token-budgeted access to the retrieval stack instead of falling back to grep/whole-file reads. It binds to `127.0.0.1` by default (`API_HOST`/`API_PORT`) — **exposing it beyond localhost requires adding authentication first; there is none today.**

- `POST /search` — `{"query": str, "k": int, "repo": str | null, "token_budget": int, "format": "json" | "compact"}`. Ranks chunks with the hybrid retriever, drops overlapping chunks from the same file, and stops once the combined token estimate would exceed `token_budget` (default 2000). Each result has `path`, `start_line`, `end_line`, `score`, `snippet`, `token_estimate`. `format=compact` returns plain text (`path:start-end (score)` + snippet) instead of a JSON envelope.
- `POST /answer` — `{"question": str, "stream": bool}`. Runs the full RAG chain and returns `answer` plus `sources` (file path + line range). `stream: true` returns Server-Sent Events (`event: token`, then a final `event: done` carrying `sources`).
- `GET /repos` — ingested repositories with freshness metadata (last-ingest time, and the indexed HEAD SHA for git-backed repos).
- `POST /ingest` — `{"source": str}`, accepting a git URL or a local filesystem path. Local paths are ingested from the working tree in place, without cloning. Re-ingesting diffs files by content hash so only changed files re-embed; unchanged chunks are left untouched. Returns a job status immediately (`202`); a second ingest while one is running gets `409`. Poll `GET /ingest/status` for completion.

## Development

The `Makefile` is the primary development interface. Run `make help` for the full list.

## License

This project is licensed under MIT. See the [LICENSE](LICENSE) file for details.
