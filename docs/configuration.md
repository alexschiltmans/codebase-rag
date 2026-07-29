# Configuration

All settings are configured via environment variables or a `.env` file in the project root.

## Core settings

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant REST API port |
| `COLLECTION_NAME` | `documents` | Qdrant collection name |
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama` or `openai-compat` (LM Studio, llama.cpp, vLLM, Jan) |
| `LLM_MODEL_NAME` | `sam860/LFM2:350m` | Model name for generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace embedding model |
| `RETRIEVER` | `bm25` | Retriever the HTTP API serves search and answer requests from: `bm25` or `hybrid`. `bm25` matches the Streamlit app's default, on ablation evidence in `evals/ablation.md` (hit rate 0.6552 vs 0.5862 for hybrid). `HybridRetriever` is still built and used by the eval ablation and by the ingestion pipeline's duplicate-detection search regardless of this setting. |

## Generation settings (all backends)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_NUM_CTX` | `8192` | Context window (tokens), used to size the prompt budget for **whichever backend `LLM_PROVIDER` selects**, not just Ollama. Raising it increases memory use; lower it on constrained hardware. Below roughly 1780 (with the default `max_tokens=1024`) the remaining prompt budget can't hold even one context chunk, and the app refuses to start rather than answer every question with no context. If your OpenAI-compatible server's actual context window differs from Ollama's, set this to match it. |

## Ollama backend

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |

## OpenAI-compatible backend

Use when `LLM_PROVIDER=openai-compat` to connect to LM Studio, llama.cpp server, vLLM, Jan, or any other server implementing the OpenAI chat-completions API.

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | _(empty)_ | Base URL of the OpenAI-compatible server (e.g., `http://localhost:1234/v1` for LM Studio). Trailing slashes are stripped. From inside the Docker app container, `localhost` refers to the container itself; point this at `host.docker.internal` instead if the server runs on the host. |
| `LLM_API_KEY` | _(empty)_ | API key if required by the server; sent as a Bearer token on every request, including the `/models` availability check. Most local servers don't require it. |

## Storage settings

| Variable | Default | Description |
|---|---|---|
| `REPO_URLS` | _(empty)_ | Comma-separated repo URLs for batch ingestion |
| `REPO_LOCAL_PATH` | `./data/repos` | Directory for cloned repositories |
| `CHAT_STORAGE_PATH` | `./data/chat_history.db` | SQLite database for chat history |

## Langfuse settings (optional)

| Variable | Default | Description |
|---|---|---|
| `LANGFUSE_ENABLED` | `false` | Enable LLM tracing |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | _(empty)_ | Langfuse secret key |

## Docker-specific settings

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_REPO_URL` | _(empty)_ | Repo to auto-ingest on first Docker start |
