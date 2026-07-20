# Configuration

All settings are configured via environment variables or a `.env` file in the project root.

## Core settings

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant REST API port |
| `COLLECTION_NAME` | `documents` | Qdrant collection name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `LLM_MODEL_NAME` | `sam860/LFM2:350m` | Ollama model for generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace embedding model |
| `OLLAMA_NUM_CTX` | `8192` | Ollama context window (tokens). Raising it increases Ollama's memory use; lower it on constrained hardware. Below roughly 1780 (with the default `max_tokens=1024`) the remaining prompt budget can't hold even one context chunk, and the app refuses to start rather than answer every question with no context. |

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
