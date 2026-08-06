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
| `RETRIEVER` | `bm25` | Retriever the HTTP API serves search and answer requests from: `bm25` or `hybrid`. `bm25` matches the Streamlit app's default, on ablation evidence in `evals/deprecated/ablation.md` (hit rate 0.6552 vs 0.5862 for hybrid; those figures are deprecated but the decision stands). `HybridRetriever` is still built and used by the eval ablation and by the ingestion pipeline's duplicate-detection search regardless of this setting. |

## Generation settings (all backends)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_NUM_CTX` | `8192` | Context window (tokens), used to size the prompt budget for **whichever backend `LLM_PROVIDER` selects**, not just Ollama. Raising it increases memory use; lower it on constrained hardware. Below roughly 1780 (with the default `max_tokens=1024`) the remaining prompt budget can't hold even one context chunk, and the app refuses to start rather than answer every question with no context. If your OpenAI-compatible server's actual context window differs from Ollama's, set this to match it. |

## Ollama backend

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API URL. The default targets a natively installed Ollama. The containerized Ollama (`make services-start PROFILE=ollama` or `PROFILE=full`) publishes on host port `11435` instead, to avoid shadowing a native install; point this at `http://127.0.0.1:11435` to use it from the host, or use `host.docker.internal` per the note below when connecting from inside another container. **The two are not interchangeable on macOS.** Containers run inside a Linux VM with no Metal passthrough, so `11435` is CPU-only there: roughly 73 tok/s generation against 380 for the native endpoint ([measurements and caveats](docker-model-runner-findings.md)). On Linux with the NVIDIA container runtime the container should have GPU access and the choice costs little, though that was not measured. The sidebar reports which endpoint resolved and whether the model is GPU-resident. |

## OpenAI-compatible backend

Use when `LLM_PROVIDER=openai-compat` to connect to LM Studio, llama.cpp server, vLLM, Jan, or any other server implementing the OpenAI chat-completions API.

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | _(empty)_ | Base URL of the OpenAI-compatible server. Trailing slashes are stripped. From inside the Docker app container, `localhost` refers to the container itself; point this at `host.docker.internal` instead if the server runs on the host. |
| `LLM_API_KEY` | _(empty)_ | API key if required by the server; sent as a Bearer token on every request, including the `/models` availability check. Most local servers don't require it. |

Each server's own default port: LM Studio serves on `http://localhost:1234/v1`, llama.cpp server and vLLM both default to `http://localhost:8000/v1`, and Jan defaults to `http://localhost:1337/v1`. `LLM_API_KEY` is typically only needed for vLLM when it's started with authentication enabled.

## Docker Model Runner

Model Runner serves an Ollama-compatible API, so it uses the Ollama settings above rather than the OpenAI-compatible ones: set `LLM_PROVIDER=ollama` and `OLLAMA_BASE_URL=http://localhost:12434`. It runs the inference engine as a host process rather than inside the Docker VM, so unlike the containerized Ollama it is GPU-accelerated on macOS. That does not make it a drop-in match for a native Ollama: its prompt evaluation measured roughly half native Ollama's, which matters more than generation speed for RAG, where every query carries a long retrieved context.

The endpoint is disabled by default and connections are refused until you run `docker desktop enable model-runner --tcp=12434`. Model names are normalized on the way back out: `hf.co` expands to `huggingface.co` and the repository path is lowercased, while the quantization tag keeps its case. Pull models with `docker model pull`, which accepts either spelling: `docker model pull hf.co/LiquidAI/LFM2-350M-GGUF:Q8_0` and the normalized `huggingface.co/liquidai/lfm2-350m-gguf:Q8_0` both work. `LLM_MODEL_NAME` has to reproduce the normalized form exactly, so take it from the endpoint's own `/api/tags` rather than retyping what you pulled.

One rough edge to expect. Placement reporting does not work against this backend: its `/api/ps` omits the VRAM field the sidebar reads, so the sidebar names the endpoint but states no placement.

[The Model Runner investigation](docker-model-runner-findings.md) records the measurements behind this section.

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
