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
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace embedding model. Changing it re-cuts the corpus as well as re-embedding it: chunk size is derived from the model's token window, so an existing index is stale in both respects and has to be rebuilt. |
| `EMBEDDING_MODEL_REVISION` | hub default branch | Commit sha or tag the embedding model name resolves to. Unset means whatever the branch points at on the day of the run, so two runs can embed with different weights under one name. Pin it before publishing a measurement. |
| `EMBEDDING_QUERY_PROMPT` | model's own | Prefix applied to queries. Unset uses the prompts the model declares. |
| `EMBEDDING_DOCUMENT_PROMPT` | model's own | Prefix applied to documents. Unset uses the prompts the model declares. |
| `EMBEDDING_MAX_SEQ_LENGTH` | model's own | Token window override. Chunk size follows it at 1.6 characters per token, capped at 2000 characters: long-context embedders declare windows in the tens of thousands, and a chunk that size matches everything and locates nothing. |
| `EMBEDDING_DTYPE` | checkpoint's own | Load precision: `float32`, `float16`, or `bfloat16`. Unset does not mean float32; the checkpoint decides, and some are stored at bfloat16. Vectors built at one precision should not be queried at another. |
| `RETRIEVER` | `bm25` | Retriever the Streamlit app, the CLI, and the HTTP API all query with: `bm25` or `hybrid`. One setting for every surface, so changing it moves all three together. `bm25` is the default on a 42-question measurement where the two split: hybrid ranks the expected file first more often (26/42 against 21/42) while BM25's contexts carry more of the answer (context_recall 0.564 against 0.479, keyword recall 0.558 against 0.505); see `evals/retrieval-stack-findings.md`. `vector` is not accepted, because the only measurement of it runs without the relevance cutoff and so describes a configuration the app cannot be set to. `HybridRetriever` is still built and used by the eval ablation and by the ingestion pipeline's duplicate-detection search regardless of this setting. Setting `hybrid` also puts the cosine relevance cutoff calibrated for `EMBEDDING_MODEL` on every query from every surface, where under `bm25` no query touches it; an embedding model with no calibrated cutoff then runs the app unfiltered, which is logged as a warning at startup. |

## Optional retrieval stages

Both stages are off by default and apply to every surface, the same way `RETRIEVER` does. They were
measured end to end on this corpus: neither improves hit rate on any retriever, prompt tokens never
fall, and time to first token regresses by 7x to 20x. What they buy is ordering and grounding. See
`evals/retrieval-stack-findings.md` for the per-arm figures.

| Variable | Default | Description |
|---|---|---|
| `RERANK_ENABLED` | `false` | Rescore the configured retriever's top candidates with a local cross-encoder. Enabling it loads the model lazily on the first query, roughly 2GB from the local cache, so the first question after startup pays for it. |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder used when reranking is enabled. |
| `RERANK_MODEL_REVISION` | hub default branch | Commit sha or tag the reranker model name resolves to. Same reasoning as `EMBEDDING_MODEL_REVISION`: unpinned weights move retrieval scores with nothing in the diff to explain it. |
| `RERANK_CANDIDATE_DEPTH` | `50` | How many candidates are pulled from the first stage for rescoring. Deeper costs time and raises the ceiling on what reranking can recover. |
| `REWRITE_ENABLED` | `false` | Expand a terse query with likely identifiers using the local model before retrieval. Expands, never replaces, and falls back to the original query on failure or timeout. |
| `REWRITE_TIMEOUT_S` | `5.0` | Seconds to wait for the expansion before giving up and retrieving on the original query. |
| `REWRITE_MAX_CONCURRENCY` | `1` | How many expansions may run at once. A query that cannot get a slot falls back to the original query immediately rather than queueing behind another's expansion and paying the full timeout. A slot is held until the model call finishes, not until the query gives up, so a timed-out expansion keeps its slot until that call drains and a query arriving in the gap also falls back. Expansions all run against one local model process, so raising this moves the contention into that server rather than removing it; the right value depends on the backend's own parallelism settings. |

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
