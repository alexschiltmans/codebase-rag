# Feature Plan: Codebase RAG as a Token-Saving Context Engine for Coding Agents

**Status:** Draft
**Date:** 2026-07-18 (updated 2026-07-19)

## Thesis

Coding agents (GitHub Copilot agent mode, Claude Code, Cursor, Continue, aider, …) spend the
majority of their *input* tokens on exploration: directory listings, grep round-trips, reading
whole files to find one function, and re-reading after failed guesses. Every one of those tokens
is billed by the API provider, and with local models it costs wall-clock prefill time instead.

This project already owns the expensive machinery to fix that: multi-repo ingestion,
language-aware chunking, and hybrid (vector + BM25) retrieval with an eval framework that
measures each retriever separately (the ablation in `evals/ablation.md` shows the BM25 side
carries exact symbol lookups, which is exactly the query shape coding agents produce). What it
lacks is a way for anything other than the Streamlit UI to consume it.

**The plan: turn the retrieval layer into a service that agents can call.** Instead of an agent
reading five 400-line files to answer "where is retry logic implemented?", it makes one tool call
and gets back the top-k relevant chunks (a few hundred tokens each), each citing the file it came
from. Generation can additionally be offloaded to the local model entirely, so
"explain this subsystem" questions cost zero paid tokens.

Two distinct savings modes fall out of this:

1. **Paid API mode** (Copilot, Claude, OpenAI, …): the paid model stays the reasoning engine but
   receives distilled context instead of raw files → fewer billed input tokens per task, fewer
   exploration turns.
2. **Local model mode** (Ollama, LM Studio): shorter prompts mean proportionally faster prefill
   (time-to-first-token) — the dominant latency cost for local inference — and small models
   answer far more accurately when the answer is already in their context window.

## The other headline selling point: private semantic indexing

Token savings are half the pitch. The other half, and for some users the more important one, is
that this indexes code semantically without anything leaving the machine.

VS Code's built-in semantic codebase search (Copilot's remote workspace index) embeds your code
on GitHub's servers and works best when the repo lives on GitHub. That rules it out precisely
where semantic search would help most: proprietary code an employer won't allow into a cloud
index, client work under NDA, repos on internal GitLab or plain file shares, and air-gapped
environments. Those developers currently get no semantic code search at all — their agents fall
back to grep and file reading, which is both worse and more expensive.

Here the whole index is local: embeddings come from sentence-transformers on the user's machine,
vectors live in a local Qdrant, and generation can stay on a local model. A coding agent pointed
at the Track 2 MCP server gets semantic search over exactly the repos that are forbidden from
cloud indexing. Nothing about the project needs to change to deliver this — it falls out of the
existing fully-local stack — but it should lead the positioning alongside token savings:
"semantic code search for the repos you can't let Copilot index."

---

## Track 1 — Retrieval HTTP API (foundation)

Everything else builds on this. Add a thin FastAPI app alongside the Streamlit UI.

**Endpoints**

| Endpoint | Purpose |
|---|---|
| `POST /search` | Query → ranked chunks. Params: `query`, `k`, `repo` filter, `token_budget`, `format` (`json` \| `compact`) |
| `POST /answer` | Full RAG answer via the local model (existing `RAGChain`), with source citations |
| `GET /repos` | List ingested repositories and index freshness |
| `POST /ingest` | Trigger ingestion of a repo URL or local path (reuses `scripts/ingest.py` pipeline) |

**Design points**

- **Token budgeting is the core feature, not an option.** `/search` should trim, dedupe
  overlapping chunks, and stop adding results once the budget (default ~2000 tokens) is spent.
  Every result carries `path`, `start_line`, `end_line`, `score` so the caller can fetch more
  precisely if needed.
- **`compact` output format:** newline-delimited `path:start-end (score)` + snippet, no JSON
  scaffolding. JSON envelopes are token overhead when the consumer is an LLM.
- **Reuses existing code:** `HybridRetriever.search()` and `RAGChain` are already
  interface-clean; this track is mostly wiring plus a `fastapi`/`uvicorn` dependency.
  `RAGChain.stream()` exists too (added 2026-07-19 for the chat UI), so `/answer` can offer
  token streaming from the start.
- Serve on a configurable port next to Streamlit; add to `compose-dev.yml` and `Makefile`.

**Prerequisite: index freshness for working trees.** Agents operate on the *current* working
tree; a stale index produces wrong answers and wasted agent turns, which erases the savings.
The idempotent ingestion (content hashing, deterministic chunk IDs) already makes incremental
re-ingest safe — add:

- `POST /ingest` accepting a **local path** (not just a git URL), diffing by content hash so only
  changed files re-embed.
- Optional file-watcher mode (`watchdog`) that re-ingests changed files with a debounce.

## Track 2 — MCP server (generic agent integration)

MCP is the one integration that covers nearly every agent at once: GitHub Copilot (VS Code agent
mode), Claude Code, Cursor, Windsurf, Continue, and Zed all speak it. One server, every client.

**Tools to expose**

- `search_codebase(query, repo?, k?, token_budget?)` — returns compact ranked chunks.
  The tool description must actively steer the agent: *"Use this before reading or grepping
  files — returns the most relevant code snippets with the file each came from."* Steering text is
  what converts an installed server into actual token savings.

  Chunk metadata carries no line spans today, so neither the CLI nor a future MCP tool can cite
  line numbers. Promising them in the steering text would be a claim the retrieval layer can't
  honor; adding real line-span tracking to the chunker is its own change.
- `ask_codebase(question, repo?)` — full RAG answer generated by the **local** model. Lets a paid
  agent delegate "understand/summarize" questions for zero billed generation tokens.
- `list_repos()` / `ingest_repo(url_or_path)` — index management from inside the agent session.

**Implementation**

- Official `mcp` Python SDK, stdio transport first (what editors spawn), streamable HTTP later
  for shared/team servers (can then mount inside the Track 1 FastAPI app).
- Entry point: `codebase-rag-mcp` console script in `pyproject.toml`, so client config is a
  one-liner (`uvx`/`uv run codebase-rag-mcp`).
- Document client setup for Copilot (`.vscode/mcp.json`), Claude Code (`claude mcp add`), and
  Cursor in `docs/`.
- Results reuse the Track 1 compact format — MCP tool results are billed as input tokens by the
  calling agent's provider, so terseness here *is* the product.

## Track 3 — LM Studio and OpenAI-compatible backends

`Config.provider` already exists but only Ollama is implemented. Add a second client:

- `OpenAICompatClient` next to `OllamaClient`, selected by `LLM_PROVIDER=openai-compat`, with
  `LLM_BASE_URL` (e.g. `http://localhost:1234/v1` for LM Studio) and optional `LLM_API_KEY`.
  Built on `langchain-openai`'s `ChatOpenAI`, so it slots into `RAGChain` unchanged.
- One client covers LM Studio, llama.cpp server, vLLM, and Jan — they all expose the
  OpenAI-compatible `/v1/chat/completions` surface.
- `check_connection()` equivalent against `/v1/models` so the UI health checks keep working.
- Embeddings stay local via sentence-transformers (already provider-independent); optionally add
  an OpenAI-compatible embeddings backend later for users who want LM Studio to own everything.

## Track 4 — Local model speed and accuracy

Retrieval quality directly converts into local-model usability; these are targeted improvements:

- **Stable prompt prefix for KV-cache reuse.** Ollama and LM Studio both reuse the KV cache when
  the prompt *prefix* is unchanged. Order the RAG prompt as: static system/template → retrieved
  context → conversation history → question. Today's template interleaves these; reordering is
  cheap and can cut repeat-turn prefill dramatically.
- **Cross-encoder reranking.** A small local reranker (e.g. a MiniLM cross-encoder via
  sentence-transformers, already a dependency) over the hybrid top-2k results improves top-k
  precision → fewer chunks needed in context → shorter prompts → faster and more accurate.
- **Local query rewriting.** Use the local model (free) to expand terse queries with likely
  symbol names before retrieval — helps BM25 especially.
- **Measure, don't claim.** Extend the existing evals framework (`evals/run_eval.py`, 16-question
  set) with two new metrics per configuration: *prompt tokens per answer* and *time-to-first-token*.
  Every track above should land with an eval delta, keeping the project's "evaluated, not just
  vibes" positioning.

## Track 5 — CLI for hook/script integration

For consumers that can't speak MCP (shell scripts, git hooks, CI, RTK-style prompt pipelines):

- `codebase-rag query "<question>" [--repo X] [--k 5] [--budget 2000] [--format compact|json]`
  → prints ranked chunks to stdout.
- `codebase-rag ask "<question>"` → full local-model answer.
- Same compact format as Tracks 1–2. This makes the tool composable: an agent hook can prepend
  retrieved context to a prompt, or a developer can pipe results straight into any LLM CLI.

## Track 6 — Savings accounting

Savings that aren't measured don't convince anyone (and can't be tuned):

- Count tokens served per `/search` response and estimate the naive alternative (total size of
  the files the chunks came from — what an agent reading whole files would have paid).
- Aggregate into a `GET /stats` endpoint and a CLI `codebase-rag stats`: queries served, tokens
  served, estimated tokens avoided.
- Langfuse is already integrated for tracing; add these counters as span metadata so per-query
  savings are inspectable in the existing dashboard.

---

## Phasing

| Phase | Contents | Rationale |
|---|---|---|
| 1 | Track 1 (`/search` + compact format + local-path ingest) | Foundation; smallest slice that saves tokens |
| 2 | Track 2 (MCP stdio server) | Unlocks Copilot/Claude Code/Cursor with one integration |
| 3 | Track 3 (LM Studio / OpenAI-compat) + Track 4 prompt reordering | Broadens local backends; cheap latency win |
| 4 | Track 5 (CLI) + Track 6 (stats) | Composability and proof of value |
| 5 | Track 4 remainder (reranker, query rewriting) + watcher-based freshness | Quality tuning, guided by eval metrics |

## Risks and open questions

- **Agents must actually prefer the tool.** If tool descriptions don't outcompete the agent's
  builtin grep/read habits, savings don't materialize. Mitigate with steering language in
  descriptions and by documenting client-side instructions (e.g. a CLAUDE.md / copilot-instructions
  snippet telling the agent to search first).
- **Stale index is worse than no index.** Local-path incremental ingest ships in Phase 1 for this
  reason; freshness metadata is exposed in `/repos` so agents can detect drift.
- **Chunk granularity vs. budget.** Current chunking is tuned for chat answers; agent consumers
  may want smaller, symbol-level chunks. Revisit chunk size once eval metrics exist (Track 4).
- **Security surface.** The HTTP API and MCP server execute ingestion (git clone, file reads).
  Bind to localhost by default; document that exposing them beyond localhost needs auth.
- **Savings estimates are estimates.** The Track 6 "naive alternative" baseline is a heuristic;
  present it as such rather than as a measured A/B.
