# Project review: Codebase RAG

Reviewed 2026-07-19, at commit `cf46cf6`. The whole codebase was read (~4,000 lines of source, plus tests, evals, Docker setup, and docs), the unit suite was run (203 tests, all passing in under 4 seconds), and git history was checked for hygiene issues. A follow-up pass (section 4) hunted for silent UI state-machine bugs and verified each suspect empirically with `AppTest`. Findings are grouped by reviewer perspective; a prioritized list is at the end.

## Summary

This is a strong portfolio project, and what makes it strong is unusual: not the RAG plumbing itself, but the evidence around it. It ships an ablation study that shows its own headline feature (hybrid retrieval) losing to plain BM25 on the current test set, says so in the docs, explains why, and describes the fix. That kind of honesty is rare and it's worth more to a hiring reviewer than a page of inflated metrics.

The engineering is also above the bar for a hobby project: strict ruff and mypy settings, a fast isolated unit suite, real end-to-end tests through Streamlit's `AppTest`, ADRs that record actual tradeoffs, and carefully reasoned concurrency code. The weak spots are a handful of real bugs (one config setting is silently ignored), some claims in the docs that the code doesn't back up, and retrieval-quality gaps that matter more than any of the code-level issues.

---

## 1. Senior AI engineer perspective

### What's good

- **The evaluation exists at all, and it's honest.** `evals/run_eval.py` runs a 30-question test set through vector-only, BM25-only, and hybrid retrieval, computes judge-free metrics (keyword recall, source precision) alongside RAGAS scores, and writes per-question reports. When RAGAS runs without a separate judge model, the report is stamped "self-judged" with an explanation of why those numbers are weak. Most production teams don't do this much.
- **The RRF migration is well reasoned.** The module docstring in `hybrid_search.py` explains exactly why per-query max-normalized BM25 scores were broken (the top keyword hit always got 1.0 regardless of match quality) and why rank fusion fixes it. The reasoning is correct.
- **Failure analysis in `docs/evaluation-results.md` names real failure modes**: retrieval (not generation) is the bottleneck, short code constructs like enum values embed poorly, and both model sizes fail on the same four questions. Those are the right observations to make.

### Findings

**AI-1. The relevance thresholds are dead code after the RRF change.** (`hybrid_search.py:108-117`, `rag_chain.py:280-283`)
RRF scores are rescaled so a document ranked #1 by both retrievers gets 1.0. Work through the arithmetic: a document found *only* by BM25 at rank 1 scores 0.3 normalized; found only by vector at rank 1, 0.7; even the last of 10 vector-only results scores about 0.6. Nothing that either retriever returns within `k*2` results ever falls below `min_score_threshold=0.1`, and almost nothing falls below the chain's `min_relevance_score=0.15`. These parameters filtered by *similarity* under the old scheme; after RRF they filter by *rank agreement*, which top-k already did. The practical consequence: the "I couldn't find any relevant information" path in `RAGChain` is nearly unreachable. An off-topic query ("what's a good lasagna recipe?") still retrieves five chunks and the 350M model will happily answer from them. If out-of-scope refusal matters, you need a threshold on the raw cosine score from Qdrant *before* fusion, not on the fused rank score.

**AI-2. The ablation was never rerun after the fusion change it motivated.** (`evals/ablation.md`, dated 2026-07-18)
The docs say this openly, which is good. But it leaves the project in an awkward state: the stated reason to believe hybrid search now works is a code change whose effect hasn't been measured, while the only published measurement shows hybrid coming last (0.3749 keyword recall vs 0.4302 for BM25-only). Rerunning the eval is one command. Until then, the README bullet "Hybrid retrieval" is selling the one feature the project's own data doesn't support.

**AI-3. `EMBEDDING_MODEL` config is silently ignored.** (`runtime.py:175`, `pipeline.py:209`, `qdrant_store.py:36`)
`Config` reads `EMBEDDING_MODEL` from the environment, but both `AppRuntime` and `IngestPipeline` construct `QdrantStore` without passing it, so the hardcoded default `all-mpnet-base-v2` in the `QdrantStore` signature always wins. Only the eval script actually respects the setting. Anyone who sets `EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code` to test the docs' own suggestion ("a specialised code embedding model might improve retrieval") will get the old model and no error. Worse, if they set it *before first ingest*, queries and documents would at least agree; set after, and it still agrees, because it's ignored everywhere. Fix: pass `config.embedding_model` through in both places, and drop the duplicated default from the `QdrantStore` signature so this class of bug can't recur.

**AI-4. "Language-aware chunking" covers Python and Markdown only.** (`chunking.py:73-88`)
Every other language, including the C++/CMake of the very repo used for evaluation, goes through the generic character splitter. So the eval questions about `CMakeLists.txt` and `.hpp` component hierarchies, several of which score poorly, are testing exactly the files that get the naive treatment. Related: `.ipynb` files are routed to the *Python* splitter, but a notebook on disk is JSON with base64 outputs embedded. It won't crash (the splitter degrades to character splitting), but it will index noise. Either strip notebook cells properly (e.g. via `nbformat`) or exclude `.ipynb`. The README's chunking claim should also be scoped honestly: "Python-specific and Markdown-aware" is what the README says in one place and "language-aware" is the impression it gives elsewhere.

**AI-5. Multi-turn retrieval uses the raw follow-up text.** (`rag_chain.py:202-204`)
Conversation history goes into the generation prompt but not into retrieval. A follow-up like "how is it tested?" retrieves on those three words alone, which will pull essentially random test files. Standard fixes are query rewriting (one extra LLM call to expand the follow-up into a standalone question) or at minimum concatenating the previous user turn into the retrieval query. For a project that advertises conversation memory, this is the gap a knowledgeable interviewer will ask about.

**AI-6. No context-window management on generation.** (`ollama_client.py:39-46`)
`ChatOllama` is constructed without `num_ctx`, so the Ollama server default applies (2048 tokens on many installs). The prompt is five 1000-character chunks plus up to ten turns of conversation history plus the template; that comfortably exceeds 2048 tokens, and Ollama truncates silently. The symptom would be answers that ignore the question or the earlier context with no error anywhere. Set `num_ctx` explicitly to something the 350M model supports, and consider budgeting context docs by token count rather than a fixed `top_k`.

**AI-7. The eval metrics conflate retrieval and generation.** Keyword recall checks whether expected keywords appear in the final *answer*, so a retrieval improvement can be masked by the 350M model failing to use the context (and the docs' own finding says generation quality varies by model). Since the ablation only varies the retriever, the metric that should drive it is retrieval-only: did the expected source appear in the top-k (hit rate / MRR against `sources_expected`, which the test set already has). Source precision is close but is computed on the answer path and skips questions where nothing was retrieved. Sixteen questions on one repo is also thin for drawing conclusions, which the docs to their credit half-acknowledge.

---

## 2. Senior software engineer perspective

### What's good

- **Module boundaries are real.** The `app/` split (runtime for process-wide resources, state for per-session state, ui_* for rendering) is clean, documented at the top of each file, and actually enforced (the folder picker lives in `services/` specifically to keep threading out of `app/`).
- **The concurrency code is careful.** `IngestionManager.start()` is a compare-and-set under a lock with a comment explaining why a render-time `disabled=` flag can't prevent the race. The folder picker uses per-request tokens so one session can't read another's dialog result. `_normalize_dialog_path` handles the `/` and `C:\` edge cases with a comment explaining both. This is the strongest code in the repo.
- **Tooling is genuinely strict**: ruff with a wide select list and *documented* ignores, mypy with `disallow_untyped_defs`, pre-commit, a coverage gate at 80%. Committed secrets were checked for: `.env` and `.sonar-token` exist locally but are not tracked, and `.env.example` is what ships. Clean.
- **The test suite is fast and layered.** 203 unit tests in 3.85s, integration and e2e kept separate, and the e2e tests drive the real `main.py` through `streamlit.testing.v1.AppTest` so the rerun/session-state machine is actually exercised. Testing Streamlit at that layer is uncommon and it shows real familiarity with the failure modes.

### Findings

**SE-1. The retriever interface is duck-typed, and the dispatch has a bug-shaped hole.** (`rag_chain.py:268-284`)
`_retrieve_documents` calls `_do_retrieve(query, top_k)` and catches `TypeError` to fall back to a single-argument call. Any `TypeError` raised *inside* a retriever (a `None` compared to an int, a bad unpack) is swallowed and the retrieval silently retried without `top_k`. `_do_retrieve` then dispatches on `hasattr(retriever, "search")`. The project already defines `VectorStoreProtocol` for stores; retrievers deserve the same: a `RetrieverProtocol` with one `search(query, k)` method would delete both the `hasattr` check and the `except TypeError`. While there: `aget_relevant_documents` is named like a coroutine and isn't one; either make it async or drop it.

**SE-2. `Config.get_instance()` is called from inside at least five classes** (`OllamaClient`, `GitLoader`, `EmbeddingManager`, `SqliteChatStorage`, `IngestPipeline`), which is how finding AI-3 happened: defaults live in three places (`Config`, the `QdrantStore` signature, the `EmbeddingManager` fallback) and nothing forces them to agree. The singleton also makes test isolation depend on resetting `_instance` by hand. The codebase is small enough that passing `Config` (or just the needed values) down explicitly is a mechanical change, and it would take the whole category of "env var read but not plumbed through" bugs off the table.

**SE-3. Conversation trimming leaves an orphaned assistant message.** (`rag_chain.py:421-439`)
The trim loop advances `i` until it has consumed the excess *user* messages, then slices. With history `[u1, a1, u2, a2, ...]` and one excess turn, it stops at `i=1` and the slice keeps `a1`: an assistant reply whose question was just removed. Every subsequent trim compounds this. The prompt then opens with a context-free "Assistant: ..." turn. Trim in user+assistant pairs instead.

**SE-4. Per-run ingest log files silently don't happen in the app.** (`pipeline.py:101-105`)
`setup_logging` uses `logging.basicConfig` with a `FileHandler` per run. `basicConfig` is a no-op once the root logger has handlers, and `app/main.py` configures logging at import. So the `logs/ingest-*.log` files the code promises are only written when the pipeline runs from the CLI before anything else configured logging. Attach the file handler explicitly to the `codebase_rag` logger instead of going through `basicConfig`.

**SE-5. Unvalidated clone URLs reach GitPython on the CLI path.** (`scripts/ingest.py`, `git_loader.py:83`)
`git.Repo.clone_from` with an attacker-shaped URL is a known command-execution vector (`ext::` transport). The UI gates on an `https://github.com/` prefix; the CLI accepts anything, and the same string is also used to build filesystem paths. For a single-user local tool the risk is low, but a portfolio project gets read as "how does this person handle input by default", so validating scheme/host in one shared place (and using it from both entry points) is cheap signal.

**SE-6. Re-ingest is delete-then-insert with no atomicity.** (`pipeline.py:386-404`)
`delete_by_repo` runs before `add_documents`, so queries issued mid-ingest see a partially empty index. Fine for this app's scale; worth a code comment at minimum, or an aliased-collection swap if you ever want to show off the production pattern. Related small dishonesty: the README says "content hashing and deterministic chunk IDs prevent duplicates," and `getting-started.md` goes further with "unchanged chunks are skipped." The `content_hash` is computed and stored but *never read*; nothing skips unchanged chunks, and the UI path even sets `use_cache=False`, so every re-ingest re-embeds everything. The system is idempotent (deterministic IDs make re-runs converge) but not incremental. Fix the docs, or implement the skip using the hash that's already there.

**SE-7. Smaller items.**
- `EmbeddingManager.__new__` returning cached instances keyed by model name works, but `__init__`-less initialization via `_initialize` is the kind of cleverness a plain module-level factory function would avoid.
- `test_remaining_coverage.py` as a filename says "written to hit a number." The tests may be fine; the name undermines them.
- The README coverage badge is a hardcoded shield (83%) rather than generated from CI, and CI itself runs only the unit tier even though the e2e tests are mocked enough to run there too.
- `main.py` calls `st.set_page_config` at import time and module-level `logging.basicConfig`; both are Streamlit-conventional but make the module import-order sensitive, which the AppTest suite quietly depends on.

---

## 3. Senior developer perspective (as a prospective user)

### What's good

- `make services-start` really is the whole setup: Qdrant, Ollama, Langfuse, and the app, with healthchecks, named volumes, and automatic model pull. The Makefile has a colored `help` target and sensible task names. `getting-started.md` gives both the Docker path and a manual path with actual commands.
- The UI covers the workflows a user actually needs: ingest by URL or by native folder picker (with a typed-path fallback for Docker), delete a repo with confirmation, persistent chats, streamed answers with source citations, and error states that are dismissible rather than stack traces.
- The eval docs tell you up front that the default model is weak and that native Ollama is 5.5x faster than Docker on a Mac. That saved-me-an-afternoon honesty is exactly what a README's audience wants.

### Findings

**DEV-1. First-run failure modes are the roughest part.** The entrypoint waits for Qdrant but not for Ollama, and the compose file's `app` service only `depends_on` Qdrant. If the model pull fails (Ollama slow to start, no network), the script prints a warning that scrolls away, and the user's first question fails with a model-not-found error they have to interpret themselves. The app also downloads the ~420MB embedding model on first start with no UI indication beyond a hung spinner. Wait for Ollama in the entrypoint, and surface "model missing, run `ollama pull ...`" in the UI (the `check_model_availability` method that produces exactly this message already exists; it's only ever logged).

**DEV-2. The Hugging Face cache volume doesn't do anything.** (`compose-dev.yml:88`, `docker/Dockerfile`)
`codebase_rag_app_cache` is mounted at `/app/cache`, but `HF_HOME` is never set, so sentence-transformers caches to `/root/.cache` inside the container layer. Every image rebuild re-downloads the embedding model. One `ENV HF_HOME=/app/cache` line in the Dockerfile makes the volume real.

**DEV-3. Answer quality with the default model will disappoint people who don't read the eval docs.** Keyword recall of 0.36 means the default experience is a demo, not a tool. That's a legitimate choice for a laptop-friendly default, but the README's front page should say it as plainly as `evaluation-results.md` does, and the sidebar could show a one-line "for better answers, set `LLM_MODEL_NAME` to a larger model" hint. Right now the path from "answers are bad" to "here's the env var" runs through two docs files.

**DEV-4. Long ingests can't be cancelled and barely report progress.** The CLI gets a progress bar; the UI gets an elapsed-seconds counter. Ingesting a large repo means minutes of CPU-bound embedding with no file count, no ETA, and no cancel. The pipeline already knows the batch count; wiring `n/total` into `IngestJob` would be a small change with outsized UX effect. A cancel flag checked between batches would cover the rest.

**DEV-5. Stale references will trip up code readers.** The README's project structure lists `app/components.py`, which was refactored away (it's now `runtime.py`/`state.py`/`ui_*.py`; the structure diagram also omits `services/`). `docker/entrypoint.sh`'s comment likewise points at `components.py`. Anyone using the README as a map, which is what it's for, starts on the wrong foot.

**DEV-6. There's no way to consume this besides the UI.** No HTTP API and no importable high-level facade means the retrieval quality work can't be reused from an editor plugin, a script, or another service. Even a minimal `POST /query` (or documenting `RAGChain` + `HybridRetriever` construction as the supported embedding path) would broaden who can use the project from "people who want this exact Streamlit app" to "people who want local codebase RAG."

**DEV-7. Small friction notes.** Ports 3000, 6333, 8501, and 11434 are all common collision targets and nothing detects a conflict. Langfuse ships with hardcoded `NEXTAUTH_SECRET`/`SALT` (fine locally, worth a comment saying so). The GitHub URL validation rejects `git@github.com:` SSH remotes, which is the first thing many developers paste.

---

## 4. Streamlit UI behavior audit (follow-up, verified with AppTest)

A targeted second pass over the UI state machine, done after the main review. Method: each suspected bug was turned into an `AppTest` test that asserts the *buggy* behavior, run against the real `app/main.py` with the same mocked-services harness the e2e suite uses. Five of the six suspects reproduced; the sixth (UI-4) can't be driven by AppTest and rests on code reading. These are exactly the silent failures the existing e2e suite doesn't cover: it tests the happy paths and single-state error handling, not transitions *between* features.

**UI-1. Both Ingest buttons ignore `IngestionManager.start()` refusing.** (`ui_sidebar.py:157`, `ui_sidebar.py:250`) — *verified*
`start()` is a compare-and-set that returns `False` when the single ingestion slot is taken — the manager's docstring even explains that the render-time `disabled=` snapshot can't prevent this race. But both call sites discard the return value. GitHub tab: the click does nothing, no message anywhere. Local-folder tab is worse: the click handler clears `selected_folder` and the preview cache *before* calling `start()`, so a refused request also destroys the user's selection. The careful concurrency design in the manager is undone by the UI ignoring its answer.

**UI-2. The typed-path flow is a one-shot.** (`ui_sidebar.py:190-194`) — *verified*
The typed path is promoted to `selected_folder` only when it *differs* from the previous render's value (`typed_path != prev_typed`). Two consequences. (a) After an ingest clears the selection, the text input still shows the path, but re-submitting the identical path is impossible — the guard never fires again, so no preview and no Ingest button appear; the user must edit the path to something else and back. (b) Clearing the input entirely leaves the stale selection active: empty text box, yet the old folder's preview and a live Ingest button remain. Both directions of the sync are broken; the widget and the selection state disagree whenever they should reconcile.

**UI-3. Query failure state is not chat-scoped.** (`state.py:117-125`, `ui_chat.py:128-132`) — *verified*
`switch_chat` and `start_new_chat` reset messages but not the query lifecycle (`query_state`, `pending_query`, `query_error`). A failed query's Retry/Dismiss card therefore renders inside whatever chat the user navigates to next. Clicking Retry there resubmits the *old* chat's question and appends the streamed answer to the *new* chat — the verification test ends with chat B containing an orphan assistant answer and zero user messages. Fix: reset the lifecycle (or scope it per chat) in `switch_chat`/`start_new_chat`.

**UI-4. Confirmation dialogs reopen after being dismissed.** (`ui_sidebar.py:123-125`, `ui_sidebar.py:309-311`) — *code-level, not AppTest-verifiable*
Both delete dialogs follow the pattern "set a session flag on button click, then call the `@st.dialog` function on every rerun while the flag is set." Streamlit dialogs dismissed via X, ESC, or clicking outside trigger a rerun without running the dialog's buttons — the flag survives, the render path calls the dialog function again, and the dialog reopens. The only real exits are the Cancel/Delete buttons. The canonical pattern is to open the dialog directly from the button-click branch (an event) rather than unconditionally from render state.

**UI-5. The folder preview count goes stale.** (`ui_sidebar.py:254-271`) — *verified*
`_preview_local_folder` caches non-zero counts per path in session state. Add files to the folder and the caption keeps reporting the old count until the path string changes. The docstring already identifies the zero-count staleness problem and solves it by not caching zero — the same reasoning applies to non-zero counts (a `Refresh` affordance, a short TTL, or just re-walking on expander open; the walk is cheap for typical repos).

The pattern across UI-1/2/3 is worth naming: individual widgets are handled carefully, but *cross-feature transitions* (ingest slot contention, form resubmission, chat navigation during an error) fall through. The verification tests invert directly into regression tests for the fixes.

---

## Prioritized recommendations

| # | Finding | Effort | Why it's ranked here |
|---|---------|--------|----------------------|
| 1 | Rerun the ablation post-RRF (AI-2) | Low | The project's central claim currently has stale counter-evidence attached. One command plus a docs update. |
| 2 | Fix `EMBEDDING_MODEL` plumbing (AI-3) | Low | Silent config-ignoring bug; also enables the docs' own suggested experiment. |
| 3 | Rethink relevance thresholds post-RRF (AI-1) | Medium | Off-topic queries always get confident answers; the refusal path is unreachable. |
| 4 | Align docs with ingestion reality, or implement hash-skip (SE-6) | Low/Medium | "Unchanged chunks are skipped" is currently false. |
| 5 | Set `num_ctx` and budget the prompt (AI-6) | Low | Silent truncation undermines both answers and the eval numbers. |
| 6 | First-run UX: wait for Ollama, surface model errors in UI (DEV-1) | Low | This is the moment a visitor decides whether the project works. |
| 7 | Retrieval-only metrics in the eval (AI-7) | Medium | Makes the ablation measure what it varies. |
| 8 | Query rewriting for follow-ups (AI-5) | Medium | Biggest quality gap in the advertised multi-turn feature. |
| 9 | `RetrieverProtocol`, drop `except TypeError` dispatch (SE-1) | Low | Removes a bug-swallowing path; cheap rigor signal. |
| 10 | Fix conversation trim pairing (SE-3) | Low | Small correctness bug with a two-line fix. |
| 11 | Reset query lifecycle on chat switch/new chat (UI-3) | Low | Verified cross-chat data leak: answers land in the wrong chat. |
| 12 | Honor `start()` refusals; fix typed-path sync (UI-1, UI-2) | Low | Verified silent drops and a one-shot form; undoes the concurrency care in the manager. |
| 13 | Stale doc references, `HF_HOME`, ingest progress (DEV-2/4/5) | Low | Polish that code readers notice quickly. |
| 14 | Event-driven dialogs, preview refresh (UI-4, UI-5) | Low | Small annoyances; fix alongside the other sidebar work. |

## Verdict

As a portfolio piece this already does its job: it demonstrates evaluation discipline, honest reporting, careful concurrency work, and real testing depth, and those are the things that distinguish an AI engineer from someone who wired up LangChain once. The highest-leverage next step isn't a new feature. It's closing the loop the project opened itself: rerun the ablation against the RRF fusion, fix the two config/threshold bugs that quietly undermine retrieval quality, and make the docs match what the code does. A project whose stated numbers are current and whose claims all check out is rarer than one with more features.
