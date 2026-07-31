# Evaluation Results

This repo ships with a reproducible evaluation framework. The system was evaluated on 16 hand-picked questions against the PowerGridModel repository with two different model sizes.

See [evals/results_small_model.md](../evals/results_small_model.md) and [evals/results_large_model.md](../evals/results_large_model.md) for full breakdowns, or run your own:

```bash
python evals/run_eval.py
```

| Metric | Small model (350M) | Large model (30B) |
|--------|-------------------|-------------------|
| Avg keyword recall | 0.36 | 0.48 (+31%) |
| Avg source precision | 0.15 | 0.18 |
| Avg latency | 6.6s (Docker) | 5.6s (native GPU) |

The "large model" column is a past run with `qwen3-coder:30b`, kept here as a historical high-water mark. Local eval models are now capped at 9B, so that 30B column is not a current target — treat it as the ceiling an earlier, unconstrained run happened to reach.

**Key findings:**
- Cross-file reasoning is the system's strength (0.55 → 0.67 keyword recall)
- Retrieval is the bottleneck, not generation. Both models fail on the same 4 questions where the relevant chunk isn't in the top-5
- Enum/constant value lookups are consistently weak: the embedding model doesn't represent short code definitions well

## Retrieval Ablation

The app's default retriever is BM25-only (see "Default retriever decision" below). To measure the alternatives, the eval runs the full test set through vector-only, BM25-only, and hybrid (RRF) retrieval. The test set is now 30 questions: the original 16 (10 exact-term lookups of function/class/enum names plus 6 multi-file/how-it-works reasoning questions) and 14 conceptual/paraphrased questions added to stop the comparison being biased toward BM25's home turf. Full results: [evals/ablation.md](../evals/ablation.md), [evals/results_vector.md](../evals/results_vector.md), [evals/results_bm25.md](../evals/results_bm25.md), [evals/results_hybrid.md](../evals/results_hybrid.md).

Figures below are from the 2026-07-31 re-run (`evals/ablation.md`), which supersedes the 2026-07-28 run and the 2026-07-21 baseline.

| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Avg Latency |
|-----------|----------|-----|----------------|------------------|-------------|
| Vector-only | 0.6207 | 0.5287 | 0.4543 | 0.2800 | 1.0s |
| BM25-only | 0.6552 | 0.4362 | 0.5217 | 0.2333 | 0.9s |
| Hybrid | 0.5862 | 0.5029 | 0.4413 | 0.2600 | 0.9s |

Hit Rate and Source Precision reproduced the 2026-07-21 baseline to four decimals on all three arms. MRR did not: it moved on all three (BM25 0.4534 to 0.4362, hybrid 0.5115 to 0.5029, vector 0.5270 to 0.5287). Since Hit Rate and Source Precision are set-based and MRR is rank-based, the retrieved sets were identical and their ordering was not. Neither metric involves a judge, so this is not judge variance. The cause has not been established; tie-break ordering changing when the corpus was rebuilt is the obvious candidate, but it is untested. Latencies are not comparable to the baseline's either: the 2026-07-21 run was served by a native Metal Ollama, this one was not.

The 2026-07-31 re-run reproduced 2026-07-28 exactly on retrieval: Hit Rate, MRR and Source Precision match to four decimals on all three arms, and the per-question retrieved sets are identical across all 90 question/arm pairs. Only Keyword Recall moved (vector 0.4432 to 0.4543, BM25 0.4802 to 0.5217, hybrid 0.4468 to 0.4413), which is generation-side sampling variance, since it scores the answer text rather than the retrieval. Latency is again not comparable: this run was served by a native Metal Ollama on an otherwise idle machine, which is where the drop from roughly 6-10s to roughly 1s comes from, not from any change to retrieval or generation. The category breakdown below is derived from the same retrieval results and is unchanged.

Read this table by Hit Rate and MRR, same as before: both score retrieval directly against each question's expected source files, with no LLM in the loop. Hit Rate is what actually reaches the LLM's context at `top_k=5` (every retrieved document is passed in, not just the top one), so it is the metric that should drive the default-retriever decision; MRR matters more for a caller that only uses the top result.

Broken out by question category, the picture is more specific than "hybrid loses":

| Retriever | Conceptual Hit Rate | Conceptual MRR | Exact-term Hit Rate | Exact-term MRR |
|-----------|---------------------|-----------------|----------------------|------------------|
| Vector-only | 0.8571 | 0.7738 | 0.4000 | 0.3000 |
| BM25-only | 0.9286 | 0.6631 | 0.4000 | 0.2244 |
| Hybrid | 0.9286 | 0.7917 | 0.2667 | 0.2333 |

Conceptual is 14 questions; exact-term is 15 of the 16 exact-term questions, excluding the one question flagged `expected_failure` in `testset.json` (a known confusable case, excluded from Hit Rate/MRR the same way the headline table's 0.6207/0.6552/0.5862 figures exclude it — both tables are on the 29-question basis). Hybrid actually leads on conceptual questions (ties BM25's hit rate, beats both on MRR) — RRF fusion does what it's meant to do there. Its overall deficit comes entirely from exact-term questions, where it trails both single components, which tie at 0.40.

### Diagnosis: why hybrid underperforms on exact-term lookups

5 of the 29 questions (excluding `expected_failure`, same basis as the tables above) have hybrid missing while a single component hits (4 exact-term, 1 conceptual). For every one of those 5, the expected document was present in the winning component's raw top-10 — at rank 3-9, never dropped by `VECTOR_SCORE_THRESHOLD` — but the *other* component never surfaced it at all. RRF's rank-only score (`weight / (rrf_k + rank)`) gives that mid-single-digit rank a small contribution; because it only comes from one list, it loses to documents that both lists rank moderately (or that one list ranks in the top 1-2), so it falls out of the fused top 5. That's the "RRF's rank-only blend discards a strong single-retriever signal" candidate from `design.md`, confirmed as the operative cause: 5/5, not the threshold and not `top_k`.

As a control, vector-only was also run with `VECTOR_SCORE_THRESHOLD=0.25` applied. That threshold pass was a separate run, not re-scored onto the 29-question basis the tables above use, so it does not yield a clean same-basis delta; on its own terms it lowered vector's hit rate by roughly one question (about 2 percentage points), with a similarly small MRR change. The threshold is a real but minor cost; it is not what drags hybrid below its components.

### Default retriever decision

**The default retriever is now BM25-only**, changed from hybrid in `AppRuntime`. BM25-only has the best overall Hit Rate (0.6552) and ties or leads hybrid in both categories on Hit Rate, which is what determines whether the right file reaches the model's context at `top_k=5`. Vector-only is a reasonable second choice (best MRR, close behind on Hit Rate); hybrid is the weakest choice on the metric that matters most for this app's context-window usage, despite genuinely helping on conceptual queries, because its exact-term regression is larger than its conceptual gain.

The RRF fusion weakness diagnosed above looks fixable in principle (e.g. don't let a single strong ranker's mid-rank hit be outscored by two weak-but-present ranks), but `design.md`'s decision was to fix fusion only if the cause is both found and safe to change quickly — guessing at fusion weights is what caused the earlier hybrid implementation's problems. A fusion algorithm change is a large enough behavior change, with its own risk of regressing the conceptual-query win it currently has, that it belongs in its own change rather than being bolted onto this one. No `retrieval-relevance` delta was written as part of this change; `HybridRetriever` and its RRF fusion code are unchanged and remain available (used directly by the ingestion pipeline's duplicate-detection search), just no longer the app's default RAG retriever.

This replaces the "provisional, pending a broader test set" caveat that used to sit here: the broader (30-question, category-balanced) set now exists, and the decision above is what it supports.

### RAGAS judge quality

The table below is from the 2026-07-21 run, judged by a fixed `qwen3.5:9b` (reasoning disabled) rather than the self-judging 350M default, so its scores are more trustworthy than earlier self-judged numbers — but read them with their coverage, which the harness now records and gates on, refusing to publish a metric that completed on fewer than 90% of questions.

The 2026-07-28 and 2026-07-31 re-runs that produced the retrieval numbers above did **not** set `--judge-model`, so both were self-judged by the 350M model. Their RAGAS scores in `evals/results_*.json` (Answer Relevancy around 0.09 to 0.17, Faithfulness 0.90 to 0.98) are therefore not comparable to this table and are not published here; the gap between them measures the judge, not the answers. Re-run with `RAGAS_JUDGE_MODEL=qwen3.5:9b` to refresh these figures.

Judge coverage did improve between those two runs, from 264 of 270 metric/question calls completing to 269 of 270. The judge client's `timeout` had been passed as a kwarg `ChatOllama` silently discards, so no configured timeout ever reached the HTTP client; moving it to `client_kwargs` fixed that. Four of the nine metric/arm cells that had been short of full coverage now complete on all 30 questions. This is a coverage change, not a quality change, and it does not affect the qwen3.5:9b table above, which was measured before either run.

| Retriever | Faithfulness | Answer Relevancy | Context Recall |
|-----------|--------------|------------------|----------------|
| Vector-only | 0.575 (30/30) | 0.833 (30/30) | 0.569 (30/30) |
| BM25-only | 0.625 (30/30) | 0.837 (30/30) | 0.579 (30/30) |
| Hybrid | 0.530 (30/30) | 0.780 (30/30) | 0.519 (30/30) |

All three RAGAS metrics report here, each at full 30/30 coverage — no `--skip-metric` was needed. Earlier runs had to drop Context Recall (and sometimes Faithfulness) because the 9B judge echoed the metric's JSON schema instead of an instance, or truncated a verbose verdict; the judge now decodes under a per-metric JSON schema constraint with a raised context/output budget, so those outputs parse. The scores are what a fixed 9B judge assigns; treat them as indicative, not ground truth, and lean on the non-judge metrics (Hit Rate, MRR, keyword recall, source precision) for retrieval comparisons.

Answer Relevancy is not directly comparable to older baselines from before the fixed-judge switch either: those were self-judged by the 350M model with coverage never recorded, so any delta reflects the judge changing from a 350M self-judge to a fixed 9B judge at least as much as anything about the answers. A separate change also landed in between — the eval chain stopped rendering "No previous conversation." into every prompt once `use_conversation_memory=False` — but its isolated effect can't be separated from the judge swap, and the numbers it would be measured against were never trustworthy in the first place. No clean before/after delta can be claimed, so none is.

This RAGAS table comes from a run whose judge was served by a native macOS Ollama with Metal acceleration; generation still used the shipped 350M model. The 2026-07-21 latencies that went with it reflect that GPU path; the ablation table above now carries the 2026-07-28 re-run's latencies instead, which do not.

## Limitations

- **Retrieval ceiling.** The embedding model (`all-mpnet-base-v2`) struggles with very short code constructs like enum values, constants, and build configuration variables. Questions about specific enum members or CMake variables often score 0% recall.
- **Single embedding model.** All content is embedded with the same model regardless of language. A specialised code embedding models might improve retrieval for code-heavy queries.
- **No incremental deletion.** When a file is removed from a repository, its chunks remain in Qdrant until a `--force` re-index is performed.
- **Local LLM quality.** The default 350M model is fast but imprecise. A mid-size model gives noticeably better answers; local eval and judging are capped at 9B (e.g. `qwen3.5:9b`), run natively for GPU access. Earlier runs went as high as 30B (`qwen3-coder:30b`), but that is no longer a target — 9B is the ceiling going forward.
- **Docker GPU limitations.** On macOS, Docker containers cannot access the GPU. Running Ollama natively on the host gives significantly better performance (5.5x faster in evaluation). Both the generation and judge clients read `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`), and the harness logs the resolved endpoint at run start. This project's optional Ollama container publishes on 11435 and can no longer shadow a native server, but any other container that binds 11434 still can: on macOS a bare `localhost` resolves to `::1` before `127.0.0.1`, reaches the CPU-only server, and turns one judge job into minutes. Keep `OLLAMA_BASE_URL` on the literal `http://127.0.0.1:11434` for the native GPU path and confirm it from the logged base URL. The runs recorded above predate the port move and were taken on a machine where both servers claimed 11434.
- **RAGAS judge model.** By default `evals/run_eval.py` scores RAGAS metrics (Faithfulness, AnswerRelevancy, ContextRecall) using the same model that generated the answers — self-judging, plus a 350M model is a weak judge regardless. Results generated this way are marked "self-judged" in the report and should be treated as indicative only; the retrieval-only metrics (Hit Rate, MRR) and the keyword recall / source precision numbers don't use an LLM judge and are the trustworthy ones. Pass `--judge-model <name>` or set `RAGAS_JUDGE_MODEL` to score with a fixed, larger model instead. The judge decodes under a JSON schema constraint (Ollama's structured output, `format=<schema>`): without it a 9B judge — the project's cap — echoes each metric's JSON schema instead of an instance (this sank Context Recall on nearly every question and Faithfulness on a fraction), which no tolerant parser can recover. Constraining decoding to the active metric's schema, with a raised judge context/output budget for verbose verdicts, makes all three metrics producible without a larger judge. The coverage gate (`--min-coverage`, default 0.9) still fails the run if a metric only completed on a fraction of questions rather than publishing a misleading average.
