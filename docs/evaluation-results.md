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

The app uses the hybrid retriever by default. To check whether that holds up, the eval runs the same 16-question test set through vector-only and BM25-only retrieval as well. Full results: [evals/ablation.md](../evals/ablation.md), [evals/results_vector.md](../evals/results_vector.md), [evals/results_bm25.md](../evals/results_bm25.md).

| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Avg Latency |
|-----------|----------|-----|----------------|------------------|-------------|
| Vector-only | 0.4000 | 0.2967 | 0.4074 | 0.1875 | 0.8s |
| BM25-only | 0.4000 | 0.2244 | 0.4473 | 0.1750 | 0.9s |
| Hybrid | 0.2667 | 0.2333 | 0.3406 | 0.1750 | 0.8s |

Read this table by Hit Rate and MRR. Both score retrieval directly against each question's expected source files, with no LLM in the loop: Hit Rate is the fraction of questions where an expected source appears anywhere in the retrieved set, and MRR (mean reciprocal rank) rewards putting it near the top. The keyword-recall and source-precision columns are kept for continuity, but keyword recall is measured on the generated answer, so the 350M model's phrasing sits between the retriever and the score — which is why these retrieval-only metrics were added.

These numbers were produced against the current Reciprocal Rank Fusion code, so they replace the pre-fusion ablation that used to sit here (the earlier version fused with a fixed 0.7/0.3 vector/BM25 blend and max-normalized BM25 per query, which could let a middling keyword match outrank a document BM25 alone would have put first; RRF fuses on rank order and removes that skew). They do not make the case for hybrid. On Hit Rate, vector-only and BM25-only tie at 0.40 while hybrid trails both at 0.27; on MRR, vector-only leads. As shipped, the hybrid arm applies the production vector relevance threshold to its vector component, and on this test set that fused, filtered path surfaces the expected file less often than either component alone. RRF fixed the old normalization skew, but it did not turn fusion into a win here.

Two caveats before drawing conclusions. The test set is 16 questions weighted toward exact-term lookups (function, class, and enum names), which is BM25's home turf and does not exercise the paraphrased or conceptual queries where vector search is meant to earn its place in the blend. And 16 questions is a thin basis for flipping a shipped default. So the default stays hybrid for now, but as a provisional call rather than a vindicated one: the current evidence argues against hybrid, and what it really points to is the need for a broader test set with conceptual questions before keeping or changing the default. The retrieval-only metrics added here exist so that call can rest on retrieval evidence rather than answer-phrasing proxies.

### RAGAS judge quality

This run was judged by a fixed `qwen3.5:9b` (reasoning disabled) rather than the self-judging 350M default, so the scores below are more trustworthy than earlier self-judged numbers — but read them with their coverage, which the harness now records and gates on, refusing to publish a metric that completed on fewer than 90% of questions.

| Retriever | Faithfulness | Answer Relevancy |
|-----------|--------------|------------------|
| Vector-only | 0.54 (16/16) | 0.80 (16/16) |
| BM25-only | 0.59 (15/16) | 0.78 (16/16) |
| Hybrid | not published (13/16) | 0.83 (16/16) |

Context recall is absent: the local judge cannot produce the structured output that metric's parser expects, so it was skipped rather than averaged from a few surviving samples. Hybrid's Faithfulness is shown as unpublished because only 13 of 16 judge calls parsed, below the coverage gate; the three failures were the judge mis-formatting otherwise-valid verdicts, not low scores (the 13 that parsed averaged 0.58, in line with the other two retrievers).

These scores are not comparable to the older baseline that reported Faithfulness 0.9048 and Context Recall 0.625 for vector. That baseline was self-judged by the 350M model and its coverage was never recorded, so the move to 0.54 here reflects the judge changing from a 350M self-judge to a fixed 9B judge at least as much as anything about the answers. A separate change also landed in between — the eval chain stopped rendering "No previous conversation." into every prompt once `use_conversation_memory=False` — but its isolated effect can't be separated from the judge swap, and the number it would be measured against was never trustworthy in the first place. No clean before/after delta can be claimed, so none is.

These figures come from a run whose judge was served by a native macOS Ollama with Metal acceleration; generation still used the shipped 350M model. Latencies here reflect that GPU path and are not comparable to Docker-deployment latencies elsewhere in this doc.

## Limitations

- **Retrieval ceiling.** The embedding model (`all-mpnet-base-v2`) struggles with very short code constructs like enum values, constants, and build configuration variables. Questions about specific enum members or CMake variables often score 0% recall.
- **Single embedding model.** All content is embedded with the same model regardless of language. A specialised code embedding models might improve retrieval for code-heavy queries.
- **No incremental deletion.** When a file is removed from a repository, its chunks remain in Qdrant until a `--force` re-index is performed.
- **Local LLM quality.** The default 350M model is fast but imprecise. A mid-size model gives noticeably better answers; local eval and judging are capped at 9B (e.g. `qwen3.5:9b`), run natively for GPU access. Earlier runs went as high as 30B (`qwen3-coder:30b`), but that is no longer a target — 9B is the ceiling going forward.
- **Docker GPU limitations.** On macOS, Docker containers cannot access the GPU. Running Ollama natively on the host gives significantly better performance (5.5x faster in evaluation).
- **RAGAS judge model.** By default `evals/run_eval.py` scores RAGAS metrics (Faithfulness, AnswerRelevancy, ContextRecall) using the same model that generated the answers — self-judging, plus a 350M model is a weak judge regardless. Results generated this way are marked "self-judged" in the report and should be treated as indicative only; the retrieval-only metrics (Hit Rate, MRR) and the keyword recall / source precision numbers don't use an LLM judge and are the trustworthy ones. Pass `--judge-model <name>` or set `RAGAS_JUDGE_MODEL` to score with a fixed, larger model instead. Even a capable local judge can be a poor fit for a specific metric's output parser: with `qwen3.5:9b`, Context Recall failed to parse on nearly every question and had to be skipped (`--skip-metric context_recall`), and Faithfulness occasionally mis-formatted a valid verdict. The coverage gate (`--min-coverage`, default 0.9) exists so a metric that only completed on a fraction of questions fails the run instead of publishing a misleading average.
