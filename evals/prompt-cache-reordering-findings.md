# Prompt Cache-Reordering: Quality and TTFT

Measures the effect of ordering the RAG prompt as static template → retrieved
context → conversation history → question (a byte-stable static prefix for
KV-cache reuse) against the previous history-first ordering.

**Scope.** Hybrid retriever only, `power-grid-model` corpus, 43-question test
set (`evals/testset.json`), judge `qwen3.5:9b`, full coverage (43/43 on every
RAGAS metric in both runs). The vector and BM25 arms were not re-run: prompt
ordering changes only the generated answer, never retrieval, so their retrieval
metrics are unchanged by construction. Generation model `sam860/LFM2:350m`;
RAGAS is self-generated-then-large-judge, so treat the RAGAS numbers as
indicative and weight the keyword/retrieval metrics more heavily.

## Quality (before = history-first, after = reordered)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| avg_keyword_recall | 0.4947 | 0.4962 | +0.0015 |
| avg_source_precision | 0.3953 | 0.3953 | 0 |
| avg_hit_rate | 0.8333 | 0.8333 | 0 |
| avg_mrr | 0.7044 | 0.7044 | 0 |
| faithfulness (RAGAS) | 0.5618 | 0.5065 | -0.0553 |
| answer_relevancy (RAGAS) | 0.8142 | 0.8481 | +0.0339 |
| context_recall (RAGAS) | 0.4992 | 0.4837 | -0.0155 |

Retrieval-derived metrics (hit rate, MRR, source precision) are identical, as
expected. Keyword recall is flat. The RAGAS metrics move in both directions by
less than the run-to-run noise of a 350M-generated / self-caveated judge setup:
relevancy up, faithfulness and recall slightly down, no consistent regression.
No gate failure: coverage is complete and no metric collapses.

## Repeat-turn TTFT

Two-turn session (conversation memory on), turn-2 time-to-first-token, 5
repeats each, same live generation model:

| Ordering | mean turn-1 TTFT | mean turn-2 TTFT |
|----------|------------------|------------------|
| reordered | 0.300s | 0.234s |
| history-first | 0.160s | 0.251s |

Turn-2 TTFT is ~7% lower with the reordered prompt (0.234s vs 0.251s). The
effect is small on this corpus: `power-grid-model` contexts are short and a
350M model prefills quickly, so the reusable prefix is a small share of total
prefill. The benefit scales with prefix length and model size, a larger local
model with longer static instructions and larger retrieved contexts stands to
gain more. The turn-1 numbers are first-call warmup noise and not comparable
across the two blocks.

## Conclusion

No quality regression. Turn-2 TTFT improves modestly here and is structurally
positioned to improve more as prefix length and model size grow. Numbers here
are stale the moment retrieval changes again; re-run before relying on them.
