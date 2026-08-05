# Retrieval Ablation

> **Deprecated: not reproducible.** These numbers were measured against a corpus of `power-grid-model` and
> `power-grid-model-ds`. Evaluation has since moved to a single repository
> (`power-grid-model` alone).
> Kept as a record of what was measured and when. Do not compare these figures against
> newer numbers or cite them as current.
>
> Current retrieval measurements are in [retrieval-stack-findings.md](../retrieval-stack-findings.md).

**Date:** 2026-08-03 15:56

Same test set (`evals/testset.json`), same LLM, same top_k — only the retriever feeding the RAG chain changes. Full per-question detail for each configuration is in `results_<retriever>.md`.

Test set composition (30 questions): 10 exact-term lookups (function/class/enum names), 6 multi-file/how-it-works reasoning questions, and 14 conceptual/paraphrased questions. The conceptual questions avoid quoting source identifiers, so a retriever's hit rate on them reflects semantic matching rather than keyword overlap.

**Embedder:** `sentence-transformers/all-mpnet-base-v2`, 768 dimensions, `max_seq_length` 384, float32,
no query or document prompt prefix (the model declares both as empty). This is one embedder's spread,
not the retrieval stack's: the retriever varies across these rows and the embedder does not. Separate
measurements of the embedder, the candidate depth, and the reranker are in
`retrieval-stack-findings.md`, and they move hit rate further than anything in the table below.

The hybrid arm applies the production cosine relevance cutoff (`VECTOR_SCORE_THRESHOLD=0.25`) to its vector component, matching the app's shipped configuration. The vector-only arm is unthresholded to isolate raw embedding ranking quality; BM25 scores are never thresholded (zero-overlap documents are excluded by construction).

Avg Latency figures are comparable only across runs with similar latency probes — see each configuration's `results_<retriever>.md` for its probe.

| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Answered | Failed | Avg Latency |
|-----------|----------|-----|----------------|-------------------|----------|--------|-------------|
| vector | 0.6207 | 0.5287 | 0.4104 | 0.2800 | 30 | 0 | 1.0s |
| bm25 | 0.6552 | 0.4362 | 0.5086 | 0.2333 | 30 | 0 | 0.9s |
| hybrid | 0.5862 | 0.5029 | 0.4713 | 0.2600 | 30 | 0 | 1.0s |
