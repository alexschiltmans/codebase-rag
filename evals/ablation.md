# Retrieval Ablation

**Date:** 2026-07-21 14:41

Same test set (`evals/testset.json`), same LLM, same top_k — only the retriever feeding the RAG chain changes. Full per-question detail for each configuration is in `results_<retriever>.md`.

Test set composition: 16 exact-term (keyword/lookup) questions, 14 conceptual/paraphrased questions (30 total). The conceptual questions avoid quoting source identifiers, so a retriever's hit rate on them reflects semantic matching rather than keyword overlap.

The hybrid arm applies the production cosine relevance cutoff (`VECTOR_SCORE_THRESHOLD=0.25`) to its vector component, matching the app's shipped configuration. The vector-only arm is unthresholded to isolate raw embedding ranking quality; BM25 scores are never thresholded (zero-overlap documents are excluded by construction).

Avg Latency figures are comparable only across runs with similar latency probes — see each configuration's `results_<retriever>.md` for its probe.

| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Answered | Failed | Avg Latency |
|-----------|----------|-----|----------------|-------------------|----------|--------|-------------|
| vector | 0.6207 | 0.5270 | 0.4298 | 0.2800 | 30 | 0 | 0.9s |
| bm25 | 0.6552 | 0.4534 | 0.4769 | 0.2333 | 30 | 0 | 1.0s |
| hybrid | 0.5862 | 0.5115 | 0.4689 | 0.2600 | 30 | 0 | 0.9s |
