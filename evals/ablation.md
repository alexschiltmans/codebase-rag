# Retrieval Ablation

**Date:** 2026-08-07 23:17

Same test set (`evals/testset.json`), same LLM, same top_k — only the retriever feeding the RAG chain changes. Full per-question detail for each configuration is in `results_<retriever>.md`.

**Repositories retrieved from:** `power-grid-model`. Every arm was restricted to these, so these figures are a measurement of this corpus rather than of whatever else shared the index when the run happened. Compare them only against runs reporting the same scope.

Test set composition (43 questions): 14 exact-term lookups (function/class/enum names), 9 multi-file/how-it-works reasoning questions, and 20 conceptual/paraphrased questions. The conceptual questions avoid quoting source identifiers, so a retriever's hit rate on them reflects semantic matching rather than keyword overlap.

The hybrid arm applies the cosine relevance cutoff calibrated for `sentence-transformers/all-mpnet-base-v2` (`0.25`) to its vector component, resolved the same way the app resolves it, so this arm matches the shipped configuration for whichever embedder is configured. The vector-only arm is unthresholded to isolate raw embedding ranking quality; BM25 scores are never thresholded (zero-overlap documents are excluded by construction).

Avg Latency figures are comparable only across runs with similar latency probes — see each configuration's `results_<retriever>.md` for its probe.

| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Answered | Failed | Avg Latency |
|-----------|----------|-----|----------------|-------------------|----------|--------|-------------|
| vector | 0.8333 | 0.7321 | 0.5127 | 0.3953 | 43 | 0 | 0.9s |
| bm25 | 0.8333 | 0.6087 | 0.5414 | 0.3163 | 43 | 0 | 0.9s |
| hybrid | 0.8333 | 0.7044 | 0.4691 | 0.3953 | 43 | 0 | 0.9s |
