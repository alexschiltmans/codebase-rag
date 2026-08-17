# Retrieval Ablation

**Date:** 2026-08-17 07:14

**Retrieval stages:** rerank (`BAAI/bge-reranker-v2-m3`, depth 50), rewrite (timeout 5.0s) (every arm ran under this stack)

Same test set (`evals/testset.json`), same LLM, same top_k; only the retriever feeding the RAG chain changes. Full per-question detail for each configuration is in `results_<retriever>_rerank_rewrite.md`.

**Repositories retrieved from:** `power-grid-model`. Every arm was restricted to these, so these figures are a measurement of this corpus rather than of whatever else shared the index when the run happened. Compare them only against runs reporting the same scope.

Test set composition (43 questions): 14 exact-term lookups (function/class/enum names), 9 multi-file/how-it-works reasoning questions, and 20 conceptual/paraphrased questions. The conceptual questions avoid quoting source identifiers, so a retriever's hit rate on them reflects semantic matching rather than keyword overlap.

The hybrid arm applies the cosine relevance cutoff calibrated for `sentence-transformers/all-mpnet-base-v2` (`0.25`) to its vector component, resolved the same way the app resolves it, so this arm matches the shipped configuration for whichever embedder is configured. The vector-only arm is unthresholded to isolate raw embedding ranking quality; BM25 scores are never thresholded (zero-overlap documents are excluded by construction).

Avg Latency figures are comparable only across runs with similar latency probes; see each configuration's `results_<retriever>_rerank_rewrite.md` for its probe. Prompt tokens per answer and TTFT are the efficiency metrics. They are reported per configuration rather than predicted from precision: top_k is fixed, so better ordering changes which chunks are sent and not how many, and prompt tokens move with chunk length instead. p95 columns sit next to the means because a mean alone hides the tail. Prompt tokens and TTFT are averaged only over questions that actually retrieved and generated; `Eff. Qs` is that denominator, so an arm that fails to retrieve does not look cheaper for it.

| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Answered | Failed | Avg Latency | p95 Latency | Prompt Tokens | Avg TTFT | p95 TTFT | Eff. Qs |
|-----------|----------|-----|----------------|-------------------|----------|--------|-------------|-------------|---------------|----------|----------|---------|
| vector | 0.7857 | 0.6663 | 0.5152 | 0.3767 | 43 | 0 | 3.9s | 5.1s | 799 | 3.3s | 4.2s | 43 |
| bm25 | 0.8095 | 0.7238 | 0.5828 | 0.4000 | 43 | 0 | 4.0s | 4.9s | 816 | 3.3s | 4.4s | 43 |
| hybrid | 0.7619 | 0.6964 | 0.5410 | 0.3535 | 43 | 0 | 4.0s | 5.1s | 814 | 3.3s | 4.4s | 43 |
