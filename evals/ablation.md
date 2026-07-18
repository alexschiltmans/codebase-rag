# Retrieval Ablation

**Date:** 2026-07-18 23:14

Same test set (`evals/testset.json`), same LLM, same top_k — only the retriever feeding the RAG chain changes. Full per-question detail for each configuration is in `results_<retriever>.md`.

| Retriever | Keyword Recall | Source Precision | Answered | Failed | Avg Latency |
|-----------|----------------|-------------------|----------|--------|-------------|
| vector | 0.3830 | 0.1875 | 16 | 0 | 0.9s |
| bm25 | 0.4302 | 0.1750 | 16 | 0 | 0.8s |
| hybrid | 0.3749 | 0.1750 | 16 | 0 | 0.8s |
