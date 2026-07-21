# Retrieval Ablation

**Date:** 2026-07-21 09:37

Same test set (`evals/testset.json`), same LLM, same top_k — only the retriever feeding the RAG chain changes. Full per-question detail for each configuration is in `results_<retriever>.md`.

The hybrid arm applies the production cosine relevance cutoff (`VECTOR_SCORE_THRESHOLD=0.25`) to its vector component, matching the app's shipped configuration. The vector-only arm is unthresholded to isolate raw embedding ranking quality; BM25 scores are never thresholded (zero-overlap documents are excluded by construction).

Avg Latency figures are comparable only across runs with similar latency probes — see each configuration's `results_<retriever>.md` for its probe.

| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Answered | Failed | Avg Latency |
|-----------|----------|-----|----------------|-------------------|----------|--------|-------------|
| vector | 0.4000 | 0.2967 | 0.4074 | 0.1875 | 16 | 0 | 0.8s |
| bm25 | 0.4000 | 0.2244 | 0.4473 | 0.1750 | 16 | 0 | 0.9s |
| hybrid | 0.2667 | 0.2333 | 0.3406 | 0.1750 | 16 | 0 | 0.8s |

## Run provenance and caveats

This table was regenerated from a run on 2026-07-21 whose judge was `qwen3.5:9b`
(reasoning disabled) served by a native macOS Ollama with Metal acceleration, not
the CPU-only Ollama in Docker. Generation still used the shipped 350M model. The
Hit Rate and MRR columns are new: they score retrieval directly against each
question's expected sources, independent of the generated answer, so they are the
metric the retriever choice should be read from. They are deterministic and do not
depend on the judge.

Latency figures here reflect Metal-accelerated inference and are not comparable to
the Docker-deployment latencies in older reports.

The RAGAS judge metrics are not shown in this cross-retriever table because they
did not complete uniformly across arms. Faithfulness and answer relevancy were
measured (context_recall was skipped — the local judge cannot produce its output
format); per-arm coverage was: vector faithfulness 16/16, bm25 15/16, hybrid 13/16.
Hybrid faithfulness fell below the 0.9 coverage gate, so `results_hybrid.*` was not
republished this cycle. The RAGAS scores that did clear the gate are in
`results_vector.md` and `results_bm25.md`, and all of them are summarized with
their coverage in `docs/evaluation-results.md`.
