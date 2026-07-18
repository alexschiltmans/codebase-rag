# Evaluation Results

This repo ships with a reproducible evaluation framework. The system was evaluated on 16 hand-picked questions against the PowerGridModel repository with two different model sizes.

See [evals/results.md](../evals/results.md), [evals/results_small_model.md](../evals/results_small_model.md) and [evals/results_large_model.md](../evals/results_large_model.md) for full breakdowns, or run your own:

```bash
python evals/run_eval.py
```

| Metric | Small model (350M) | Large model (30B) |
|--------|-------------------|-------------------|
| Avg keyword recall | 0.36 | 0.48 (+31%) |
| Avg source precision | 0.15 | 0.18 |
| Avg latency | 6.6s (Docker) | 5.6s (native GPU) |

**Key findings:**
- Cross-file reasoning is the system's strength (0.55 → 0.67 keyword recall)
- Retrieval is the bottleneck, not generation. Both models fail on the same 4 questions where the relevant chunk isn't in the top-5
- Enum/constant value lookups are consistently weak: the embedding model doesn't represent short code definitions well

## Retrieval Ablation

The app uses the hybrid retriever by default. To check whether that's actually the right call, the eval also runs the same 16-question test set through vector-only and BM25-only retrieval. Full results: [evals/ablation.md](../evals/ablation.md), [evals/results_vector.md](../evals/results_vector.md), [evals/results_bm25.md](../evals/results_bm25.md), [evals/results_hybrid.md](../evals/results_hybrid.md).

| Retriever | Keyword Recall | Source Precision | Avg Latency |
|-----------|----------------|-------------------|-------------|
| Vector-only | 0.3830 | 0.1875 | 0.9s |
| BM25-only | 0.4302 | 0.1750 | 0.8s |
| Hybrid | 0.3749 | 0.1750 | 0.8s |

On this test set, hybrid does not beat either component on its own: BM25-only gets the highest keyword recall, and hybrid comes in lowest of the three. Part of this is the test set itself, which leans heavily on exact-term lookups (function, class, and enum names), a case BM25 is built for. But the fusion logic probably isn't helping either. Scores are combined as a fixed 0.7/0.3 vector/BM25 blend, and BM25 is max-normalized per query so its top hit is always scaled to 1.0 (see `hybrid_search.py`) — that can let a middling keyword match drag down a document that BM25 alone would have ranked first. None of this means hybrid search is a bad idea; vector search should still pull its weight on paraphrased or conceptual questions that a keyword-heavy test set won't surface. It does mean the "hybrid beats either alone" claim wasn't backed by evidence when this was written, and that the 0.7/0.3 weights read like an initial guess rather than something tuned. Reciprocal Rank Fusion, which combines rankings instead of raw scores, is worth trying next.

## Limitations

- **Retrieval ceiling.** The embedding model (`all-mpnet-base-v2`) struggles with very short code constructs like enum values, constants, and build configuration variables. Questions about specific enum members or CMake variables often score 0% recall.
- **Single embedding model.** All content is embedded with the same model regardless of language. A specialised code embedding models might improve retrieval for code-heavy queries.
- **No incremental deletion.** When a file is removed from a repository, its chunks remain in Qdrant until a `--force` re-index is performed.
- **Local LLM quality.** The default 350M model is fast but imprecise. For production-quality answers, use a larger model (30B+ parameters) with GPU access.
- **Docker GPU limitations.** On macOS, Docker containers cannot access the GPU. Running Ollama natively on the host gives significantly better performance (5.5x faster in evaluation).
