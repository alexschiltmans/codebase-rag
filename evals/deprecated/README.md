# Deprecated evaluation results

Nothing in this directory reproduces against the current setup. It is kept because the
reproduction lineage is what makes later claims checkable: several of these runs re-measured
each other across dates and explained their own deltas, and that record is worth more than the
disk it costs.

**Current results live one directory up, in `../retrieval-stack-findings.md`.**

| file | why it is here |
|---|---|
| `ablation.md`, `results_vector.md`, `results_bm25.md`, `results_hybrid.md` | measured against a `power-grid-model` plus `power-grid-model-ds` corpus |
| `results_small_model.md`, `results_large_model.md` | older 16-question test set, multi-repository corpus |

The `.json` files alongside each `.md` are the raw per-question output of the same runs.

Two things that are still true and are not deprecated with the numbers:

- The default retriever is BM25-only, and the decision these figures drove still stands. What
  lapsed is reproducibility of the figures, not the choice they justified.
- `results_large_model.json` holds the historical 30B run, still readable as an upper reference
  point for model-size questions.

`run_eval.py` writes to `evals/`, not here, so a fresh run produces fresh files one level up and
leaves this directory alone.
