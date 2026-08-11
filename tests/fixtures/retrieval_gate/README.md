# Retrieval gate fixture

These files are frozen inputs to a measurement, not example code. Nothing imports them and
nothing runs them. They exist so that `tests/evaluation/test_retrieval_gate.py` can chunk and
retrieve over a corpus that does not move when the application changes.

## Contents

- `corpus/` — the documents. Deliberately unrelated to this project's own subject matter, so a
  query cannot accidentally match on vocabulary the retrieval code itself uses. Sixteen files
  across Python and Markdown, chunking to 92 chunks at the default 614-character chunk size.
- `queries.json` — the query set. `expected_sources` are matched as case-insensitive substrings
  against retrieved document paths, the same convention `evals/retrieval_metrics.py` uses.
- `vector_ranking.json` — a frozen ranked list per query, standing in for what a vector search
  returned. Generated once with a TF-IDF cosine scorer, which is a different scoring family from
  BM25, so the two arms genuinely disagree and rank fusion has something to fuse. It is a fixture,
  not a measurement: it says nothing about vector retrieval quality.
- `thresholds.json` — the recorded band. Both a floor and a ceiling per arm per metric.

## Changing anything here

Editing the corpus, the queries, or the vector ranking changes the measurement. Any such edit has
to re-record `thresholds.json` in the same commit, and the change has to say why the previous band
no longer describes correct behaviour. Widening a bound to make a red gate go green is the one
thing this fixture exists to prevent.

## Cost

The gate runs both arms over 18 queries against 92 chunks in roughly 0.4 seconds on a 2021
MacBook Pro, dominated by chunking rather than retrieval. If it grows enough to be worth
optimising, the chunking step is what to cache.
