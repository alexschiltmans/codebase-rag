# Retrieval Stack Findings

**Date:** 2026-08-05
**Corpus:** power-grid-model only, 12346 chunks
**Test set:** `evals/testset.json`, 29 scored questions (one is marked `expected_failure` and excluded)
**Harness:** `evals/bench_retrieval.py` for first stages, `evals/rerank_bench.py` for rerankers

This measures four model slots that had never been measured separately: the embedder, the candidate
depth, the reranker, and the generation/judge model size limit. No shipped default changed as a result
of this work.

The retrieval sections are retrieval-only, produced without an LLM and without a judge, so they say
nothing directly about answer quality. The generation and judge section is separate and says so.

## Read this before reading any table

**29 questions means one question is 3.45 points of hit rate.** Differences smaller than roughly 7
points are two questions or fewer and are reported here as not distinguishable, not as rankings. Four
decimal places appear below because that is what the harness emits, not because the third decimal
means anything.

MRR is the more informative metric in this test set. It is not quantized to whole questions the way
hit rate is, and several of the conclusions below rest on MRR moving consistently while hit rate sits
still.

## Embedder

All arms: vector retrieval, depth 10, unthresholded, fp16, one corpus, one metric.

| embedder | dim | hit | MRR | index build | query |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 1024 | 0.7931 | **0.6371** | 22.0 min | 40 ms |
| BAAI/bge-m3 | 1024 | 0.7931 | 0.5931 | 7.8 min | 31 ms |
| Qwen3-Embedding-4B | 2560 | 0.6897 | 0.5210 | 122.7 min | 104 ms |
| all-mpnet-base-v2 (shipped) | 768 | 0.6552 | 0.5020 | reused | 27 ms |

Two results matter here.

**Both sub-1B modern embedders beat the shipped one by 13.8 points, which is 4 questions and outside
the resolution limit.** This is the largest single-slot gain measured anywhere in this work.

**Scaling the embedder does not help.** The 4B model finishes last among the modern models while
costing 16 times bge-m3's index build and 3.8 times its query latency. It is not a marginal loss: it
trails the 0.6B by 10.3 points, which is 3 questions. Nothing above 1B is worth further evaluation in
this slot on a corpus of this kind.

Between the two winners, 0.6B has the better ordering (MRR 0.6371 against 0.5931) and bge-m3 builds
its index roughly three times faster. They tie exactly on hit rate, but see the ensemble note below,
because that tie is not the whole story.

## Candidate depth

Depth was fixed at 10 before this work and had never been varied.

| first stage | d10 | d50 | d100 |
|---|---|---|---|
| 0.6B hybrid | 0.8276 | 0.8966 | 0.9310 |
| bge-m3 hybrid | 0.7931 | 0.8966 | 0.9310 |
| shipped hybrid | 0.6897 | 0.7931 | 0.9310 |
| shipped vector | 0.6552 | 0.8276 | 0.8621 |
| shipped BM25 | 0.6897 | 0.7586 | 0.7931 |

The table is hit rate at each depth. Read at a single depth, hit rate and recall are the same
quantity here: a hit is scored by scanning the whole returned list, so "recall at depth 10" and "hit
at depth 10" are one number under two names, and the harness reports both. They are not two
independent signals and nothing below treats them as such.

What the table shows is that this one quantity climbs steeply with depth while MRR stays nearly flat
(shipped hybrid moves 0.4911 to 0.5186 across the same range). The answers were being retrieved and
then discarded by the depth cutoff, not missed. Deepening the candidate list costs almost nothing in
query latency, since every arm above stays within a few milliseconds of itself from depth 10 to
depth 100.

The place recall is genuinely a separate number is the reranker section, where a depth-50 or
depth-100 input list is scored against a depth-10 output. There the input list's recall is a real
ceiling on what reranking can reach, and it is reported next to each arm.

**0.9310 is the ceiling.** Three unrelated first stages reach exactly 27 of 29 and none exceeds it.
Across every configuration measured, exactly one question is retrieved by nothing at any depth:

> How do you create input data for a power-grid-model calculation?

## Reranker

Scored at output depth 10, over frozen candidate lists so no first stage is re-run.

| reranker | first stage | hit | MRR | latency mean / p95 |
|---|---|---|---|---|
| none | 0.6B hybrid d10 | 0.8276 | 0.6266 | 0 |
| bge-reranker-v2-m3 | 0.6B hybrid d50 | 0.8276 | **0.6580** | 1.80s / 2.80s |
| bge-reranker-v2-m3 | 0.6B hybrid d100 | **0.8621** | 0.6564 | 3.63s / 5.56s |
| bge-reranker-v2-m3 | 0.6B vector d50 | 0.8276 | 0.6592 | 1.82s / 2.84s |
| bge-reranker-v2-m3 | bge-m3 hybrid d50 | 0.7931 | 0.6331 | measured on CPU |
| bge-reranker-v2-m3 | shipped BM25 d100 | 0.6897 | 0.6092 | measured on CPU |
| qwen3.5:9b listwise | 0.6B hybrid d50 | 0.7931 | 0.6782 | 43.17s / 97.64s |
| ms-marco-MiniLM-L6-v2 | 0.6B hybrid d50 | 0.6897 | 0.5009 | 0.32s / 0.36s |

Latencies are comparable only within the same device. The CPU-measured rows ran while the GPU was
occupied by an embedder sweep, and CPU is roughly 2.2 times slower for this model.

**Reranking here buys ordering, not coverage.** Depth 50 plus a reranker ties depth 10 with no
reranker on hit rate, and gains 0.031 MRR for 1.8 seconds per query. The honest case for a reranker in
this stack is better ranking of what was already found, not finding more.

**Where it clearly pays is on a weak first stage.** On the shipped BM25 configuration it moved MRR from
0.4374 to 0.6092 with hit rate unchanged, and that gain requires no re-ingestion of anything.

**The cheap cross-encoder is worse than no reranker.** `ms-marco-MiniLM-L6-v2` degraded two of the
three lists it was given and flattened all three to an MRR of 0.4983 to 0.5009 regardless of what it
started from, including a list handed to it at 0.5677. It replaces the first stage's ordering with its
own rather than refining it. It is trained on web QA passages, and source code with reference
documentation is not that.

**The LLM listwise reranker is not worth its latency.** It produced the best MRR measured anywhere
(0.6782) and moved hit rate not at all, at a mean of 43 seconds per query and a p95 of 98 seconds.
Reporting only the mean would have understated the tail by more than a factor of two. One query in
twenty taking a minute and a half is not a cost a user-facing retrieval path can absorb.

A note for anyone re-running the listwise arm: `qwen3.5:9b` is a reasoning model and spends about 188
seconds on a single 20-passage window when allowed to think, against 7 to 10 seconds with thinking
disabled. The orderings it produced were comparable either way.

## Per-question flips against the shipped configuration

Aggregates hide which questions moved. Baseline is the shipped configuration (all-mpnet-base-v2,
hybrid, depth 10) at hit 0.6897.

| arm | hit | gained | lost | net |
|---|---|---|---|---|
| 0.6B hybrid d10 | 0.8276 | 4 | 0 | +4 |
| 0.6B vector d10 | 0.7931 | 4 | 1 | +3 |
| bge-m3 vector d10 | 0.7931 | 4 | 1 | +3 |
| 4B vector d10 | 0.6897 | 2 | 2 | 0 |

The four questions the modern embedders recover are all specific factual lookups that the shipped
embedder was missing: the required C++ standard, the third-party C++ dependencies, the calculation
method enum values, and the per-unit base power constant.

The 4B arm is the clearest argument for publishing flips rather than aggregates. It ties the shipped
configuration exactly on hit rate while disagreeing with it on four questions, two in each direction.
An aggregate tie can mean two systems behave identically or that they fail in different places, and
those call for different decisions.

**0.6B and bge-m3 tie at 0.7931 and fail on different questions.** bge-m3 recovers the short-circuit
fault types question that 0.6B misses; 0.6B recovers the C++ standard question that bge-m3 misses.
That is an ensemble result rather than a tie, and it means the ceiling for a combination of the two is
above either one alone. Nothing here measured such a combination.

## Recommendations per slot

These are inputs to an adoption decision, not the decision. Adopting a new embedder forces a full
re-ingestion of every existing collection, which is a user-visible migration.

**Embedder: Qwen3-Embedding-0.6B**, or bge-m3 if index build time matters more than ordering quality.
Either is worth 4 questions over the shipped model. Do not go above 1B.

**Depth: 50.** Depth 100 buys one more question for double the rerank latency, which is inside the
resolution limit. Depth 10 is too shallow and was leaving retrieved answers on the floor.

**Reranker: `BAAI/bge-reranker-v2-m3`**, if seconds per query are affordable. If they are not, ship the
first stage alone at depth 10 rather than reaching for a cheaper cross-encoder, because the cheap one
measured worse than nothing.

**Score threshold: none, for any newly adopted embedder.** The 0.6B model's relevant and irrelevant
score distributions overlap almost entirely (relevant mean 0.6750, irrelevant mean 0.6244, irrelevant
p90 0.7266 above the relevant mean). No cutoff separates them, and the first cutoff that removes
anything real costs 3 questions. `resolve_score_threshold` already returns no threshold for models
absent from its calibrated table, which is the correct behavior. Do not copy the shipped model's 0.25
onto a different embedder.

**Model cap: holds at 9B, now measured rather than assumed.** A dense `qwen3.5:27b` runs fine here
(18GB resident, degrading no worse under a concurrent embedding workload than the 9B does), so the
memory exhaustion the cap was set to avoid did not reproduce. It simply is not worth running: answer
quality at fixed retrieval is 6.9 points worse on keyword recall, and as a RAGAS judge it offers
nothing, because the 9B already completes 30 of 30 on every metric with no parse failures. It costs
3x the generation latency and 2.8x the judge latency to be no better.

The cap should be read as a claim about **dense** models. `qwen3.5:35b` is Mixture-of-Experts, larger
than both dense arms in total parameters, and the fastest model measured: 56.86 tok/s against the
dense 27B's 13.02 and the dense 9B's 38.59, at 23GB resident. Sparsity rather than size decides what
is cheap to run on this hardware, so a limit phrased in billions of parameters will wrongly exclude
models that run perfectly well.

## Generation and judge models

Measured at fixed BM25 retrieval on the same 29 questions, with model reasoning disabled (see the
caveat below).

| model | keyword recall | per question | as RAGAS judge | judge wall clock |
|---|---|---|---|---|
| qwen3.5:9b (dense 9.7B) | **0.6556** | 7.0s | 30/30 parsed, all metrics | 22.3 min |
| qwen3.5:27b (dense 27.8B) | 0.5867 | 20.9s | 30/30 parsed, all metrics | 61.5 min |

Retrieval metrics were identical to four decimals across both arms, confirming retrieval was held
fixed and the generation model was the only variable.

**RAGAS scores are judge-dependent and are not comparable across judges.** Scoring one fixed answer
set, the two judges disagreed by 0.100 on faithfulness (0.9046 against 0.8045), 0.069 on context
recall, and 0.034 on answer relevancy. There is no ground truth for which is correct; one calls the
answers more faithful and the other calls them more relevant. Any published RAGAS number has to name
its judge or it cannot be compared to anything.

**Thinking models cannot be evaluated through this harness as it stands.** `qwen3.5` models reason
before answering and the reasoning is charged against the generation token budget. At the shipped
1024-token limit the reasoning consumes the allowance and the answer is truncated mid-word or never
emitted, which scores as a quality collapse rather than as a configuration problem: the 27B first
measured 0.04 on keyword recall for this reason alone. Every number in the table above was produced
with reasoning disabled. This has not affected any previously published result, because the
configured generation model does not reason.

## Losing arms and dropped candidates

Published so the same ground is not re-covered:

- `Qwen3-Embedding-4B`: worst modern embedder measured, 16x the index build cost. Do not retry.
- `cross-encoder/ms-marco-MiniLM-L6-v2`: worse than no reranker on this corpus.
- `qwen3.5:9b` listwise: best MRR, unusable latency profile.
- Depth 5 on the shipped stack: 0.5862 hit, materially worse than depth 10 and included only to
  confirm the direction.

## Caveats

**Source matching is file-level and substring-based.** A question counts as a hit when any expected
source string appears anywhere in a retrieved document's path. It does not check that the retrieved
chunk contains the answer. This metric understates quality for arms that retrieve the right content
from a file the test set did not name, which was observed during this work.

**One corpus, named.** Every number is power-grid-model, a C++ and Python library with substantial
reference documentation. Results on a corpus of a different shape are not implied.

**Benchmark runtime is not production load.** Index build times and query latencies were measured on a
48GB M4 running one arm at a time, and in some cases while another sweep held the GPU. They are
ordering information, not capacity planning.

**Test set sources are narrow.** Several questions name fewer source files than legitimately answer
them, which caps measured hit rate below what a user would judge correct. Widening the test set would
change these numbers and is not attempted here.

**One embedder's index build time is missing.** The shipped model reused an existing collection rather
than rebuilding, so its build cost is not comparable to the others.

## Reproducing

```
python evals/bench_retrieval.py --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --retriever hybrid --depth 50 --dtype float16 --max-seq-length 768

python evals/rerank_bench.py \
  --candidates evals/bench_candidates/<arm>.json \
  --reranker BAAI/bge-reranker-v2-m3
```

Per-arm JSON is in `evals/bench_results/` and reranker output in `evals/rerank_results/`, both
tracked. The frozen candidate lists are written to `evals/bench_candidates/`, which is not tracked:
they run to 24MB because each one carries the full text of every candidate chunk, and re-running the
first stage regenerates them. Rescoring a reranker arm without re-embedding needs those lists, so
regenerate them before reaching for `rerank_bench.py`.

Large embedders need `--batch-size 16` and `--dtype float16`. The 4B at the default batch size and
float32 exhausted memory and stalled for over ten hours.
