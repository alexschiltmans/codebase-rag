# Retrieval Stack Findings

**Date:** 2026-08-05, updated same day after the test set was strengthened, again after chunking
changed, again after the modern embedders were re-measured under that chunking, on 2026-08-06
after chunk size was swept per embedder, and twice on 2026-08-17: after the rerank and rewrite
stages were measured end to end, and after the shipped retriever default was re-examined
**Corpus:** power-grid-model only, 19637 chunks under the current chunking. Every table below the
chunking update was measured at 12346, before it.
**Test set:** `evals/testset.json`, 42 scored questions as of the update below (originally 29; one
question is marked `expected_failure` and excluded either way)
**Harness:** `evals/bench_retrieval.py` for first stages, `evals/rerank_bench.py` for rerankers, and
`evals/run_eval.py` for the end-to-end stage measurement in the first update below

This measures four model slots that had never been measured separately: the embedder, the candidate
depth, the reranker, and the generation/judge model size limit. No shipped default changed as a result
of this work.

The retrieval sections are retrieval-only, produced without an LLM and without a judge, so they say
nothing directly about answer quality. The generation and judge section is separate and says so.

## Update: the shipped retriever default re-examined (2026-08-17)

The default was reconsidered because the reason recorded for it had stopped being true: BM25 was
selected on recall, and all three retrievers now find the expected source on the same count of
questions. A full re-run of `run_eval.py` with the fixed `qwen3.5:9b` judge, both optional stages
off, same corpus and scope as the update below. All three arms scored 43/43 on every ragas metric
and none is self-judged. **The default did not change; the justification for it did.**

| arm | coverage | rank-1 | MRR | source precision | context recall | keyword recall | faithfulness | answer rel |
|---|---|---|---|---|---|---|---|---|
| bm25 (ships) | 35/42 | 21/42 | 0.6087 | 0.3163 | **0.5638** | **0.5577** | 0.5438 | 0.8176 |
| hybrid | 35/42 | **26/42** | 0.7044 | **0.3953** | 0.4791 | 0.5050 | 0.4750 | **0.8332** |
| vector | 35/42 | 27/42 | **0.7321** | 0.3953 | 0.5240 | 0.4786 | 0.5148 | 0.8559 |

Rank-1 is not emitted by the harness; it is recomputed from each arm's saved `sources_actual` with
`compute_retrieval_hit_and_reciprocal_rank`, the same function behind hit rate and MRR.

What reproduces, and what does not. Every retrieval-side figure above is bit-identical to the
previous run, as are the retrieved contexts themselves: 43/43 questions returned the same chunks in
the same order in all three arms. Retrieval here is deterministic and carries no run-to-run error.
Generation is not, despite `temperature=0.0`: only 1/43 and 2/43 answers were identical between the
two runs, and every judged metric inherits that. Measured between the two runs, context_recall moved
by at most 0.021, answer_relevancy by at most 0.034, and **faithfulness reversed sign** (hybrid led
BM25 by 0.077, then trailed by 0.069). Faithfulness cannot support a comparison at this sample size
and should not be quoted from either run.

The two retrievers split, reproducibly, along a line worth naming:

- **Hybrid is better at file-level targeting.** It ranks the expected file first on 26/42 questions
  against 21/42, and a higher fraction of what it returns matches the expected sources (0.3953
  against 0.3163). The rank-1 gap is broad rather than driven by outliers: hybrid wins 11 questions,
  BM25 wins 6, 15 are tied at rank 1, 10 at neither.
- **BM25 retrieves more of the answer's content.** It leads context_recall by 0.085 and keyword
  recall by 0.053, both same-sign across two runs and both outside their measured noise. These come
  from different mechanisms, one judged and one purely lexical, which is why they carry more weight
  together than either would alone.

The default stays `bm25` on that split. All five retrieved chunks are concatenated into one prompt,
so on the app's answer surface what the context contains matters more than the order it arrives in;
rank-1 is the better metric for the ranked-list surfaces (`codebase-rag query`, `POST /search`), and
those give up something real here. Caveats worth carrying: 42 questions on one repository, and both
content metrics could carry a lexical bias toward the keyword retriever, since reference answers and
BM25 both key off the question's wording.

Two things the coverage column hides. The 35/42 tie is a coincidence of equal-sized failure sets, not
agreement: BM25 finds 4 questions hybrid misses entirely and hybrid finds 4 BM25 misses. And the arms
are not returning the same material reordered, mean Jaccard overlap of their retrieved chunk sets is
0.183.

The `vector` arm remains unshippable evidence and is listed for reference only: it is measured with
the relevance cutoff disabled to isolate raw embedding ranking, so its leading MRR does not describe
a configuration the application can be set to.

Availability, measured separately against a live index rather than argued from the code: with
`RETRIEVER=hybrid` and Qdrant unreachable, retrieval degrades to BM25 ordering rather than failing,
because the store's existence check swallows the connection error and the fusion rescales against
whichever rankers returned results. Cost is 0.01s on a refused connection and 1.06s on an unroutable
host, bounded near 5s by the client's default timeout. The exception is a store that answers its
existence check and then fails the query itself, which propagates and fails the question outright.

## Update: rerank and rewrite measured end to end (2026-08-17)

Every reranker figure elsewhere in this document is retrieval-only, scored over frozen candidate lists
at output depth 10. This update is the first measurement of those stages in the path the application
actually runs: `run_eval.py`, all 43 questions, generation and a fixed `qwen3.5:9b` judge, three
retrievers times four stage configurations. All twelve arms scored 43/43 on every ragas metric and
none is self-judged. Per-arm files are `evals/results_<retriever>[_rerank][_rewrite].{json,md}`.

One caveat on those files, added after the fact: the three no-stage arms were re-run later the same
day for the default re-examination above, so `results_{bm25,hybrid,vector}.{json,md}` now hold the
newer run's numbers and no longer match the `baseline` rows below. The retrieval columns are
identical between the two runs; the judged columns are not, faithfulness least of all. The tables in
this section are left as measured, because every row in them comes from one run and they are only
meaningful compared against each other. Read a baseline row against the rows beside it, not against
the regenerated file.

On the shipped BM25 retriever:

| config | hit | MRR | prompt tokens | TTFT | p95 latency | faithfulness / context recall |
|---|---|---|---|---|---|---|
| baseline | **0.8333** | 0.6087 | **801** | **0.163s** | **1.532s** | 0.462 / 0.564 |
| rewrite | 0.7381 | 0.5595 | 809 | 1.173s | 2.923s | 0.560 / 0.572 |
| rerank | 0.7857 | 0.6833 | 841 | 1.505s | 2.821s | 0.593 / 0.631 |
| rerank + rewrite | 0.8095 | **0.7238** | 816 | 3.263s | 4.903s | 0.555 / 0.676 |

**Both stages ship off by default.** Hit rate does not improve in any of the nine stage configurations
on any retriever, prompt tokens never fall, and TTFT regresses by 7x to 20x. What the stages buy is
ordering and grounding, not coverage, which is the same conclusion the retrieval-only tables reach,
now with the live cost attached to it.

**Output depth changes the reranker's hit rate story.** The tables below show hit rate flat under
reranking, scored at output depth 10. The application returns 5, and at 5 the same reranker loses
coverage on every first stage: BM25 0.8333 to 0.7857, vector and hybrid 0.8333 to 0.8095. Over a
depth-50 pool the cross-encoder promotes documents the first stage ranked 6 to 50 above one that was
already inside the top 5. A reranker evaluated at output depth 10 will look better than it behaves at
output depth 5.

**The losses are one failure mode.** On vector and hybrid the five lost questions are identical and
all five are exact-term lookups: the C++ standard, the third-party dependencies, the transformer
winding types, the short-circuit fault types, and the minimum CMake version. Three were at rank 1
before reranking. A cross-encoder scores how much a passage reads like an answer, and a `find_package`
line or an enum block does not read like one next to prose discussing the same topic.

**The stages do not compose additively.** Rewriting alone is the worst configuration measured on BM25
(MRR 0.5595). Reranking alone is 0.6833. Together they reach 0.7238 and recover three exact-term
lookups that reranking alone loses, including the dependencies question. Rewriting widens the
candidate pool but rescores the top 5 and damages the ordering; reranking cannot reach what the first
stage never surfaced but repairs ordering once it is there. Neither is worth enabling alone on this
corpus, and the pair costs 3.263s to first token.

**Candidate depth 50 confirmed against 100 on the shipped first stage**, which had only been measured
at 100 before. From one BM25 candidate list truncated to each depth, scored at output depth 10: input
recall is identical at 0.9048, so ranks 51 to 100 hold nothing BM25 had not already found by rank 50.
MRR is 0.7357 at depth 50 against 0.7294 at depth 100, and latency is 2.13s / 2.44s p95 against
4.32s / 4.85s. Depth 100 costs twice the latency for no reachable document.

**What blocks the efficiency case is `top_k`, not the models.** It is fixed at 5, so better ordering
changes which five chunks are sent and never how many, and added query terms displace documents rather
than supplementing them. Adaptive k, or fusing the expanded and unexpanded result lists, is where the
next attempt should start.

## Update: chunk size swept per embedder

The section below this one showed that both modern embedders lose ground when chunks shrink while the
shipped model gains it, which means chunk size and embedder were confounded in every comparison in
this document. Each candidate had only ever been scored at a chunk size derived from the incumbent's
own token window. This sweep separates them: four chunk sizes, three embedders and a BM25 control,
one corpus rebuilt per size, everything else held fixed. All figures N=42, depth 10, unthresholded,
the two modern models at `--dtype float16 --max-seq-length 768` as their published arms used.

| model | arm | @614 | @1000 | @1228 | @1800 |
|---|---|---|---|---|---|
| all-mpnet-base-v2 | vector | 0.8571 / **0.7345** | 0.8571 / 0.7133 | 0.8333 / 0.6854 | 0.8333 / 0.6431 |
| all-mpnet-base-v2 | hybrid | **0.9048** / **0.7560** | 0.8810 / 0.7508 | 0.8571 / 0.6952 | 0.8571 / 0.6894 |
| bge-m3 | vector | 0.8571 / 0.6693 | **0.8810** / **0.7002** | 0.8571 / 0.6953 | 0.8571 / 0.6963 |
| bge-m3 | hybrid | **0.8810** / **0.7448** | 0.8571 / 0.7361 | 0.8571 / 0.7143 | 0.8571 / 0.7176 |
| Qwen3-0.6B | vector | 0.7381 / 0.5088 | **0.8095** / **0.6271** | **0.8095** / 0.6209 | 0.7619 / 0.5658 |
| Qwen3-0.6B | hybrid | 0.7619 / 0.5415 | **0.8333** / 0.6755 | **0.8333** / **0.6903** | 0.7857 / 0.6244 |
| BM25 | keyword | 0.8571 / 0.6111 | 0.8571 / 0.5830 | 0.8333 / **0.6334** | **0.8810** / 0.5874 |

Chunks per corpus and the share each model would silently truncate:

| chunk size | chunks | mpnet @384 | bge-m3 @768 | 0.6B @768 |
|---|---|---|---|---|
| 614 | 19637 | 0.59% | 0.00% | 0.00% |
| 1000 | 12346 | 31.34% | 0.00% | 2.38% |
| 1228 | 10276 | 44.19% | 0.02% | 17.50% |
| 1800 | 7518 | 64.63% | 12.28% | 27.76% |

**Every arm at 1000 reproduces its previously published figure exactly**, to four decimals: mpnet
vector 0.8571/0.7133, bge-m3 vector 0.8810/0.7002, 0.6B vector 0.8095/0.6271, 0.6B hybrid
0.8333/0.6755. The corpora were rebuilt from the checkout by a different code path than the one that
originally produced them, and the chunk counts (19637 at 614, 12346 at 1000) and truncation shares
(0.59% and 31.34% for the incumbent) came out identical too. The sweep is measuring the same
instrument the older tables were measured on.

### No embedder beats the incumbent when each is given its own best size

This is the question the sweep was run to answer, and the answer is no.

| model | best arm | hit | MRR |
|---|---|---|---|
| all-mpnet-base-v2 | hybrid @614 | **0.9048** | **0.7560** |
| bge-m3 | hybrid @614 | 0.8810 | 0.7448 |
| Qwen3-0.6B | hybrid @1228 | 0.8333 | 0.6903 |
| BM25 | keyword @1800 | 0.8810 | 0.5874 |

The incumbent leads on both metrics at every model's own best. bge-m3 is 1 question behind on hit and
0.0112 behind on MRR, which is inside the resolution floor and should be read as a tie rather than a
loss. 0.6B is 3 questions behind, which is exactly the width this document calls a ranking.

**Both embedders' best size is 614, the size that already ships.** Only 0.6B prefers something else,
and it is the arm that loses. Giving each model its own chunk size changes no ordering, so the
recommendation not to migrate is unchanged and now rests on the axis that was previously confounded.

### The window under-fill explanation is half right, which means it is not the explanation

The prediction was that each model's best size tracks its sequence window: the incumbent peaking at or
below 614, the two 768-token models near 1228.

It fits 0.6B well. 614 is its worst size by a wide margin, it gains 3 questions moving to 1000, and its
best MRR is at 1228. It does not fit bge-m3 at all, whose hybrid best is at 614 and which moves less
across the whole sweep (0.031 MRR) than 0.6B moves between two adjacent sizes. **Two models with the
same 768-token cap behave oppositely**, so the window is not what is driving it, and the mechanism
remains unexplained.

Truncation explains a different part of the picture and explains it cleanly. The incumbent's MRR falls
monotonically as chunks grow (0.7345 to 0.6431 on vector) and its truncation rises monotonically over
the same range (0.59% to 64.63%). Nothing else in the grid moves monotonically with anything. Note
what this means for 614: it is not a size that flatters small windows, it is the only size where no
model truncates materially, and the incumbent wins there on the merits.

**1228 was chosen badly and should not be read as a neutral point.** It was picked as 1.6 characters
per token against a 768-token window, the point where the modern models' windows would be nominally
full and not yet over. Measured, 0.6B truncates 17.50% of chunks there while bge-m3 truncates 0.02%.
The 1.6 ratio in `chunking.py` is one tokenizer's number applied to every model, and for 0.6B on this
corpus it is optimistic by enough to matter.

### fp16 arms do not reproduce across runs, and that bounds every modern-embedder number here

Four arms were run twice, once for the re-measurement below and once inside this sweep, over corpora
that are provably identical:

| arm | precision | first run | second run |
|---|---|---|---|
| BM25 @614 | n/a | 0.8571 / 0.6111 | 0.8571 / 0.6111 |
| mpnet vector @614 | float32 | 0.8571 / 0.7345 | 0.8571 / 0.7345 |
| bge-m3 vector @614 | float16 | 0.8571 / 0.6739 | 0.8571 / 0.6693 |
| 0.6B vector @614 | float16 | 0.7143 / 0.5378 | 0.7381 / 0.5088 |

BM25 is deterministic given a corpus, and it came out identical, which is what rules out the corpus
and the scorer as the source. The float32 arm is identical too. Both float16 arms moved, one of them
by a full question of hit rate and 0.029 of MRR.

The clean reading is that fp16 embedding on this hardware is not bit-reproducible, and cosine
neighbours close enough to swap places do swap. Precision is the only variable separating the arms
that reproduced from the arms that did not, though two paired runs cannot fully separate precision
itself from anything else that differs between fp16 and fp32 execution.

**Every modern-embedder figure in this document is fp16**, so all of them carry this on top of the
2.38-point test-set resolution. A one-question difference between two fp16 arms is not evidence of
anything. This is a further reason the bge-m3 result above is a tie: its margin is smaller than the
movement its own arm showed between two runs of the same configuration.

## Update: the modern embedders re-measured under current chunking

Every modern-embedder arm in this document was built at the old fixed 1000-character chunking. Both
bench collections held 12346 points. When chunking changed, only the shipped model was re-measured on
the rebuilt 19637-chunk index; the alternatives were left at the chunking of the time. The corrected
embedder table below is internally consistent, since all four of its rows sit at 12346, but it stopped
describing what ships the moment chunk size changed.

The two leading alternatives were therefore re-embedded over the current corpus at the same encoding
settings their published arms used (`--dtype float16 --max-seq-length 768`), so chunk size is the only
variable that moved. All figures N=42, unthresholded.

| arm | hit @614 | MRR @614 | hit @1000 | MRR @1000 |
|---|---|---|---|---|
| shipped vector d10 | 0.8571 | **0.7345** | 0.8571 | 0.7133 |
| shipped hybrid d10 | **0.9048** | **0.7560** | 0.8810 | 0.7508 |
| shipped vector d50 | 0.9286 | **0.7392** | 0.8810 | — |
| shipped hybrid d50 | 0.9286 | **0.7583** | 0.9048 | — |
| bge-m3 vector d10 | 0.8571 | 0.6739 | 0.8810 | 0.7002 |
| bge-m3 hybrid d10 | 0.8571 | 0.7167 | 0.8571 | — |
| bge-m3 vector d50 | 0.9286 | 0.6788 | — | — |
| bge-m3 hybrid d50 | 0.9286 | 0.7268 | 0.9286 | — |
| 0.6B vector d10 | 0.7143 | 0.5378 | 0.8095 | 0.6271 |
| 0.6B hybrid d10 | 0.7619 | 0.5953 | 0.8333 | 0.6755 |
| 0.6B vector d50 | 0.7619 | 0.5415 | — | — |
| 0.6B hybrid d50 | 0.7619 | 0.5916 | 0.8810 | — |

**Both alternatives lost ground when chunks shrank, and the shipped model gained it.** At 614 characters
the shipped model is at least tied on hit rate and ahead on MRR at every arm measured here. bge-m3
matches it on hit rate at both depths and trails on MRR by 0.03 to 0.06 throughout.
Qwen3-Embedding-0.6B falls to net −6 questions against the shipped model at d10, gaining 3 and losing 9,
against net −3 at the old chunking.

**Chunk size and embedder are not independent slots, and this document had been treating them as if
they were.** The four questions the 0.6B lost to the chunking change are the serialization formats,
the batch-data representation, the sensor-mixing rule, and the default realism checks. The likely
mechanism is window under-fill: a 768-token window at 614 characters holds about a third of capacity,
so the capacity that distinguishes these models from a 384-token one goes unused, while the shipped
model's window is well matched to that size. This is a hypothesis consistent with the direction and
size of the movement, not something these runs isolate; testing it would mean sweeping chunk size per
embedder, which has not been done.

Depth does not rescue the 0.6B. Its d50 arms reach the same 0.7619 as d10, so what it loses at this
chunking is coverage rather than ordering, and no amount of candidate depth returns a document the
first stage never surfaced.

**Nothing measured justifies an embedder migration, and that now holds at two chunkings rather than
one.** The fallback previously recorded here, that bge-m3 is the one to take if a
migration is wanted anyway, is weaker than it was: at current chunking it buys no hit rate at either
depth and costs MRR at both. That sentence also overstated its case when written, describing bge-m3 as
the only modern embedder not behind the shipped model on either metric while the table it sat above
showed 0.7002 against 0.7133 on MRR. It was behind on MRR then too, by less than a question's worth.

Method note: when these runs were made, arm records encoded the model, retriever, depth, precision and
sequence length in their filenames but not the chunking, so a re-run at a new chunk size overwrote the
old arm in place. Chunking is in the filename now, as `_chunk<size>`.

**The @614 figures above are not backed by the arm files that carry those names.** Naming by chunk
size arrived after these runs and generates exactly the `_chunk614` suffix they had been given by
hand, so the sweep in the section below collided with them and overwrote them. The files now hold the
sweep's values: 0.7381 / 0.5088 for `0.6B vector d10` against the 0.7143 / 0.5378 published here, and
0.6693 for `bge-m3 vector d10` against 0.6739. Both runs of both arms are preserved in the fp16
reproducibility table above, which is what the difference between them is evidence about. The table
here is left at its original values because the conclusions in this section were drawn from them.

## Update: re-measured under model-derived chunking (19637 chunks, was 12346)

Chunk size stopped being a fixed 1000 characters and is now derived from the embedding model's token
window, which for `all-mpnet-base-v2` gives 614 characters with 122 overlap. The corpus was rebuilt at
that size, growing from 12346 to 19637 chunks. The reason was truncation rather than retrieval: at
1000 characters, 31.34% of chunks exceeded the model's 384-token limit and had their tails silently
dropped before embedding, including 9.11% of `.cpp` chunks and 69.23% of `.json`. At 614 that falls to
0.59% overall, and only `.svg` still truncates meaningfully.

Every table below this section was measured at the old chunking. The three shipped arms were
re-measured on the rebuilt index, against the same 42-question set:

| arm | hit (12346) | hit (19637) | MRR (12346) | MRR (19637) |
|---|---|---|---|---|
| vector d10 | 0.8571 | 0.8571 | 0.7133 | **0.7345** |
| hybrid d10 | 0.8810 | **0.9048** | 0.7508 | **0.7560** |
| BM25 d10 | 0.8571 | 0.8571 | 0.5830 | **0.6111** |

**Nothing regressed.** MRR rose on all three arms and hybrid picked up a question, though at 2.38 points
per question none of that is wide enough to call a ranking by this document's own convention. The
useful result is the negative one: cutting chunks to 61% of their former size was expected to cost
context, and it cost no measurable retrieval quality.

The other tables have not been re-run. Read them as what their arms scored at the chunking of the time,
not as current figures.

## Update: measured against the strengthened test set (42 scored, was 29)

The test set changed after this document was first published: expected sources were widened where
they omitted a file that legitimately answers the question, one previously unanswerable question was
given real ground truth, and 13 new questions were added. Every source pattern was then scoped against
the corpus so it can only match the files it was meant to name, which corrected a set of hits that were
being scored against files that do not answer the question; see the two sections on the CMake case
below. The isolated effect of widening alone, at the set still held to 30 questions, is in
`evals/widening-rescore-n30.md`.

**Figures below this point and figures above it are not comparable.** The denominator, the ground
truth, and in most cases the retrieved candidate set all changed. Do not read a number changing across
this line as retrieval improving or regressing; read it as the instrument changing, exactly as
`evals/widening-rescore-n30.md` demonstrates for the widening-only case. The tables above this section
are kept as the historical record of what was measured on 2026-08-05 against the original 29-question
set, not as something to compare against.

**Resolution is now 100/42 = 2.38 points per question.** Following the same convention as the original
measurement, a difference is only called a ranking here when it spans at least 3 questions (roughly
7.1 points), a full question wider than the 2-question ambiguity zone the original set could not clear.

### The headline embedder claim reverses at the corrected N

| embedder | dim | hit | MRR |
|---|---|---|---|
| BAAI/bge-m3 | 1024 | 0.8810 | 0.7002 |
| all-mpnet-base-v2 (shipped) | 768 | 0.8571 | **0.7133** |
| Qwen3-Embedding-4B | 2560 | 0.8333 | 0.5998 |
| Qwen3-Embedding-0.6B | 1024 | 0.8095 | 0.6271 |

Same arms as the original embedder table: vector retrieval, depth 10, unthresholded, fp16, **chunk
size 1000**. Every row here was scored over a corpus cut at 1000 characters, which is not the size
that ships; see the per-embedder sweep at the top of this document for what these models do at 614
and elsewhere.

**"Both sub-1B modern embedders beat the shipped one by 13.8 points" is withdrawn; at the corrected
ground truth neither beats it at all.** The shipped model is second on hit rate, 1 question behind
bge-m3, and first on MRR ahead of every modern embedder. 0.6B, the model that led the original table,
is now last on hit rate, 2 questions behind the shipped model. The only difference in this table wide
enough to call a ranking is bge-m3 over 0.6B: 7.15 points, exactly 3 questions.

Two separate corrections produced this, and it is worth keeping them apart. Growing the set from 29 to
42 questions cost the modern embedders their margin; scoping the source patterns cost them the lead.
Two of the four questions the modern embedders were originally credited with recovering are the C++
build questions, and both were scored against a bare `CMakeLists.txt` pattern that matches 16 files in
this corpus. Scoped to the root file that actually answers them, both become misses for every modern
embedder, while the shipped model still hits both, reaching them through `build-guide.md` rather than
through any build file.

**Scaling still doesn't help.** 4B trails bge-m3 by 2 questions and beats 0.6B by 1, so it is no longer
the worst modern embedder, but it is worst on MRR of anything measured and still costs 16 times bge-m3's
index build. Nothing here argues for going above 1B, so that recommendation stands, now on cost rather
than on a measured quality gap.

**Per-question net movement against the shipped hybrid-d10 baseline (0.8810), corrected testset:**

| arm | hit | gained | lost | net |
|---|---|---|---|---|
| bge-m3 vector d10 | 0.8810 | 2 | 2 | 0 |
| 0.6B hybrid d10 | 0.8333 | 3 | 5 | −2 |
| 4B vector d10 | 0.8333 | 3 | 5 | −2 |
| 0.6B vector d10 | 0.8095 | 3 | 6 | −3 |

No modern embedder is now net positive against the shipped configuration, against the +3/+4 the
original table reported. The gains are real and stable across all three corrections: each modern arm
still recovers 2 to 3 questions the shipped model misses. What changed is the other column. The losses
were being undercounted, first by a test set too small to expose them and then by source patterns loose
enough to score a miss as a hit.

### One new question had the same defect this work targets, and one only looked like it

Reviewing the newly added questions against what the arms actually retrieved turned up one real
narrow-source case and one false alarm, and the difference between them is worth recording.

The real one: Qwen3-Embedding-4B missed *"What serialization formats does power-grid-model's
serializer support?"* by retrieving `_core/serialization.py`, which defines `SerializationType.JSON`
and the msgpack path, against ground truth that listed only `serialization.md`. Widening the sources
to include `serialization.py` is correct and the numbers above reflect it.

The false alarm: *"What is the minimum CMake version required to build power-grid-model?"* looked like
the same defect, because 0.6B and 4B both retrieved a file named `CMakeLists.txt` and both scored a
miss. Widening the sources to a bare `CMakeLists.txt` did turn both into hits, and that was wrong.
Sources are matched as case-insensitive substrings of the full path, so `CMakeLists.txt` credits any
of the 16 `CMakeLists.txt` files in this corpus, ten of which are under `tests/`. The file that
actually states `cmake_minimum_required(VERSION 3.23)` is the repository root `CMakeLists.txt`, and no
arm retrieves it at any depth; what 0.6B and 4B retrieved were the `power_grid_model_c` subdirectory
build files, which do not state a minimum version. The ground truth now reads
`power-grid-model/CMakeLists.txt`, scoped so it can only match the root file, and both arms are back
to a miss, which is the honest answer.

Finding it prompted an audit of every source pattern in the test set against the corpus, on the theory
that a defect this easy to introduce is unlikely to be alone. It was not. Eleven more patterns matched
more than one file, and two of them were doing real damage. `LICENSE` matched 671 paths, because this
repository carries a `.license` sidecar next to almost every data file and all of them are ingested, so
that question could be scored a hit on any of them. `CMakeLists.txt` appeared in two further questions,
the C++ standard and the third-party dependencies, both predating this work and both credited on the
strength of build files that do not answer them. Those two are what moved the embedder table above.
Every pattern is now scoped with enough of its parent path to be unambiguous, verified by matching each
one against the corpus file list and requiring exactly one file.

Three things to carry forward. Widening ground truth is not free: a source pattern loose enough to catch
the file you meant is often loose enough to catch files you did not, and the substring convention gives
no warning when it does. A miss is only evidence of narrow ground truth once you have read the chunk
that was actually retrieved, not merely its filename. And a basename is not an identifier: check a new
pattern against the corpus before trusting a number it produces.

### Depth and reranker tables at the new N

| first stage | d10 | d50 | d100 |
|---|---|---|---|
| bge-m3 hybrid | 0.8571 | 0.9286 | 0.9762 |
| shipped hybrid | 0.8810 | 0.9048 | 0.9762 |
| shipped vector | 0.8571 | 0.8810 | 0.9286 |
| shipped BM25 | 0.8571 | 0.9048 | 0.9048 |
| 0.6B hybrid | 0.8333 | 0.8810 | 0.9286 |

**0.9762 (41/42) is the new ceiling**, but unlike the original 0.9310 ceiling it is not a shared wall.
Two first stages reach it at depth 100 and they miss different questions: bge-m3 hybrid misses the line
voltage question, shipped hybrid misses the per-unit base power question, and each retrieves the other's
miss. No question in the corrected set is missed by every arm at every depth, which the original set had
one of. The combined ceiling across arms is therefore 42/42, and the gap to it is an ordering problem
rather than a coverage one.

The hardest single question is *"What happens when you try to connect a line between nodes with different
rated voltages?"* (`component/line.hpp`, checked to be genuinely single-source: no other file in the corpus
explains this validation rule), missed by 20 of the 22 first-stage arms. It is not unanswerable: the
shipped model's d100 vector and hybrid arms both retrieve it, at ranks 62 and 71, just consistently
missed by every other arm's top 10.

| reranker | first stage | hit | MRR | baseline hit | baseline MRR |
|---|---|---|---|---|---|
| none | shipped hybrid d100 | 0.8810 | **0.7667** | — | — |
| none | shipped BM25 d100 | 0.8571 | 0.5830 | — | — |
| none | bge-m3 hybrid d50 | 0.8333 | 0.7280 | — | — |
| bge-reranker-v2-m3 | bge-m3 hybrid d50 | **0.9048** | 0.7280 | 0.8333 | 0.7280 |
| bge-reranker-v2-m3 | 0.6B hybrid d100 | **0.9048** | 0.7643 | 0.8333 | 0.6616 |
| bge-reranker-v2-m3 | 0.6B hybrid d50 | 0.8571 | 0.7292 | 0.8333 | 0.6616 |
| bge-reranker-v2-m3 | 0.6B vector d50 | 0.8333 | 0.7016 | 0.8095 | 0.6271 |
| bge-reranker-v2-m3 | shipped BM25 d100 | 0.8571 | 0.7294 | 0.8571 | 0.5830 |
| bge-reranker-v2-m3 | shipped hybrid d100 | 0.8571 | 0.7068 | 0.8810 | 0.7667 |
| qwen3.5:9b listwise | 0.6B hybrid d50 | 0.8810 | 0.6946 | 0.8333 | 0.6616 |
| ms-marco-MiniLM-L6-v2 | bge-m3 hybrid d50 | 0.8571 | 0.7009 | 0.8333 | 0.7280 |
| ms-marco-MiniLM-L6-v2 | shipped BM25 d100 | 0.8095 | 0.6683 | 0.8571 | 0.5830 |
| ms-marco-MiniLM-L6-v2 | shipped hybrid d100 | 0.8571 | 0.6892 | 0.8810 | 0.7667 |

The `none` rows are the unranked arms that were actually measured, not a no-reranker baseline for
every first stage in the table; the `baseline` columns carry that per row.

`bge-reranker-v2-m3` remains the strongest reranker measured and the only one that improves both
metrics on most inputs. It gains hit rate on four of the six lists it was given, leaves one unchanged,
and loses it only on the shipped hybrid d100 list, which is also the only list whose MRR it degrades
(0.7667 to 0.7068). That list was already the best-ordered input in the table. The bge-m3 d50 row is
the interesting one: it recovers a question (0.8333 to 0.9048) at exactly unchanged MRR, which is the
reranker pulling something up from deep in the list without reordering what was already near the top.

Two conclusions moved. **The best MRR in this table is now no reranker at all** (shipped hybrid
d100, 0.7667), narrowly ahead of `bge-reranker-v2-m3` on the 0.6B d100 list at 0.7643. **The LLM
listwise reranker is no longer close to the best**: it produces 0.6946, tenth of the thirteen arms in
this table, for the same unusable latency profile as before. Its original selling point does not survive the corrected
ground truth. Per-query latency was not re-measured, since nothing about the reranking cost changed,
only what it is scored against.

The cheap cross-encoder is confirmed unreliable rather than uniformly bad. It improves the bge-m3
hybrid arm's hit rate (0.8333 to 0.8571) and lifts the poorly ordered BM25 list's MRR (0.5830 to
0.6683), while degrading every other arm on at least one metric and losing to `bge-reranker-v2-m3` on
every input it shares with it. "Worse than no reranker" is better read as "unreliable, and still not
worth it on this corpus" than as "always worse."

### An ensemble of the two strongest embedders is bounded out without building one

The original measurement left one question open: bge-m3 and Qwen3-Embedding-0.6B tied at 0.7931 while
failing on different questions, which is the shape of an ensemble opportunity rather than of
equivalence, and no combination was ever measured. It can be answered from what is already on disk.

A combined first stage draws its output from the union of its components' candidate lists, and
truncating that union to an output depth can only drop documents from it. The recall of the union at
matched input depth is therefore a ceiling on any fusion of those components, whether by rank, by
score, or by anything else. Every arm's retrieved order is saved per question, so that ceiling is
computable without retrieving anything.

| arm | bge-m3 | 0.6B | union ceiling | headroom over best component |
|---|---|---|---|---|
| vector d10 | 37/42 | 34/42 | 38/42 | 1 |
| hybrid d10 | 36/42 | 35/42 | 37/42 | 1 |
| vector d50 | 39/42 | 37/42 | 39/42 | **0** |
| hybrid d50 | 39/42 | 37/42 | 39/42 | **0** |
| hybrid d100 | 41/42 | 39/42 | 41/42 | **0** |
| vector d10 (N=29, historical) | 23/29 | 23/29 | 24/29 | 1 |
| hybrid d50 (N=29, historical) | 26/29 | 26/29 | 26/29 | **0** |
| hybrid d100 (N=29, historical) | 27/29 | 27/29 | 27/29 | **0** |

**The union column is an upper bound on a combination nobody built, not a measured arm.** A real
fusion can fall short of these numbers and cannot exceed them. Nothing in this table was retrieved,
embedded, or reranked for it.

**At depth 50 and beyond, the 0.6B retrieves nothing bge-m3 misses**, on both test sets. The best case
anywhere is one question at depth 10, which is 2.38 points against a floor of 3 questions, so even a
fusion that reached its ceiling exactly would produce nothing this document would report as a ranking.
The result does not rest on the corrected ground truth either: the headroom was already zero at depth
50 and 100 on the original 29-question set, before the correction removed the tie that motivated the
question.

Which question supplies the headroom moved, while the headroom itself did not. At both depth-10 arms
on the current set bge-m3 is the single best component, and the union gains *"How do you create input
data for a power-grid-model calculation?"* over it, the question that no arm at any depth could
retrieve on the original set. On the original set the two models tie, so there is no single best
component to attribute the gain to: the union gains the C++ standard question over bge-m3 and the
short circuit fault types question over the 0.6B, one question either way. The C++ standard question
is the one the source-pattern audit later reclassified as an honest miss for both models.

Two limits on this. Both components were measured at 12346 chunks, while the shipped arms have since
been re-measured at 19637; a component and its union move together under a re-chunk, which is a reason
to expect the headroom to be the stable quantity rather than a measurement that it is. And there is no
0.6B vector d100 arm at the corrected set, so that one cell is absent rather than computed. Filling it
means a full index build of roughly 22 minutes, and the hybrid d100 row already covers that depth at
zero headroom, so the build was declined.

**Recommendation: do not ensemble these two embedders.** Judged against bge-m3, its own best component
rather than against the shipped model, a fusion is bounded at one question at depth 10 and at exactly
nothing at the depths a reranker actually consumes. No cost figure is needed to reach that: an
ensemble means two models resident, two index builds and two query encodings, and the quality column
it would have to justify is empty. This is a result about these two models on this corpus. A model
less correlated with bge-m3 than the 0.6B is could have headroom against it, and nothing here measures
that.

### Recommendations, restated

**Embedder: nothing measured here justifies a migration.** The original case was a 4-question hit-rate
win. At the corrected ground truth there is no win: bge-m3 is 1 question ahead of the shipped model,
0.6B is 2 behind, and the shipped model has the best MRR of the four. Adopting an embedder forces a
full re-ingestion of every existing collection, and a 1-question margin inside the resolution floor is
not a reason to ask users for that. If a migration is wanted for other reasons, bge-m3 is the one to
take: it is the only modern embedder within a question of the shipped model on both metrics, and it
builds its index roughly three times faster than 0.6B. Do not go above 1B regardless. Both claims in
this paragraph were measured at chunk size 1000; the sweep at the top of this document re-tested them
at four sizes and the conclusion held, with bge-m3 a tie at best and 0.6B 3 questions behind.

The honest next step before any embedder decision is a larger test set. Every pairwise difference in
that table except bge-m3 over 0.6B is 2 questions or fewer, and this set has now reversed its own
headline result under correction. Combining the two strongest embedders instead of choosing between
them is not the way out: it is bounded at zero questions of gain at the depths that matter, as the
ensemble section above shows.

**Depth: the case for 100 over 50 is stronger than it was.** The original recommendation of 50 rested
on depth 100 buying exactly one more question. At N=42 it buys 2 on three of the five first stages and
3 on the shipped hybrid arm, which is the first depth difference in this work wide enough to call a
ranking. Depth 50 is still the better default once rerank latency is priced in, since the reranker cost
scales with the input list, but "depth 100 buys nothing" is no longer an accurate summary of the table.

**Reranker and score-threshold recommendations below are unchanged** from the original measurement:
they were never resting on the embedder margin that just reversed, and the figures above confirm the
same shape (bge-reranker-v2-m3 is the strongest reranker; no score threshold is safe for a newly
adopted embedder). See those sections below for the original reasoning, which still applies.

## Read this before reading any table (original measurement, N=29, historical, see the update above)

**29 questions means one question is 3.45 points of hit rate.** Differences smaller than roughly 7
points are two questions or fewer and are reported here as not distinguishable, not as rankings. Four
decimal places appear below because that is what the harness emits, not because the third decimal
means anything.

MRR is the more informative metric in this test set. It is not quantized to whole questions the way
hit rate is, and several of the conclusions below rest on MRR moving consistently while hit rate sits
still.

## Embedder (original measurement, N=29, historical, see the update above)

All arms: vector retrieval, depth 10, unthresholded, fp16, one corpus, one metric, **chunk size
1000**.

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
because that tie is not the whole story. Both the tie and the ensemble opportunity it suggested have
since been answered; see the ensemble section in the update above.

## Candidate depth (original measurement, N=29, historical, see the update above)

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

## Reranker (original measurement, N=29, historical, see the update above)

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

## Per-question flips against the shipped configuration (original measurement, N=29, historical, see the update above)

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

**Since measured, and the answer is no.** The ceiling for a combination is above either component by
exactly one question at depth 10 and by nothing at all at depth 50 and 100, on this set and on the
corrected one. The two questions named just above are exactly what the combination would have gained
here, one over each model, and the source-pattern audit later reclassified the C++ standard one as a
miss for both. See the ensemble section in the update above.

## Recommendations per slot (embedder bullet superseded, see the update above; rest still current)

These are inputs to an adoption decision, not the decision. Adopting a new embedder forces a full
re-ingestion of every existing collection, which is a user-visible migration.

**Embedder: superseded.** This bullet originally read "Qwen3-Embedding-0.6B, or bge-m3 ... either is
worth 4 questions over the shipped model." At the corrected ground truth 0.6B is 2 questions behind
the shipped model and bge-m3 is 1 ahead; see "Recommendations, restated" in the update above. Do not go
above 1B still holds.

**Depth: 50, on a narrower argument than this bullet originally made.** It read "depth 100 buys one
more question for double the rerank latency, which is inside the resolution limit." The first half no
longer holds at N=42, where depth 100 buys 2 to 3 questions; see the depth bullet in the update above.
The latency argument is what carries the recommendation now. Depth 10 is too shallow and was leaving
retrieved answers on the floor.

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

**Not re-measured against the widened or grown test set.** This section is stale relative to the
current 42-question set, and left to whoever wants the current numbers: this section takes tens of
minutes per model where the retrieval sections above rescore in seconds.

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

- `Qwen3-Embedding-4B`: 16x the index build cost of bge-m3 and the worst MRR of any embedder measured.
  At N=42 it is no longer last on hit rate, but nothing about that makes the build cost worth paying.
  Do not retry.
- `cross-encoder/ms-marco-MiniLM-L6-v2`: unreliable on this corpus. It degrades most inputs, though
  not uniformly all of them at N=42, and loses to `bge-reranker-v2-m3` on every input it was given.
  Not worth reaching for.
- `qwen3.5:9b` listwise: tenth of thirteen reranker arms on MRR at N=42 (0.6946 against a best of
  0.7667), unusable latency profile. Its original near-best MRR did not survive the corrected ground
  truth.
- Depth 5 on the shipped stack: 0.5862 hit (N=29 measurement), materially worse than depth 10 and
  included only to confirm the direction; re-measured at N=42 as 0.8333, still the shallowest and
  worst depth-5 option, direction unchanged.

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

**Test set sources were narrow; this is now addressed, not eliminated.** The update above widened
ground truth and grew the set from 29 to 42 scored questions. Any future question added to this set
should be checked the same way: read what the corpus actually says before trusting a single source.

**Source patterns are substrings of the full path, and two remain deliberately ambiguous.** Every
pattern in the test set was audited against the corpus and scoped until it matched exactly one file,
with two exceptions that substring matching cannot express. `power-grid-model/LICENSE` also matches
`LICENSES/MPL-2.0.txt`, which is the actual licence text and a correct source for that question.
`Power Flow Example.ipynb` also matches the `.ipynb.license` SPDX sidecar sitting next to it. Neither
can be excluded without anchoring the end of a path, which the matcher does not do, and neither can
credit a file that answers a different question. Any new pattern should be checked the same way:
match it against the corpus file list and require exactly one file, or understand why not.

**One embedder's index build time is missing.** The shipped model reused an existing collection rather
than rebuilding, so its build cost is not comparable to the others. Index build times were not
re-measured for the N=42 update: retrieval itself did not change, only what it is scored against, so
re-timing an unchanged build would have added cost without adding information.

## Reproducing

```
python evals/bench_retrieval.py --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --retriever hybrid --depth 50 --dtype float16 --max-seq-length 768

python evals/rerank_bench.py \
  --candidates evals/bench_candidates/<arm>.json \
  --reranker BAAI/bge-reranker-v2-m3
```

Per-arm JSON is in `evals/bench_results/` and reranker output in `evals/rerank_results/`, both
tracked. Each arm record carries `testset_size` and `testset_hash` for the test set it was scored
against, because both directories hold arms from more than one test set under names that encode the
model and depth but nothing about the questions. Compare two arms only when those two fields agree.
Records written before this field existed carry neither, and are the N=29 historical runs.

Arm records also carry `chunk_size`, `chunk_overlap` and `chunk_max_seq_length`, read from the corpus
they scored rather than from the command line, and the chunk size appears in the arm name. The same
reading rule applies and for the same reason: an arm written before this field existed carries `null`
for all three, and `null` is not evidence that it matches an arm that records a value. Two arms of one
model at two chunk sizes used to be identical in every saved field, so the second overwrote the first,
which is how a tracked result was lost while producing the re-measurement at the top of this document.
`fusion_bound.py` now gates on chunking as it does on the test set, and refuses an arm that records it
against one that does not.

The frozen candidate lists are written to `evals/bench_candidates/`, which is not tracked:
they run to 24MB because each one carries the full text of every candidate chunk, and re-running the
first stage regenerates them. Rescoring a reranker arm without re-embedding needs those lists, so
regenerate them before reaching for `rerank_bench.py`.

Large embedders need `--batch-size 16` and `--dtype float16`. The 4B build for the N=42 update tracked
roughly 100 minutes at those settings on the measurement machine, slower than the 122.7-minute figure
above but consistent with it; treat this as a floor, not a guarantee. Swap usage climbed to within
~1.5GB of exhaustion partway through and did not degrade further, but a machine under more concurrent
load than this one was under should expect it to be closer.

**Bounding a combination of arms, without building one:** `evals/fusion_bound.py` scores the union of
two or more saved arms at matched depth, through the same metric code as every other number here, and
prints each component's recall, the union's, and the headroom between them. It refuses arms scored
against different test sets, at different candidate depths, over different repositories or under
different ground truth, and it refuses the same arm given twice: `bench_results/` holds arms from more
than one test set under filenames that say nothing about the questions. Chunking it cannot refuse.
Arm records do not save the chunk configuration their index was built with, and two arms of the same
model over different chunkings agree in every field that is saved, so the report prints that gate as
missing and matching it is the caller's job. When two components tie for best, both are named with
the questions the union gains over each, since the headroom is the same number against either and the
questions supplying it are not. This is what produced the ensemble table above:

```
python evals/fusion_bound.py \
  baai-bge-m3_vector_d10_float16_seq768 \
  qwen-qwen3-embedding-0-6b_vector_d10_float16_seq768
```

**Rescoring only, when just `testset.json` changed and retrieval did not:** `evals/rescore_testset.py`
recomputes hit rate, MRR, recall, and category breakdown for every saved arm in `evals/bench_results/`
and `evals/rerank_results/` directly from their recorded retrieved-document order, without querying
Qdrant or a model. It only produces correct numbers for questions that already have a saved retrieval
result; a newly added question has no saved order to rescore from, and needs a real
`bench_retrieval.py` run before it means anything.

By default it only reports a before/after summary. `--write` folds the recomputed numbers back into
the arm files, and is what keeps the tables above in step with the JSON they came from after a
`testset.json` edit. It writes only to arms that cover every scored question in the current test set,
so the N=29 arms kept as historical record are left as the runs they were:

```
python evals/rescore_testset.py --write
```

**`run_eval.py` declares the repositories it retrieves from.** That is the other harness in this
directory, the one that scores answers rather than retrieval alone. Both its arms are restricted to a
declared repository scope, defaulting to whatever the shipped test set is written against, currently
`power-grid-model` alone, and overridden with `--repos a,b` or `EVAL_REPOS`. The vector arm applies it
as a Qdrant payload filter on `repo`, so asking for `k` documents gets `k` in-scope ones rather than
whatever survives discarding. The BM25 arm loads only the named repositories' corpus files: BM25 scores
depend on corpus-wide document frequencies and average document length, so filtering a merged corpus
scores differently from never merging it. A scope naming a repository missing from either the
collection or the corpus directory stops the run before the first question, and every report records
the scope it ran under.

Read anything that harness published earlier with that in mind. Those figures were measured against
whatever happened to sit in the shared collection and corpus directory at the time, and that state is
not recoverable now. One run collapsed toward zero Hit Rate on every arm because an unrelated
repository had been ingested alongside the corpus; working out why took an investigation instead of a
startup error. `bench_retrieval.py` was never exposed to this. It reads a corpus directory built for
the arm and records `ingested_repositories` in the arm record, so its scope is fixed by construction.
