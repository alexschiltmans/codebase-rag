# Widening rescore, set held at 30 questions

**Purpose:** isolate the effect of widening `sources` from the effect of growing the set. Every arm
below is rescored from its already-saved retrieval output against the widened `testset.json`, still at
30 questions (29 scored). Retrieval itself was not re-run: this is the same retrieved order, rescored.

Produced by `evals/rescore_testset.py` at the point the test set held 30 questions. It is not
reproducible from the current `testset.json`, which holds 43; the numbers here are a record of that
intermediate state, kept because the whole point is to separate the two effects.

**This snapshot was taken before the source patterns were scoped, and some of the gains it records are
false credits.** The widening step left several patterns loose enough to match files that do not answer
their question, most consequentially a bare `CMakeLists.txt` matching 16 files in this corpus. Every
"after" number below therefore includes hits that the scoped ground truth does not allow. The
separation of scoring effect from set-growth effect is still valid and is what this file is for; the
magnitudes are an upper bound on the real widening effect, not a measurement of it. `evals/retrieval-stack-findings.md`
carries the corrected figures and supersedes every conclusion drawn here.

## Every arm rises

Widening `sources` can only turn a miss into a hit, never the reverse, so every arm's hit rate and
MRR should rise or hold. They all rose. No arm fell. This is the expected signature of a scoring
change, not a retrieval change, and its absence would have meant a bug in the rescoring script.

## Embedder table (vector, depth 10, unthresholded): the published headline comparison

| embedder | hit before | hit after | delta (points) | delta (questions) |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 0.7931 | 0.8966 | +10.3 | +3 |
| BAAI/bge-m3 | 0.7931 | 0.8621 | +6.9 | +2 |
| Qwen3-Embedding-4B | 0.6897 | 0.7931 | +10.3 | +3 |
| all-mpnet-base-v2 (shipped) | 0.6552 | 0.7931 | +13.8 | +4 |

**The shipped model rose the most in absolute terms.** It gained 4 questions against the modern
embedders' 2-3, because it was the model most often penalized by narrow ground truth: content it
retrieved correctly, from a file the old test set didn't list, is what widening now credits it for.

**This compresses, but does not reverse, the published margins.**

- 0.6B vs shipped: 13.8 points before, 10.3 after (still outside the ~7-point resolution floor at N=29).
- **bge-m3 vs shipped: 13.8 points before, 6.9 after.** This margin now sits inside the resolution
  floor the findings themselves define ("differences under roughly 7 points ... are reported as not
  distinguishable"). The claim "both sub-1B modern embedders beat the shipped one by 13.8 points" no
  longer holds for bge-m3 specifically; only 0.6B's margin over shipped survives widening as
  distinguishable.
- 4B vs shipped: 3.45 points before (1 question, already discarded as noise), now exactly tied
  (0.7931 both). Was never a headline claim.
- 0.6B vs bge-m3: exact tie before (0.7931 both). After widening, 0.6B leads by 3.45 points (1
  question), inside the resolution floor, so still not a distinguishable ranking, but no longer a
  literal tie either. The pre-existing MRR-based preference for 0.6B (0.6869 vs 0.6514 after
  rescoring) is unaffected and remains the stronger signal, consistent with how the original findings
  already read this pair.

**Stop condition check:** the ranking by hit rate, 0.6B first, bge-m3 second, 4B and
shipped tied last, did not reverse at this point. But the *size* of one claimed margin (bge-m3 over
shipped) crossed from "outside the resolution floor" to "inside it," which the findings need to state
plainly rather than silently continue repeating the old 13.8-point figure for both modern embedders.

That stop condition was passed on incomplete information. Scoping the source patterns afterwards did
reverse the ranking: at the final test set the shipped model is second on hit rate and first on MRR,
and 0.6B is last. The gains this table credits to the modern embedders were partly the loose
`CMakeLists.txt` pattern. See the update section of `evals/retrieval-stack-findings.md`.

## Candidate depth and reranker tables

Every first-stage arm and every reranker arm rose, by amounts broadly proportional to how much narrow
ground truth had been suppressing it (BM25 and the shipped model, previously the most narrowly scored,
rose the most). No arm's relative order within the depth table or the reranker table changed. The
bge-reranker-v2-m3 arms remain best on hit rate and the listwise LLM reranker remains highest on MRR.
The cheap cross-encoder remains worse than no reranker on MRR on the two dense first stages it was
tested against; on the BM25 list, whose ordering was poor to begin with, it raises MRR from 0.5018 to
0.5471 while dropping hit rate from 0.7931 to 0.7241.

The listwise claim did not survive scoping: at the final test set that reranker is sixth of thirteen
arms on MRR. The rest of this paragraph is unaffected in direction.

`evals/rescore_testset.py` scores against the current test set, so re-running it produces the corrected
figures rather than the 30-question ones above. There is no way to regenerate this table.

## The unanswerable question is now answerable

*"How do you create input data for a power-grid-model calculation?"* was retrieved by nothing at any
depth in the original sweep. With sources widened to `quickstart.md` and `Power Flow Example.ipynb`:

- `sentence-transformers-all-mpnet-base-v2_bm25_d100`: rank 1 is `docs/quickstart.md`.
- `qwen-qwen3-embedding-0-6b_hybrid_d100`: rank 3 is `docs/examples/Power Flow Example.ipynb`.

The retrieved content was there all along; the old ground truth was the reason it scored as a miss
everywhere, exactly the finding that motivated widening the sources in the first place.

## What this does and doesn't show

This isolates the scoring-only effect. It says nothing about whether growing the set moves
these numbers further, and the two effects are measured separately by design so that a reader can
tell how much of any final delta is "the ground truth was wrong" versus "the set is now bigger."
