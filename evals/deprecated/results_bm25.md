# Evaluation Results

> **Deprecated: not reproducible.** These numbers were measured against a corpus of `power-grid-model` and
> `power-grid-model-ds`. Evaluation has since moved to a single repository
> (`power-grid-model` alone).
> Kept as a record of what was measured and when. Do not compare these figures against
> newer numbers or cite them as current.
>
> Current retrieval measurements are in [retrieval-stack-findings.md](../retrieval-stack-findings.md).

**Date:** 2026-08-03 15:54

**Test set:** 30 questions

**Latency probe:** 1.12s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5086 |
| avg_source_precision | 0.2333 |
| avg_hit_rate | 0.6552 |
| avg_mrr | 0.4362 |
| questions_answered | 30 |
| questions_failed | 0 |
| avg_latency_s | 0.9013 |

## RAGAS Scores (judge: `sam860/LFM2:350m`)

> ⚠️ **Self-judged.** No `--judge-model`/`RAGAS_JUDGE_MODEL` was set, so the same model that generated these answers (`sam860/LFM2:350m`) also scored them. This adds self-preference bias, and a model this size is a weak judge to begin with — treat these numbers as indicative at best. The custom keyword recall / source precision metrics above don't use an LLM judge and are more trustworthy.

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.9722 | 30/30 |
| answer_relevancy | 0.1480 | 30/30 |
| context_recall | 0.7247 | 28/30 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0 | 0.00 | 0.67 | 5 | 0.9s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.7s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.25 | 5 | 1.1s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.86 | 5 | 0.9s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 0.9s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 1.5s | False |
| 7 | medium | cross_file_reasoning | 1 | 0.20 | 0.75 | 5 | 0.6s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.9s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.5s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.50 | 5 | 0.4s | False |
| 11 | hard | how_does_it_work | 1 | 0.33 | 0.20 | 5 | 1.4s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.00 | 5 | 1.0s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 0.2s | False |
| 14 | medium | factual_lookup | 1 | 0.33 | 1.00 | 5 | 0.6s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 0.6s | False |
| 16 | easy | factual_lookup | - | - | 1.00 | 5 | 0.7s | True |
| 17 | hard | conceptual | 0 | 0.00 | 0.00 | 5 | 0.8s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.40 | 5 | 0.9s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 1.1s | False |
| 20 | medium | conceptual | 1 | 0.50 | 1.00 | 5 | 1.2s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 1.2s | False |
| 22 | hard | conceptual | 1 | 0.50 | 0.25 | 5 | 1.5s | False |
| 23 | hard | conceptual | 1 | 0.33 | 0.20 | 5 | 0.8s | False |
| 24 | hard | conceptual | 1 | 0.25 | 0.80 | 5 | 0.5s | False |
| 25 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 0.7s | False |
| 26 | medium | conceptual | 1 | 0.50 | 0.67 | 5 | 0.4s | False |
| 27 | medium | conceptual | 1 | 0.20 | 0.50 | 5 | 1.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 1.2s | False |
| 29 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 0.9s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 1.2s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the `power-grid-model` Python package is **3.12**. The build system used by this package is **CMake**, as indicated in the `dependencies` section of the context information.

Here's a breakdown:

*   **Python Version Requirement:** 3.12
    *   This is specified directly under the `requirements` section for both the source and build systems.
*   **Build System:** CMake
    *   The `build-system` section lists CMake as the primary build system, which installs the necessary tools (like `scikit-build-core`, `pgm-build-deps`, etc.) to compile the Python package.

Therefore, you need a version of Python 3.12 or higher to successfully build this C++ library for distribution power system analysis using CMake.
