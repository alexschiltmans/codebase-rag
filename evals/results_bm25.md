# Evaluation Results

**Date:** 2026-07-31 14:43

**Test set:** 30 questions

**Latency probe:** 0.97s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5217 |
| avg_source_precision | 0.2333 |
| avg_hit_rate | 0.6552 |
| avg_mrr | 0.4362 |
| questions_answered | 30 |
| questions_failed | 0 |
| avg_latency_s | 0.9365 |

## RAGAS Scores (judge: `sam860/LFM2:350m`)

> ⚠️ **Self-judged.** No `--judge-model`/`RAGAS_JUDGE_MODEL` was set, so the same model that generated these answers (`sam860/LFM2:350m`) also scored them. This adds self-preference bias, and a model this size is a weak judge to begin with — treat these numbers as indicative at best. The custom keyword recall / source precision metrics above don't use an LLM judge and are more trustworthy.

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.9770 | 29/30 |
| answer_relevancy | 0.0842 | 30/30 |
| context_recall | 0.7000 | 30/30 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0 | 0.00 | 0.67 | 5 | 1.0s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.9s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.9s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 1.00 | 5 | 0.8s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 1.0s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 1.6s | False |
| 7 | medium | cross_file_reasoning | 1 | 0.20 | 0.75 | 5 | 0.8s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.9s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.5s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.50 | 5 | 0.5s | False |
| 11 | hard | how_does_it_work | 1 | 0.33 | 0.00 | 5 | 1.5s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.00 | 5 | 1.0s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 0.3s | False |
| 14 | medium | factual_lookup | 1 | 0.33 | 1.00 | 5 | 0.7s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 0.8s | False |
| 16 | easy | factual_lookup | - | - | 1.00 | 5 | 0.7s | True |
| 17 | hard | conceptual | 0 | 0.00 | 0.00 | 5 | 0.6s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.40 | 5 | 0.6s | False |
| 19 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 0.5s | False |
| 20 | medium | conceptual | 1 | 0.50 | 1.00 | 5 | 1.5s | False |
| 21 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 1.4s | False |
| 22 | hard | conceptual | 1 | 0.50 | 0.25 | 5 | 1.6s | False |
| 23 | hard | conceptual | 1 | 0.33 | 0.20 | 5 | 0.5s | False |
| 24 | hard | conceptual | 1 | 0.25 | 1.00 | 5 | 0.7s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 0.9s | False |
| 26 | medium | conceptual | 1 | 0.50 | 0.67 | 5 | 0.5s | False |
| 27 | medium | conceptual | 1 | 0.20 | 0.50 | 5 | 1.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 1.4s | False |
| 29 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 1.1s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 1.4s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the `power-grid-model` Python package is **3.12**. The build system used by this package is **CMake**, as indicated in the `dependencies` section of Document 1 and the `build-system` section within Document 2.

Here's a breakdown:

*   **Python Version:** C++17 or later (as specified in `requires = ["scikit-build-core", "pgm-build-deps"]`, which is compatible with CMake builds).
*   **Build System:** `CMake` (specifically, the `build` subdirectory within `build-system`).

This setup ensures that the necessary C++ libraries are automatically downloaded and used during the build process when using `uv sync`.
