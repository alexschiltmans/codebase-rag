# Evaluation Results

**Date:** 2026-07-21 21:27

**Test set:** 30 questions

**Latency probe:** 1.16s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5371 |
| avg_source_precision | 0.2333 |
| avg_hit_rate | 0.6552 |
| avg_mrr | 0.4534 |
| questions_answered | 30 |
| questions_failed | 0 |
| avg_latency_s | 0.8771 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.6246 | 30/30 |
| answer_relevancy | 0.8371 | 30/30 |
| context_recall | 0.5787 | 30/30 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0 | 0.00 | 0.67 | 5 | 1.0s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.3s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.7s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.71 | 5 | 1.0s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 0.9s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.50 | 5 | 1.3s | False |
| 7 | medium | cross_file_reasoning | 1 | 0.20 | 0.75 | 5 | 0.5s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.5s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.5s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 1.00 | 5 | 0.5s | False |
| 11 | hard | how_does_it_work | 1 | 0.33 | 0.00 | 5 | 1.4s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.00 | 5 | 1.2s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 0.2s | False |
| 14 | medium | factual_lookup | 1 | 0.33 | 0.75 | 5 | 0.6s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 1.0s | False |
| 16 | easy | factual_lookup | - | - | 1.00 | 5 | 0.7s | True |
| 17 | hard | conceptual | 0 | 0.00 | 0.20 | 5 | 1.0s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.40 | 5 | 0.6s | False |
| 19 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 1.0s | False |
| 20 | medium | conceptual | 1 | 0.50 | 0.80 | 5 | 0.9s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 0.4s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 1.5s | False |
| 23 | hard | conceptual | 1 | 0.33 | 0.20 | 5 | 0.4s | False |
| 24 | hard | conceptual | 1 | 0.25 | 1.00 | 5 | 0.7s | False |
| 25 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 0.9s | False |
| 26 | medium | conceptual | 1 | 0.50 | 0.67 | 5 | 1.1s | False |
| 27 | medium | conceptual | 1 | 0.20 | 0.50 | 5 | 1.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 1.6s | False |
| 29 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 1.2s | False |
| 30 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 1.3s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the `power-grid-model` Python package is **3.12**. The build system used is **CMake with scikit-build-core**, as indicated by the `requirements` section in Document 1 and the `build-system` section in Document 2.

Here's a breakdown:

*   **Python Version:** C++11 or later (specifically version >=3.12) is required.
*   **Build System:** The package utilizes CMake as its build system, specifically the `scikit_build_core.build` sub-branch for building Python packages. This ensures that all dependencies are correctly installed and configured during the build process.

This setup allows for a clean and modular build process where CMake handles the dependency resolution and compilation of the necessary libraries, while scikit-build-core manages the specific C++ build tasks.
