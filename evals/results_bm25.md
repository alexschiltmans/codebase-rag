# Evaluation Results

**Date:** 2026-07-18 23:01

**Test set:** 16 questions

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4302 |
| avg_source_precision | 0.1750 |
| questions_answered | 16 |
| questions_failed | 0 |
| avg_latency_s | 0.8367 |

## RAGAS Scores

| Metric | Score |
|--------|-------|
| faithfulness | 1.0000 |
| answer_relevancy | 0.0000 |
| context_recall | 0.7500 |

## Per-Question Breakdown

| # | Difficulty | Category | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0.67 | 5 | 1.0s | False |
| 2 | easy | factual_lookup | 0.00 | 5 | 0.7s | False |
| 3 | easy | factual_lookup | 0.00 | 5 | 0.8s | False |
| 4 | medium | cross_file_reasoning | 1.00 | 5 | 1.1s | False |
| 5 | medium | factual_lookup | 0.80 | 5 | 1.1s | False |
| 6 | medium | how_does_it_work | 0.25 | 5 | 1.6s | False |
| 7 | medium | cross_file_reasoning | 0.75 | 5 | 1.0s | False |
| 8 | hard | factual_lookup | 0.00 | 5 | 0.7s | False |
| 9 | medium | factual_lookup | 0.00 | 5 | 0.4s | False |
| 10 | medium | factual_lookup | 0.50 | 5 | 0.4s | False |
| 11 | hard | how_does_it_work | 0.00 | 5 | 1.4s | False |
| 12 | hard | cross_file_reasoning | 0.00 | 5 | 0.8s | False |
| 13 | easy | factual_lookup | 0.50 | 5 | 0.2s | False |
| 14 | medium | factual_lookup | 0.75 | 5 | 0.5s | False |
| 15 | hard | cross_file_reasoning | 0.67 | 5 | 0.9s | False |
| 16 | easy | factual_lookup | 1.00 | 5 | 0.6s | True |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the `power-grid-model` Python package is **3.12**. The build system used is **CMake with scikit-build-core**, as indicated by the `requirements` section in Document 1 and the `build-system` section in Document 2.

Here's a breakdown:

*   **Python Version:** C++11 or later (specifically version >=3.12) is required.
*   **Build System:** The package utilizes CMake as its build system, specifically the `scikit_build_core.build` backend. This allows for flexible and modular building processes using CMake.

This setup ensures that the dependencies are downloaded automatically during the build process when running `uv sync`.
