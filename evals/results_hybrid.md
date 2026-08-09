# Evaluation Results

**Date:** 2026-08-07 23:17

**Test set:** 43 questions

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 1.44s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4691 |
| avg_source_precision | 0.3953 |
| avg_hit_rate | 0.8333 |
| avg_mrr | 0.7044 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 0.9172 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5018 | 43/43 |
| answer_relevancy | 0.8391 | 43/43 |
| context_recall | 0.4791 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 0.7s | False |
| 2 | easy | factual_lookup | 1 | 0.25 | 0.00 | 5 | 0.5s | False |
| 3 | easy | factual_lookup | 1 | 0.33 | 0.00 | 5 | 0.5s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.57 | 5 | 0.9s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 1.6s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 0.5s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.8s | False |
| 9 | medium | factual_lookup | 1 | 0.50 | 0.00 | 5 | 0.5s | False |
| 10 | medium | factual_lookup | 1 | 0.25 | 0.50 | 5 | 0.6s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 1.3s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 0.9s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 0.4s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.8s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 0.7s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 0.5s | True |
| 17 | hard | conceptual | 1 | 0.25 | 0.20 | 5 | 1.5s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 0.8s | False |
| 19 | medium | conceptual | 0 | 0.00 | 0.25 | 5 | 1.2s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 0.8s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 0.4s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 1.6s | False |
| 23 | hard | conceptual | 1 | 0.50 | 0.00 | 5 | 0.8s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 0.4s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 1.0s | False |
| 26 | medium | conceptual | 1 | 0.50 | 0.33 | 5 | 0.4s | False |
| 27 | medium | conceptual | 1 | 0.50 | 0.50 | 5 | 1.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 1.0s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.00 | 5 | 1.1s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 1.3s | False |
| 31 | easy | factual_lookup | 1 | 1.00 | 0.67 | 5 | 1.2s | False |
| 32 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 0.5s | False |
| 33 | medium | factual_lookup | 1 | 1.00 | 0.00 | 5 | 1.4s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 0.7s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.00 | 5 | 1.2s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 1.0s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 1.1s | False |
| 38 | hard | conceptual | 0 | 0.00 | 0.60 | 5 | 1.2s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 1.4s | False |
| 40 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 0.9s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 0.9s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 1.3s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.60 | 5 | 1.5s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is **3.6**. The package uses the `CMake` build system, which is automatically downloaded and utilized during the build process when using `uv sync`.

Here's a breakdown of the relevant information from the context:

- **Python Version Requirement:** 3.6
- **Build System Used:** CMake

This setup ensures that the dependencies are correctly installed and used during the Python package build process, as specified in Document 1.
