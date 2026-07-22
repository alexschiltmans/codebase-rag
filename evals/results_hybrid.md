# Evaluation Results

**Date:** 2026-07-21 21:58

**Test set:** 30 questions

**Latency probe:** 1.35s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4206 |
| avg_source_precision | 0.2600 |
| avg_hit_rate | 0.5862 |
| avg_mrr | 0.5115 |
| questions_answered | 30 |
| questions_failed | 0 |
| avg_latency_s | 0.9732 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5297 | 30/30 |
| answer_relevancy | 0.7805 | 30/30 |
| context_recall | 0.5186 | 30/30 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0 | 0.00 | 1.00 | 5 | 0.8s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 1.1s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.6s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 1.00 | 5 | 1.3s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.50 | 5 | 1.3s | False |
| 7 | medium | cross_file_reasoning | 0 | 0.00 | 0.75 | 5 | 0.8s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.8s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.9s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.50 | 5 | 0.6s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 1.3s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.00 | 5 | 0.4s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.00 | 5 | 0.3s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.25 | 5 | 1.0s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 1.00 | 5 | 1.2s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 0.5s | True |
| 17 | hard | conceptual | 1 | 0.33 | 0.00 | 5 | 0.7s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 0.7s | False |
| 19 | medium | conceptual | 1 | 0.50 | 0.75 | 5 | 1.6s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 1.2s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 1.3s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.25 | 5 | 1.6s | False |
| 23 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 0.9s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 1.4s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 0.7s | False |
| 26 | medium | conceptual | 0 | 0.00 | 0.33 | 5 | 0.4s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 1.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.50 | 5 | 0.8s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.00 | 5 | 1.6s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 1.1s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is **3.4**. The Python package uses CMake as its build system, which is a build tool that generates build files from source code. This allows for cross-platform builds and supports various build configurations like standalone or CMake builds.

Here's a quick summary:

- **Minimum Python Version:** 3.4
- **Build System:** CMake

This setup ensures compatibility with different operating systems and platforms, while also providing flexibility in the build process through CMake.
