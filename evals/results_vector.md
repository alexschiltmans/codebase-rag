# Evaluation Results

**Date:** 2026-07-21 09:03

**Test set:** 16 questions

**Latency probe:** 0.98s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4074 |
| avg_source_precision | 0.1875 |
| avg_hit_rate | 0.4000 |
| avg_mrr | 0.2967 |
| questions_answered | 16 |
| questions_failed | 0 |
| avg_latency_s | 0.7544 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5415 | 16/16 |
| answer_relevancy | 0.7972 | 16/16 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 0.20 | 1.00 | 5 | 0.6s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.8s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.7s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.29 | 5 | 0.5s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.50 | 5 | 1.4s | False |
| 7 | medium | cross_file_reasoning | 0 | 0.00 | 0.75 | 5 | 0.8s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.4s | False |
| 9 | medium | factual_lookup | 1 | 0.25 | 0.50 | 5 | 0.6s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.75 | 5 | 0.6s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.40 | 5 | 1.4s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 1.3s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 0.3s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.25 | 5 | 0.8s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 1.00 | 5 | 0.9s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 0.5s | True |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is **3.4**. The Python package uses the CMake project with the `python` executable as its build tool, which is compatible with this version of Python.

Here's a breakdown based on the context provided:

- **Minimum Python Version:** 3.4
- **Build System:** CMake project using the `python` executable

This ensures that the package can be built and run correctly with the specified Python version, leveraging the CMake build system for dependency management and compilation.
