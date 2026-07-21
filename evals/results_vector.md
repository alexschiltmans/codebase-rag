# Evaluation Results

**Date:** 2026-07-21 14:32

**Test set:** 30 questions

**Latency probe:** 1.00s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4298 |
| avg_source_precision | 0.2800 |
| avg_hit_rate | 0.6207 |
| avg_mrr | 0.5270 |
| questions_answered | 30 |
| questions_failed | 0 |
| avg_latency_s | 0.9074 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| answer_relevancy | 0.8100 | 30/30 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 0.20 | 1.00 | 5 | 0.5s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.6s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.6s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.43 | 5 | 0.6s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 1.2s | False |
| 7 | medium | cross_file_reasoning | 0 | 0.00 | 0.75 | 5 | 1.0s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.5s | False |
| 9 | medium | factual_lookup | 1 | 0.25 | 0.75 | 5 | 0.6s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.75 | 5 | 0.7s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 1.1s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 0.9s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 0.2s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.25 | 5 | 0.9s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 1.00 | 5 | 1.5s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 0.6s | True |
| 17 | hard | conceptual | 1 | 0.50 | 0.20 | 5 | 1.2s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 0.8s | False |
| 19 | medium | conceptual | 0 | 0.00 | 0.25 | 5 | 1.3s | False |
| 20 | medium | conceptual | 1 | 0.33 | 1.00 | 5 | 1.1s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 1.6s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.25 | 5 | 1.6s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.60 | 5 | 0.9s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.60 | 5 | 0.4s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 0.6s | False |
| 26 | medium | conceptual | 0 | 0.00 | 0.33 | 5 | 0.7s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 1.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 1.0s | False |
| 29 | medium | conceptual | 1 | 1.00 | 0.00 | 5 | 1.4s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 1.1s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is **3.4**. The Python package uses the CMake project by default, which supports both CMake and Visual Studio Build Tools (VS). However, it also allows using Miniforge as an alternative if BSD license compliance is a concern.

Here's a summary of the build system requirements:

- **Minimum Python Version:** 3.4
- **Build System:** CMake with or without VS (Visual Studio Build Tools)

This setup ensures that the package can be built using standard CMake commands, while also providing flexibility through Miniforge for those preferring a more lightweight build process.
