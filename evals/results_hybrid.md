# Evaluation Results

**Date:** 2026-07-18 23:14

**Test set:** 16 questions

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.3749 |
| avg_source_precision | 0.1750 |
| questions_answered | 16 |
| questions_failed | 0 |
| avg_latency_s | 0.8227 |

## RAGAS Scores

| Metric | Score |
|--------|-------|
| faithfulness | 1.0000 |
| context_recall | 0.8750 |

## Per-Question Breakdown

| # | Difficulty | Category | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1.00 | 5 | 0.8s | False |
| 2 | easy | factual_lookup | 0.00 | 5 | 0.8s | False |
| 3 | easy | factual_lookup | 0.00 | 5 | 0.6s | False |
| 4 | medium | cross_file_reasoning | 0.71 | 5 | 0.8s | False |
| 5 | medium | factual_lookup | 0.00 | 5 | 0.4s | False |
| 6 | medium | how_does_it_work | 0.50 | 5 | 1.2s | False |
| 7 | medium | cross_file_reasoning | 0.75 | 5 | 1.2s | False |
| 8 | hard | factual_lookup | 0.00 | 5 | 1.0s | False |
| 9 | medium | factual_lookup | 0.00 | 5 | 0.7s | False |
| 10 | medium | factual_lookup | 0.50 | 5 | 0.6s | False |
| 11 | hard | how_does_it_work | 0.20 | 5 | 1.1s | False |
| 12 | hard | cross_file_reasoning | 0.25 | 5 | 1.0s | False |
| 13 | easy | factual_lookup | 0.50 | 5 | 0.3s | False |
| 14 | medium | factual_lookup | 0.25 | 5 | 1.0s | False |
| 15 | hard | cross_file_reasoning | 1.00 | 5 | 1.2s | False |
| 16 | easy | factual_lookup | 0.33 | 5 | 0.5s | True |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is **3.4**. The Python package uses the CMake project's default Python, which is licensed under the BSD license. This means it can be built using any compatible build system that supports CMake and Python 3.

Here's a quick summary based on the context:

- **Minimum Python Version:** Python 3.4
- **Build System:** CMake project (with BSD license)
