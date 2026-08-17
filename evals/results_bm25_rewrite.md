# Evaluation Results

**Date:** 2026-08-17 02:18

**Test set:** 43 questions

**Retrieval stages:** rewrite (timeout 5.0s)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 2.02s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5312 |
| avg_source_precision | 0.2977 |
| avg_hit_rate | 0.7381 |
| avg_mrr | 0.5595 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 1.9061 |
| p95_latency_s | 2.9234 |
| avg_prompt_tokens | 808.8837 |
| avg_ttft_s | 1.1733 |
| p95_ttft_s | 1.6139 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5601 | 43/43 |
| answer_relevancy | 0.8167 | 43/43 |
| context_recall | 0.5715 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 0.67 | 5 | 690 | 1.6s | 2.2s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 821 | 1.6s | 2.1s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 818 | 1.0s | 1.4s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.86 | 5 | 885 | 0.6s | 1.3s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.60 | 5 | 925 | 0.6s | 1.4s | False |
| 6 | medium | how_does_it_work | 1 | 1.00 | 0.50 | 5 | 730 | 1.6s | 2.9s | False |
| 7 | medium | cross_file_reasoning | 1 | 0.25 | 0.75 | 5 | 950 | 0.6s | 0.9s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 934 | 0.5s | 1.1s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 875 | 1.6s | 2.0s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.75 | 5 | 851 | 1.6s | 1.9s | False |
| 11 | hard | how_does_it_work | 1 | 0.50 | 0.20 | 5 | 775 | 1.6s | 2.7s | False |
| 12 | hard | cross_file_reasoning | 0 | 0.00 | 0.25 | 5 | 801 | 1.6s | 1.8s | False |
| 13 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 651 | 1.6s | 1.8s | False |
| 14 | medium | factual_lookup | 1 | 1.00 | 0.75 | 5 | 631 | 0.8s | 1.2s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 723 | 1.6s | 2.6s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 523 | 0.4s | 0.8s | True |
| 17 | hard | conceptual | 0 | 0.00 | 0.20 | 5 | 809 | 1.6s | 2.4s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.40 | 5 | 827 | 1.6s | 2.6s | False |
| 19 | medium | conceptual | 1 | 0.33 | 0.50 | 5 | 819 | 1.6s | 1.9s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 879 | 1.6s | 2.7s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 871 | 1.6s | 1.8s | False |
| 22 | hard | conceptual | 1 | 0.50 | 1.00 | 5 | 827 | 1.6s | 3.0s | False |
| 23 | hard | conceptual | 1 | 0.33 | 0.00 | 5 | 815 | 1.6s | 2.1s | False |
| 24 | hard | conceptual | 1 | 0.33 | 0.60 | 5 | 813 | 1.6s | 1.8s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 895 | 1.6s | 1.9s | False |
| 26 | medium | conceptual | 1 | 0.25 | 0.67 | 5 | 768 | 0.5s | 1.3s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 866 | 1.3s | 2.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.50 | 5 | 805 | 0.6s | 1.6s | False |
| 29 | medium | conceptual | 1 | 1.00 | 0.00 | 5 | 645 | 1.6s | 2.3s | False |
| 30 | hard | conceptual | 1 | 0.33 | 0.50 | 5 | 766 | 1.6s | 2.9s | False |
| 31 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 775 | 1.5s | 1.7s | False |
| 32 | easy | factual_lookup | 0 | 0.00 | 1.00 | 5 | 844 | 0.4s | 1.1s | False |
| 33 | medium | factual_lookup | 1 | 0.50 | 0.40 | 5 | 818 | 0.9s | 2.2s | False |
| 34 | easy | factual_lookup | 1 | 0.50 | 0.75 | 5 | 862 | 0.5s | 1.0s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 876 | 1.1s | 2.4s | False |
| 36 | medium | conceptual | 1 | 0.33 | 0.75 | 5 | 760 | 0.4s | 1.4s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 828 | 1.6s | 1.9s | False |
| 38 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 798 | 1.2s | 2.1s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 869 | 0.6s | 1.0s | False |
| 40 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 943 | 0.8s | 1.6s | False |
| 41 | hard | cross_file_reasoning | 1 | 0.33 | 1.00 | 5 | 875 | 1.6s | 2.5s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 0.40 | 5 | 805 | 0.7s | 1.9s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.80 | 5 | 741 | 0.6s | 2.0s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for this package is **Python 3.6** or higher. The package uses CMake as its build system, which is compatible with Python 3.6.

Here's a breakdown of the relevant sections from the context:

- **Minimum Python Version:** `pip install power-grid-model | 3.6`
- **Build System:** `#### C++
The C++ dependencies are **build-time only**, meaning they are automatically downloaded and used during the build process when using CMake for building the package from source (via `uv sync`). Manual installation is not required for these builds.

This setup ensures that the package can be easily integrated into a Python environment while leveraging CMake's dependency management capabilities.
