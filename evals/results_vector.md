# Evaluation Results

**Date:** 2026-08-16 21:16

**Test set:** 43 questions

**Retrieval stages:** none (base retriever only)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 0.83s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4679 |
| avg_source_precision | 0.3953 |
| avg_hit_rate | 0.8333 |
| avg_mrr | 0.7321 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 0.8936 |
| p95_latency_s | 1.5328 |
| avg_prompt_tokens | 805.3488 |
| avg_ttft_s | 0.1660 |
| p95_ttft_s | 0.1835 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5056 | 43/43 |
| answer_relevancy | 0.8224 | 43/43 |
| context_recall | 0.5328 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 887 | 0.2s | 0.6s | False |
| 2 | easy | factual_lookup | 1 | 0.25 | 0.00 | 5 | 726 | 0.2s | 0.4s | False |
| 3 | easy | factual_lookup | 1 | 0.50 | 0.00 | 5 | 932 | 0.2s | 0.5s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.57 | 5 | 769 | 0.2s | 0.6s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 799 | 0.2s | 0.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 836 | 0.2s | 1.5s | False |
| 7 | medium | cross_file_reasoning | 1 | 0.50 | 0.75 | 5 | 659 | 0.2s | 0.8s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 811 | 0.2s | 0.8s | False |
| 9 | medium | factual_lookup | 1 | 1.00 | 0.25 | 5 | 637 | 0.1s | 0.7s | False |
| 10 | medium | factual_lookup | 1 | 1.00 | 0.50 | 5 | 878 | 0.2s | 0.7s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 823 | 0.2s | 1.4s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 878 | 0.2s | 0.6s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.00 | 5 | 527 | 0.1s | 0.3s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 908 | 0.2s | 0.6s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 708 | 0.2s | 0.4s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 772 | 0.2s | 0.5s | True |
| 17 | hard | conceptual | 1 | 0.50 | 0.20 | 5 | 895 | 0.2s | 1.0s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 886 | 0.2s | 0.4s | False |
| 19 | medium | conceptual | 0 | 0.00 | 0.25 | 5 | 761 | 0.2s | 1.5s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 795 | 0.2s | 0.8s | False |
| 21 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 702 | 0.2s | 1.5s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 781 | 0.2s | 1.5s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 733 | 0.2s | 0.7s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 808 | 0.2s | 0.3s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 916 | 0.2s | 0.4s | False |
| 26 | medium | conceptual | 1 | 0.50 | 0.33 | 5 | 764 | 0.2s | 0.4s | False |
| 27 | medium | conceptual | 1 | 0.50 | 0.50 | 5 | 845 | 0.2s | 1.5s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 650 | 0.2s | 1.0s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.00 | 5 | 925 | 0.2s | 1.3s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 873 | 0.2s | 0.9s | False |
| 31 | easy | factual_lookup | 1 | 1.00 | 0.67 | 5 | 723 | 0.2s | 0.6s | False |
| 32 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 733 | 0.2s | 0.4s | False |
| 33 | medium | factual_lookup | 1 | 1.00 | 0.40 | 5 | 726 | 0.1s | 1.4s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 856 | 0.2s | 1.6s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.00 | 5 | 879 | 0.2s | 1.4s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 762 | 0.2s | 1.0s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 967 | 0.2s | 0.8s | False |
| 38 | hard | conceptual | 0 | 0.00 | 0.40 | 5 | 819 | 0.2s | 1.1s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 867 | 0.2s | 1.3s | False |
| 40 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 945 | 0.2s | 1.2s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 928 | 0.2s | 1.1s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 0.60 | 5 | 807 | 0.2s | 1.2s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.80 | 5 | 734 | 0.2s | 1.3s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the library from source is **Python 3.6**. The Python package uses the `python3` compiler, which requires Python 3.6 or higher to compile successfully. This is specified in the build requirements section under "Build Requirements".

Here's a breakdown of the relevant information:

- **Minimum Python Version:** Python 3.6
- **Build System:** `python3` (or `setuptools` for CMake builds)

This ensures that the compiler used during the build process is compatible with the minimum version of Python required to compile and run the library.
