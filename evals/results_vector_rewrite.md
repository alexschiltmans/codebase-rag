# Evaluation Results

**Date:** 2026-08-17 01:34

**Test set:** 43 questions

**Retrieval stages:** rewrite (timeout 5.0s)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 2.16s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5018 |
| avg_source_precision | 0.3442 |
| avg_hit_rate | 0.7857 |
| avg_mrr | 0.6865 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 1.8457 |
| p95_latency_s | 2.9475 |
| avg_prompt_tokens | 808.0930 |
| avg_ttft_s | 1.1139 |
| p95_ttft_s | 1.6586 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.4450 | 43/43 |
| answer_relevancy | 0.8392 | 43/43 |
| context_recall | 0.4689 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 832 | 1.7s | 2.1s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 855 | 0.8s | 0.9s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 766 | 0.6s | 1.2s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.43 | 5 | 819 | 0.6s | 1.1s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 826 | 0.6s | 0.9s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.50 | 5 | 922 | 1.7s | 3.1s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.50 | 5 | 772 | 0.4s | 0.6s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 836 | 0.5s | 0.6s | False |
| 9 | medium | factual_lookup | 1 | 1.00 | 0.25 | 5 | 619 | 1.6s | 1.9s | False |
| 10 | medium | factual_lookup | 1 | 0.50 | 0.50 | 5 | 823 | 1.7s | 2.4s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 721 | 1.5s | 2.7s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 816 | 0.5s | 1.1s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 632 | 0.8s | 1.0s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 880 | 1.0s | 1.6s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 638 | 1.5s | 2.4s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 824 | 0.6s | 0.9s | True |
| 17 | hard | conceptual | 1 | 0.50 | 0.20 | 5 | 865 | 0.5s | 1.7s | False |
| 18 | medium | conceptual | 1 | 0.50 | 0.40 | 5 | 957 | 1.7s | 1.9s | False |
| 19 | medium | conceptual | 0 | 0.00 | 0.25 | 5 | 829 | 1.6s | 2.3s | False |
| 20 | medium | conceptual | 1 | 1.00 | 0.80 | 5 | 874 | 0.7s | 1.4s | False |
| 21 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 692 | 1.4s | 2.5s | False |
| 22 | hard | conceptual | 1 | 0.50 | 0.50 | 5 | 789 | 1.6s | 2.9s | False |
| 23 | hard | conceptual | 1 | 0.33 | 0.40 | 5 | 733 | 1.6s | 1.9s | False |
| 24 | hard | conceptual | 1 | 0.50 | 0.80 | 5 | 808 | 1.6s | 1.8s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 909 | 1.6s | 2.1s | False |
| 26 | medium | conceptual | 1 | 1.00 | 0.33 | 5 | 846 | 0.5s | 1.1s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 869 | 0.5s | 1.9s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 650 | 0.4s | 1.3s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.25 | 5 | 916 | 1.6s | 3.0s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 873 | 1.6s | 2.4s | False |
| 31 | easy | factual_lookup | 1 | 1.00 | 0.67 | 5 | 850 | 1.6s | 2.1s | False |
| 32 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 733 | 0.7s | 1.0s | False |
| 33 | medium | factual_lookup | 1 | 1.00 | 0.20 | 5 | 710 | 0.5s | 1.8s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 719 | 0.5s | 1.0s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.00 | 5 | 879 | 1.0s | 2.3s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 728 | 1.6s | 2.4s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 903 | 1.6s | 2.7s | False |
| 38 | hard | conceptual | 0 | 0.00 | 0.40 | 5 | 824 | 1.6s | 2.7s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 798 | 1.6s | 2.5s | False |
| 40 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 901 | 0.8s | 1.7s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 928 | 1.6s | 2.6s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 0.80 | 5 | 850 | 1.6s | 2.7s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.40 | 5 | 734 | 0.4s | 1.6s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is 3.64, as indicated in Document 5. The Python package uses the Miniforge build system, which is published under BSD license. This system allows for a lightweight and fast build process suitable for projects that don't require extensive C++ optimizations.

Here's a breakdown of the details:

- **Minimum Python Version:** 3.64
- **Build System:** Miniforge
- **License:** BSD

This choice is recommended because it balances performance with ease of use, making it suitable for projects where you want to minimize overhead while still benefiting from some C++ optimizations.
