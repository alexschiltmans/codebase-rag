# Evaluation Results

**Date:** 2026-08-17 04:29

**Test set:** 43 questions

**Retrieval stages:** rerank (`BAAI/bge-reranker-v2-m3`, depth 50), rewrite (timeout 5.0s)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 7.18s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5828 |
| avg_source_precision | 0.4000 |
| avg_hit_rate | 0.8095 |
| avg_mrr | 0.7238 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 3.9710 |
| p95_latency_s | 4.9028 |
| avg_prompt_tokens | 815.6279 |
| avg_ttft_s | 3.2627 |
| p95_ttft_s | 4.3541 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5550 | 43/43 |
| answer_relevancy | 0.8315 | 43/43 |
| context_recall | 0.6756 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 858 | 4.5s | 4.8s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 824 | 3.7s | 4.3s | False |
| 3 | easy | factual_lookup | 1 | 0.50 | 0.50 | 5 | 642 | 2.9s | 3.3s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.43 | 5 | 612 | 2.6s | 3.1s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 866 | 2.9s | 3.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.50 | 5 | 866 | 4.5s | 5.5s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 860 | 2.9s | 3.7s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 762 | 2.3s | 2.5s | False |
| 9 | medium | factual_lookup | 1 | 0.50 | 0.00 | 5 | 654 | 3.4s | 3.5s | False |
| 10 | medium | factual_lookup | 1 | 0.20 | 0.75 | 5 | 904 | 4.0s | 4.7s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 821 | 3.3s | 4.6s | False |
| 12 | hard | cross_file_reasoning | 0 | 0.00 | 0.00 | 5 | 715 | 3.6s | 4.6s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 656 | 4.4s | 4.5s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.50 | 5 | 794 | 2.4s | 2.9s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 673 | 3.5s | 3.7s | False |
| 16 | easy | factual_lookup | - | - | 0.67 | 5 | 612 | 2.7s | 3.0s | True |
| 17 | hard | conceptual | 1 | 0.20 | 0.20 | 5 | 813 | 3.1s | 4.2s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.40 | 5 | 949 | 3.4s | 4.0s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 877 | 2.8s | 4.0s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 891 | 2.8s | 3.5s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 824 | 3.4s | 4.6s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 756 | 3.5s | 4.9s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 917 | 3.7s | 4.4s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 946 | 3.4s | 3.6s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 901 | 3.5s | 4.1s | False |
| 26 | medium | conceptual | 1 | 1.00 | 0.33 | 5 | 870 | 2.7s | 2.9s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 844 | 3.2s | 4.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.50 | 5 | 837 | 2.5s | 3.3s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.25 | 5 | 776 | 2.8s | 4.0s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 873 | 2.4s | 3.4s | False |
| 31 | easy | factual_lookup | 1 | 1.00 | 0.67 | 5 | 806 | 3.5s | 3.7s | False |
| 32 | easy | factual_lookup | 0 | 0.00 | 0.50 | 5 | 810 | 2.6s | 2.8s | False |
| 33 | medium | factual_lookup | 1 | 0.50 | 0.20 | 5 | 744 | 2.2s | 3.2s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 797 | 2.5s | 3.0s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.00 | 5 | 905 | 3.6s | 4.6s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 752 | 4.2s | 4.9s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 923 | 3.7s | 4.3s | False |
| 38 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 904 | 3.8s | 5.0s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 877 | 3.9s | 4.4s | False |
| 40 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 827 | 2.8s | 3.6s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 940 | 3.7s | 4.7s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 861 | 3.6s | 4.6s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 1.00 | 5 | 733 | 3.5s | 4.5s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for this package is `>=3.12`. The Python package uses CMake as its build system, which is compatible with Windows (specifically Microsoft's Windows) and Linux (POSIX).

Here's a breakdown of the requirements:

- **Minimum Python Version:** 3.12
- **Build System:** CMake

This ensures compatibility and proper compilation for the specified operating systems and algorithms used in the Power Grid Model package.
