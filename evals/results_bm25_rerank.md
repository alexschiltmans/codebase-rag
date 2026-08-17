# Evaluation Results

**Date:** 2026-08-17 00:09

**Test set:** 43 questions

**Retrieval stages:** rerank (`BAAI/bge-reranker-v2-m3`, depth 50)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 6.01s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5847 |
| avg_source_precision | 0.4093 |
| avg_hit_rate | 0.7857 |
| avg_mrr | 0.6833 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 2.2108 |
| p95_latency_s | 2.8207 |
| avg_prompt_tokens | 840.6977 |
| avg_ttft_s | 1.5052 |
| p95_ttft_s | 1.7448 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5928 | 43/43 |
| answer_relevancy | 0.8504 | 43/43 |
| context_recall | 0.6306 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 905 | 1.4s | 1.6s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 817 | 1.7s | 1.8s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 814 | 1.4s | 1.7s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.86 | 5 | 774 | 1.4s | 2.0s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 938 | 1.6s | 2.3s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.50 | 5 | 861 | 1.4s | 2.7s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 846 | 1.5s | 1.8s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 888 | 1.5s | 1.7s | False |
| 9 | medium | factual_lookup | 1 | 0.25 | 0.00 | 5 | 637 | 1.4s | 1.7s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 1.00 | 5 | 811 | 1.4s | 2.3s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 798 | 1.4s | 2.4s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 654 | 1.7s | 3.0s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 659 | 1.2s | 1.3s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 891 | 1.4s | 1.9s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 666 | 1.5s | 2.3s | False |
| 16 | easy | factual_lookup | - | - | 0.67 | 5 | 793 | 1.6s | 1.9s | True |
| 17 | hard | conceptual | 0 | 0.00 | 0.20 | 5 | 772 | 1.3s | 2.4s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 949 | 1.6s | 1.9s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 895 | 1.3s | 1.5s | False |
| 20 | medium | conceptual | 1 | 0.50 | 1.00 | 5 | 891 | 1.4s | 2.3s | False |
| 21 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 783 | 1.4s | 1.8s | False |
| 22 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 837 | 1.6s | 2.8s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 897 | 1.5s | 2.1s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 951 | 1.5s | 2.0s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 932 | 1.8s | 2.4s | False |
| 26 | medium | conceptual | 1 | 1.00 | 0.67 | 5 | 882 | 1.4s | 1.6s | False |
| 27 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 830 | 1.3s | 2.4s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 826 | 1.4s | 2.6s | False |
| 29 | medium | conceptual | 1 | 0.25 | 0.25 | 5 | 889 | 1.6s | 2.5s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 880 | 1.6s | 2.7s | False |
| 31 | easy | factual_lookup | 1 | 0.50 | 0.33 | 5 | 767 | 1.7s | 1.7s | False |
| 32 | easy | factual_lookup | 1 | 0.20 | 1.00 | 5 | 872 | 1.6s | 2.2s | False |
| 33 | medium | factual_lookup | 1 | 0.50 | 0.20 | 5 | 771 | 1.3s | 2.4s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 922 | 1.7s | 2.5s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 838 | 1.6s | 2.6s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 818 | 1.5s | 2.5s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 965 | 1.5s | 2.0s | False |
| 38 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 855 | 1.5s | 2.2s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 860 | 1.4s | 2.1s | False |
| 40 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 963 | 1.7s | 2.6s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 959 | 1.8s | 2.8s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 861 | 1.6s | 2.7s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.80 | 5 | 733 | 1.7s | 2.8s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the `power-grid-model` Python package is **3.12**. The Python package uses the CMake build system, specifically targeting a dynamic library format with a stable pure C API/ABI that can be used by any application.

Here's a breakdown:

*   **Python Version Requirement:** 3.12
*   **Build System:** CMake (specifically, `build-python` target)

This setup allows for building the library from source using `uv sync`, which automatically downloads and uses the dependencies during the build process without needing manual installation.
