# Evaluation Results

**Date:** 2026-08-17 00:51

**Test set:** 43 questions

**Retrieval stages:** rerank (`BAAI/bge-reranker-v2-m3`, depth 50)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 4.56s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5269 |
| avg_source_precision | 0.4093 |
| avg_hit_rate | 0.8095 |
| avg_mrr | 0.7032 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 2.3240 |
| p95_latency_s | 3.1398 |
| avg_prompt_tokens | 839.7209 |
| avg_ttft_s | 1.5746 |
| p95_ttft_s | 2.1506 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.6149 | 43/43 |
| answer_relevancy | 0.8321 | 43/43 |
| context_recall | 0.6605 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 910 | 1.2s | 1.5s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 831 | 1.4s | 2.3s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 838 | 1.5s | 1.8s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.57 | 5 | 747 | 1.5s | 2.1s | False |
| 5 | medium | factual_lookup | 1 | 0.20 | 0.80 | 5 | 852 | 1.4s | 1.7s | False |
| 6 | medium | how_does_it_work | 1 | 0.33 | 0.75 | 5 | 878 | 1.4s | 2.8s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 852 | 1.4s | 1.9s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 823 | 1.1s | 1.7s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 691 | 1.6s | 1.7s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.25 | 5 | 723 | 1.5s | 2.2s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 821 | 1.2s | 2.3s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 646 | 1.6s | 1.9s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 562 | 1.2s | 1.2s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 879 | 1.3s | 1.8s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 629 | 1.6s | 1.8s | False |
| 16 | easy | factual_lookup | - | - | 0.67 | 5 | 793 | 1.6s | 1.9s | True |
| 17 | hard | conceptual | 1 | 1.00 | 0.00 | 5 | 933 | 1.4s | 2.5s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 965 | 1.5s | 2.3s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 938 | 1.5s | 2.2s | False |
| 20 | medium | conceptual | 1 | 0.50 | 1.00 | 5 | 951 | 1.7s | 3.0s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 777 | 2.2s | 3.1s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 840 | 2.2s | 3.5s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 817 | 1.7s | 2.3s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.60 | 5 | 951 | 1.6s | 1.8s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 933 | 1.6s | 1.8s | False |
| 26 | medium | conceptual | 1 | 1.00 | 0.33 | 5 | 835 | 1.7s | 2.3s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 852 | 1.6s | 2.9s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 826 | 1.6s | 2.6s | False |
| 29 | medium | conceptual | 1 | 0.25 | 0.00 | 5 | 889 | 1.7s | 2.7s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 880 | 1.5s | 2.5s | False |
| 31 | easy | factual_lookup | 1 | 0.25 | 0.67 | 5 | 831 | 1.5s | 2.1s | False |
| 32 | easy | factual_lookup | 0 | 0.00 | 1.00 | 5 | 797 | 1.8s | 2.1s | False |
| 33 | medium | factual_lookup | 1 | 0.50 | 0.20 | 5 | 771 | 1.4s | 2.4s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 919 | 1.7s | 2.5s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 838 | 1.5s | 2.9s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 818 | 1.4s | 2.6s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 962 | 1.5s | 2.6s | False |
| 38 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 931 | 1.6s | 2.6s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 866 | 1.6s | 2.3s | False |
| 40 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 960 | 1.7s | 2.6s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 959 | 1.9s | 2.4s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 861 | 1.7s | 2.8s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.80 | 5 | 733 | 2.2s | 3.5s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the `power-grid-model` Python package is **3.12**. The package uses the CMake build system, specifically targeting a dynamic library format with a stable pure C API/ABI. This means it can be compiled and linked against C++ libraries using CMake.

Here's a breakdown:

*   **Python Version:** 3.12
*   **Build System:** CMake (specifically the `build-python` target)

This setup allows for easy integration of Python code with C++ calculations, leveraging CMake's build automation capabilities.
