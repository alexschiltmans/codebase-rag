# Evaluation Results

**Date:** 2026-08-16 23:24

**Test set:** 43 questions

**Retrieval stages:** rerank (`BAAI/bge-reranker-v2-m3`, depth 50)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 6.13s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5604 |
| avg_source_precision | 0.4047 |
| avg_hit_rate | 0.8095 |
| avg_mrr | 0.6841 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 2.2847 |
| p95_latency_s | 3.0834 |
| avg_prompt_tokens | 836.4186 |
| avg_ttft_s | 1.5370 |
| p95_ttft_s | 2.0384 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5949 | 43/43 |
| answer_relevancy | 0.8522 | 43/43 |
| context_recall | 0.6372 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 910 | 1.2s | 1.4s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 831 | 1.4s | 2.1s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 832 | 1.5s | 1.9s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.71 | 5 | 747 | 1.5s | 2.0s | False |
| 5 | medium | factual_lookup | 1 | 0.20 | 0.80 | 5 | 852 | 1.4s | 1.7s | False |
| 6 | medium | how_does_it_work | 1 | 0.33 | 0.75 | 5 | 878 | 1.2s | 2.6s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 819 | 1.4s | 1.6s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 825 | 1.1s | 1.4s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.75 | 5 | 691 | 1.6s | 1.8s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 723 | 1.5s | 2.1s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.40 | 5 | 745 | 1.0s | 2.0s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 684 | 1.6s | 2.8s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 561 | 1.2s | 1.4s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 879 | 1.3s | 1.7s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 629 | 1.6s | 2.4s | False |
| 16 | easy | factual_lookup | - | - | 0.67 | 5 | 801 | 1.5s | 1.9s | True |
| 17 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 933 | 1.1s | 2.0s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 965 | 1.5s | 1.8s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 872 | 1.5s | 2.7s | False |
| 20 | medium | conceptual | 1 | 0.50 | 1.00 | 5 | 951 | 1.7s | 3.1s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 777 | 2.1s | 2.5s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.25 | 5 | 840 | 2.1s | 3.5s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.60 | 5 | 805 | 1.7s | 2.0s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 951 | 1.6s | 2.0s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 933 | 1.5s | 1.8s | False |
| 26 | medium | conceptual | 1 | 0.20 | 0.33 | 5 | 882 | 1.7s | 2.3s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 818 | 1.5s | 2.5s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.50 | 5 | 821 | 1.6s | 3.0s | False |
| 29 | medium | conceptual | 1 | 0.25 | 0.25 | 5 | 889 | 1.6s | 2.9s | False |
| 30 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 873 | 1.5s | 2.5s | False |
| 31 | easy | factual_lookup | 1 | 0.25 | 0.67 | 5 | 831 | 1.5s | 2.0s | False |
| 32 | easy | factual_lookup | 0 | 0.00 | 1.00 | 5 | 797 | 1.8s | 2.1s | False |
| 33 | medium | factual_lookup | 1 | 0.50 | 0.20 | 5 | 771 | 1.2s | 2.3s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 939 | 1.7s | 2.4s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 838 | 1.5s | 2.6s | False |
| 36 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 762 | 1.4s | 2.2s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 962 | 1.5s | 2.6s | False |
| 38 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 903 | 1.5s | 2.6s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 866 | 1.6s | 2.5s | False |
| 40 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 960 | 1.7s | 2.7s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 959 | 1.8s | 2.6s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 861 | 1.8s | 3.1s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.60 | 5 | 800 | 2.0s | 3.2s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for this C++ library to be built is **3.12** using the `pip` command with the `--build-essential` option, which includes the necessary dependencies. The Python package uses the **CMake** build system. Specifically, it utilizes a dynamic build target defined in the provided documentation as follows:

```cmake
requires-python = ">=3.12"
dependencies = ["numpy>=2.0.0"]
dynamic = ["version"]
```

This indicates that CMake is used to manage dependencies and build configurations, with `version` specifying the minimum Python version required for successful compilation.
