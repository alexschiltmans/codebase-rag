# Evaluation Results

**Date:** 2026-08-17 05:14

**Test set:** 43 questions

**Retrieval stages:** rerank (`BAAI/bge-reranker-v2-m3`, depth 50), rewrite (timeout 5.0s)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 7.48s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5410 |
| avg_source_precision | 0.3535 |
| avg_hit_rate | 0.7619 |
| avg_mrr | 0.6964 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 3.9820 |
| p95_latency_s | 5.1088 |
| avg_prompt_tokens | 814.0930 |
| avg_ttft_s | 3.2515 |
| p95_ttft_s | 4.4085 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.6146 | 43/43 |
| answer_relevancy | 0.8176 | 43/43 |
| context_recall | 0.5996 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 0.50 | 1.00 | 5 | 821 | 2.0s | 2.3s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 789 | 4.0s | 4.2s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 420 | 3.2s | 3.6s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.43 | 5 | 696 | 2.6s | 3.5s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.60 | 5 | 855 | 2.7s | 3.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 1.00 | 5 | 835 | 4.7s | 6.1s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 855 | 2.1s | 2.7s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 836 | 3.6s | 4.1s | False |
| 9 | medium | factual_lookup | 1 | 0.50 | 0.00 | 5 | 785 | 2.5s | 2.7s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 912 | 4.1s | 4.7s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 778 | 3.0s | 4.4s | False |
| 12 | hard | cross_file_reasoning | 0 | 0.00 | 0.00 | 5 | 627 | 3.6s | 4.0s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 579 | 4.5s | 4.6s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.50 | 5 | 856 | 3.2s | 4.1s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 675 | 3.5s | 4.4s | False |
| 16 | easy | factual_lookup | - | - | 0.67 | 5 | 627 | 2.7s | 3.1s | True |
| 17 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 857 | 3.0s | 3.8s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 946 | 3.4s | 3.7s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 865 | 4.1s | 4.6s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 893 | 3.1s | 4.3s | False |
| 21 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 755 | 3.8s | 4.8s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 871 | 3.4s | 4.7s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 816 | 4.3s | 4.5s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.60 | 5 | 951 | 3.5s | 3.7s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 890 | 3.7s | 4.5s | False |
| 26 | medium | conceptual | 1 | 0.25 | 0.33 | 5 | 891 | 2.8s | 3.8s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 846 | 2.6s | 4.0s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.50 | 5 | 796 | 2.7s | 3.3s | False |
| 29 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 886 | 2.4s | 3.3s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 880 | 2.5s | 3.6s | False |
| 31 | easy | factual_lookup | 1 | 0.50 | 0.67 | 5 | 674 | 3.1s | 3.2s | False |
| 32 | easy | factual_lookup | 0 | 0.00 | 0.50 | 5 | 797 | 2.9s | 3.4s | False |
| 33 | medium | factual_lookup | 1 | 0.50 | 0.20 | 5 | 756 | 2.2s | 2.7s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 851 | 2.8s | 3.5s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 905 | 3.2s | 4.4s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 752 | 4.3s | 5.1s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 985 | 3.8s | 4.2s | False |
| 38 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 912 | 3.9s | 5.0s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 862 | 2.5s | 3.4s | False |
| 40 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 887 | 2.7s | 3.5s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 942 | 4.4s | 5.5s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 0.80 | 5 | 855 | 3.7s | 4.7s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 1.00 | 5 | 739 | 2.9s | 4.2s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for this package is `>=3.12`. The Python package uses CMake as its build system, as indicated by the `dependencies` section in the context information.

Here's a summary of the details:
- **Minimum Python Version:** `>=3.12`
- **Build System:** CMake
- **Additional Notes:**
  - The package is compatible with Python 2.x (as indicated by the `--python=<version>` flag).
  - It requires `numpy` version `>=2.0.0`.

This ensures that the code can be compiled and run on a Windows operating system using CMake, which is commonly used for building Python packages in a cross-platform manner.
