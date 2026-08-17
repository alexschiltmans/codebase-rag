# Evaluation Results

**Date:** 2026-08-17 03:46

**Test set:** 43 questions

**Retrieval stages:** rerank (`BAAI/bge-reranker-v2-m3`, depth 50), rewrite (timeout 5.0s)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 5.71s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5152 |
| avg_source_precision | 0.3767 |
| avg_hit_rate | 0.7857 |
| avg_mrr | 0.6663 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 3.9446 |
| p95_latency_s | 5.0921 |
| avg_prompt_tokens | 798.6977 |
| avg_ttft_s | 3.2574 |
| p95_ttft_s | 4.2056 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5589 | 43/43 |
| answer_relevancy | 0.8578 | 43/43 |
| context_recall | 0.5883 | 42/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 821 | 3.5s | 3.8s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 789 | 3.7s | 4.4s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 554 | 3.1s | 3.6s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.57 | 5 | 832 | 3.8s | 4.1s | False |
| 5 | medium | factual_lookup | 1 | 0.20 | 0.80 | 5 | 876 | 2.7s | 3.2s | False |
| 6 | medium | how_does_it_work | 1 | 0.33 | 0.50 | 5 | 748 | 4.5s | 5.5s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 855 | 2.5s | 2.9s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 662 | 2.4s | 2.6s | False |
| 9 | medium | factual_lookup | 1 | 0.20 | 0.25 | 5 | 813 | 4.2s | 4.3s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.50 | 5 | 787 | 4.0s | 4.8s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 713 | 2.7s | 3.8s | False |
| 12 | hard | cross_file_reasoning | 0 | 0.00 | 0.00 | 5 | 636 | 3.5s | 3.8s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 579 | 3.5s | 3.7s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 801 | 2.7s | 3.1s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 583 | 3.5s | 3.7s | False |
| 16 | easy | factual_lookup | - | - | 0.67 | 5 | 516 | 2.5s | 2.7s | True |
| 17 | hard | conceptual | 1 | 0.50 | 0.20 | 5 | 863 | 3.7s | 4.7s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 946 | 2.1s | 2.3s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 821 | 4.2s | 5.1s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 934 | 3.6s | 4.4s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 782 | 3.7s | 4.6s | False |
| 22 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 886 | 3.5s | 4.9s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.20 | 5 | 831 | 3.5s | 4.5s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.60 | 5 | 951 | 3.5s | 4.0s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 886 | 3.4s | 3.9s | False |
| 26 | medium | conceptual | 0 | 0.00 | 0.33 | 5 | 948 | 3.6s | 4.2s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 738 | 3.7s | 4.9s | False |
| 28 | easy | conceptual | 1 | 0.50 | 0.50 | 5 | 805 | 2.4s | 3.0s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.00 | 5 | 903 | 3.9s | 5.0s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 880 | 3.3s | 4.3s | False |
| 31 | easy | factual_lookup | 1 | 0.50 | 0.67 | 5 | 813 | 2.7s | 2.9s | False |
| 32 | easy | factual_lookup | 0 | 0.00 | 0.50 | 5 | 645 | 3.0s | 3.5s | False |
| 33 | medium | factual_lookup | 1 | 0.25 | 0.20 | 5 | 718 | 2.4s | 3.6s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 714 | 2.5s | 3.1s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 905 | 3.5s | 4.7s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 669 | 3.5s | 4.0s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 960 | 4.0s | 4.4s | False |
| 38 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 841 | 2.4s | 3.5s | False |
| 39 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 862 | 2.2s | 2.6s | False |
| 40 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 847 | 2.6s | 3.7s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 959 | 4.4s | 5.5s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 0.80 | 5 | 866 | 3.5s | 4.7s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.80 | 5 | 806 | 2.6s | 3.7s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for this package is 3.12, as specified in Document 5. The Python package uses CMake (version 3.12 or higher) to build. This can be verified by checking the `requirements.txt` file under `dependencies`, which lists `numpy>=2.0.0`.

The build system used is CMake, as indicated in Document 1 and explicitly stated in the context provided.
