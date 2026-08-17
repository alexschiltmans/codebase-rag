# Evaluation Results

**Date:** 2026-08-17 11:58

**Test set:** 43 questions

**Retrieval stages:** none (base retriever only)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 1.59s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5050 |
| avg_source_precision | 0.3953 |
| avg_hit_rate | 0.8333 |
| avg_mrr | 0.7044 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 0.9215 |
| p95_latency_s | 1.5536 |
| avg_prompt_tokens | 797.2791 |
| avg_ttft_s | 0.1903 |
| p95_ttft_s | 0.2162 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.4750 | 43/43 |
| answer_relevancy | 0.8332 | 43/43 |
| context_recall | 0.4791 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 887 | 0.2s | 0.6s | False |
| 2 | easy | factual_lookup | 1 | 0.25 | 0.00 | 5 | 726 | 0.2s | 0.4s | False |
| 3 | easy | factual_lookup | 1 | 0.33 | 0.00 | 5 | 863 | 0.2s | 0.6s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.71 | 5 | 756 | 0.2s | 1.0s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 799 | 0.2s | 0.4s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 797 | 0.2s | 1.6s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.75 | 5 | 659 | 0.2s | 0.4s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 811 | 0.2s | 0.4s | False |
| 9 | medium | factual_lookup | 1 | 0.50 | 0.25 | 5 | 637 | 0.2s | 0.6s | False |
| 10 | medium | factual_lookup | 1 | 0.25 | 0.75 | 5 | 803 | 0.2s | 0.7s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 823 | 0.2s | 1.3s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 685 | 0.2s | 0.9s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 527 | 0.2s | 0.3s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 837 | 0.2s | 1.1s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 638 | 0.2s | 1.2s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 849 | 0.2s | 0.5s | True |
| 17 | hard | conceptual | 1 | 0.25 | 0.20 | 5 | 880 | 0.2s | 1.5s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 946 | 0.2s | 0.5s | False |
| 19 | medium | conceptual | 0 | 0.00 | 0.25 | 5 | 779 | 0.2s | 1.3s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 857 | 0.2s | 1.2s | False |
| 21 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 702 | 0.2s | 0.5s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 772 | 0.2s | 1.6s | False |
| 23 | hard | conceptual | 1 | 0.50 | 0.00 | 5 | 836 | 0.2s | 0.9s | False |
| 24 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 808 | 0.2s | 0.8s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 916 | 0.2s | 1.0s | False |
| 26 | medium | conceptual | 1 | 0.50 | 0.33 | 5 | 764 | 0.2s | 0.4s | False |
| 27 | medium | conceptual | 1 | 0.50 | 0.50 | 5 | 845 | 0.2s | 1.6s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 541 | 0.2s | 0.9s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.00 | 5 | 925 | 0.2s | 1.4s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 864 | 0.2s | 1.1s | False |
| 31 | easy | factual_lookup | 1 | 1.00 | 0.67 | 5 | 723 | 0.2s | 1.1s | False |
| 32 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 691 | 0.2s | 0.3s | False |
| 33 | medium | factual_lookup | 1 | 1.00 | 0.40 | 5 | 726 | 0.2s | 1.2s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 856 | 0.2s | 0.8s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 896 | 0.2s | 1.1s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 762 | 0.2s | 1.0s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 967 | 0.2s | 0.5s | False |
| 38 | hard | conceptual | 0 | 0.00 | 0.60 | 5 | 819 | 0.2s | 1.4s | False |
| 39 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 867 | 0.2s | 1.3s | False |
| 40 | hard | conceptual | 1 | 1.00 | 0.25 | 5 | 959 | 0.2s | 0.8s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 865 | 0.2s | 0.8s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 0.80 | 5 | 820 | 0.2s | 1.3s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.60 | 5 | 800 | 0.2s | 1.5s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for this Python package is **3.6**. The package uses the `pytest` build system, which is a build tool for Python that supports unit testing and integration tests. It's noted to be compatible with BSD-3 license software.

Here's a quick summary based on the context:

- **Python Version Requirement:** 3.6
- **Build System Used:** `pytest`

This ensures compatibility with the latest version of `pytest`, which is actively maintained and supports various testing scenarios, including unit tests, integration tests, and more.
