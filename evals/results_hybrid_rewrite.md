# Evaluation Results

**Date:** 2026-08-17 03:02

**Test set:** 43 questions

**Retrieval stages:** rewrite (timeout 5.0s)

**Repositories retrieved from:** `power-grid-model` (retrieval was restricted to these; anything else in the index or the corpus directory was not searched)

**Latency probe:** 2.08s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe; a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.5511 |
| avg_source_precision | 0.3581 |
| avg_hit_rate | 0.7857 |
| avg_mrr | 0.7163 |
| questions_answered | 43 |
| questions_failed | 0 |
| avg_latency_s | 1.9299 |
| p95_latency_s | 2.9764 |
| avg_prompt_tokens | 808.7209 |
| avg_ttft_s | 1.2121 |
| p95_ttft_s | 1.6816 |
| efficiency_questions | 43 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5990 | 43/43 |
| answer_relevancy | 0.8619 | 43/43 |
| context_recall | 0.5854 | 43/43 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Prompt Tokens | TTFT | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 743 | 1.6s | 2.0s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 810 | 1.6s | 2.2s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 746 | 0.6s | 1.0s | False |
| 4 | medium | cross_file_reasoning | 1 | 1.00 | 0.71 | 5 | 804 | 0.7s | 1.4s | False |
| 5 | medium | factual_lookup | 1 | 0.33 | 0.60 | 5 | 876 | 0.7s | 1.3s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 892 | 1.7s | 2.3s | False |
| 7 | medium | cross_file_reasoning | 1 | 1.00 | 0.50 | 5 | 757 | 0.6s | 1.2s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 847 | 1.7s | 2.0s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 670 | 1.6s | 1.9s | False |
| 10 | medium | factual_lookup | 1 | 0.25 | 0.75 | 5 | 878 | 1.6s | 2.1s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 713 | 1.6s | 2.7s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 822 | 1.1s | 1.6s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 527 | 1.7s | 2.1s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 938 | 1.2s | 2.0s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 1.00 | 5 | 638 | 1.6s | 3.0s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 824 | 1.0s | 1.2s | True |
| 17 | hard | conceptual | 1 | 0.50 | 0.40 | 5 | 865 | 0.7s | 1.7s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 942 | 1.6s | 2.1s | False |
| 19 | medium | conceptual | 0 | 0.00 | 0.25 | 5 | 829 | 1.6s | 2.7s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 923 | 0.6s | 1.8s | False |
| 21 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 692 | 1.6s | 3.0s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 837 | 1.6s | 3.0s | False |
| 23 | hard | conceptual | 1 | 0.50 | 0.40 | 5 | 732 | 1.7s | 2.2s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 867 | 1.6s | 1.8s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 919 | 1.7s | 1.9s | False |
| 26 | medium | conceptual | 0 | 0.00 | 0.33 | 5 | 719 | 0.5s | 0.7s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 854 | 0.8s | 2.2s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 666 | 0.6s | 1.2s | False |
| 29 | medium | conceptual | 1 | 0.50 | 0.25 | 5 | 888 | 0.4s | 1.2s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 880 | 1.6s | 2.6s | False |
| 31 | easy | factual_lookup | 1 | 1.00 | 0.67 | 5 | 850 | 1.6s | 1.9s | False |
| 32 | easy | factual_lookup | 1 | 1.00 | 1.00 | 5 | 740 | 0.9s | 1.2s | False |
| 33 | medium | factual_lookup | 1 | 1.00 | 0.40 | 5 | 727 | 0.6s | 1.6s | False |
| 34 | easy | factual_lookup | 1 | 1.00 | 0.75 | 5 | 849 | 0.6s | 1.1s | False |
| 35 | medium | conceptual | 1 | 1.00 | 0.00 | 5 | 896 | 0.8s | 1.7s | False |
| 36 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 762 | 1.6s | 2.2s | False |
| 37 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 842 | 1.6s | 2.2s | False |
| 38 | hard | conceptual | 1 | 1.00 | 1.00 | 5 | 763 | 1.0s | 1.8s | False |
| 39 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 798 | 0.4s | 1.5s | False |
| 40 | hard | conceptual | 1 | 1.00 | 0.75 | 5 | 915 | 0.9s | 1.7s | False |
| 41 | hard | cross_file_reasoning | 1 | 1.00 | 1.00 | 5 | 878 | 1.7s | 2.9s | False |
| 42 | hard | cross_file_reasoning | 1 | 1.00 | 0.80 | 5 | 865 | 1.2s | 2.3s | False |
| 43 | hard | how_does_it_work | 1 | 1.00 | 0.60 | 5 | 792 | 1.7s | 2.8s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is **3.6**. The Python package uses the **CMake build system**, which is automatically downloaded and utilized during the build process when using `uv sync` to source a Python package from source. Manual installation of these dependencies is not necessary unless you are working with standalone CMake builds.
