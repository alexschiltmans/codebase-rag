# Evaluation Results

**Date:** 2026-07-28 20:56

**Test set:** 30 questions

**Latency probe:** 2.71s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4468 |
| avg_source_precision | 0.2600 |
| avg_hit_rate | 0.5862 |
| avg_mrr | 0.5029 |
| questions_answered | 30 |
| questions_failed | 0 |
| avg_latency_s | 6.0087 |

## RAGAS Scores (judge: `sam860/LFM2:350m`)

> ⚠️ **Self-judged.** No `--judge-model`/`RAGAS_JUDGE_MODEL` was set, so the same model that generated these answers (`sam860/LFM2:350m`) also scored them. This adds self-preference bias, and a model this size is a weak judge to begin with — treat these numbers as indicative at best. The custom keyword recall / source precision metrics above don't use an LLM judge and are more trustworthy.

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.9833 | 30/30 |
| answer_relevancy | 0.1227 | 30/30 |
| context_recall | 0.5517 | 29/30 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0 | 0.00 | 1.00 | 5 | 4.1s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 1.9s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 4.6s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.57 | 5 | 4.3s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.20 | 5 | 5.5s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.75 | 5 | 9.5s | False |
| 7 | medium | cross_file_reasoning | 0 | 0.00 | 0.75 | 5 | 5.8s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 3.9s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.25 | 5 | 8.0s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.50 | 5 | 4.7s | False |
| 11 | hard | how_does_it_work | 1 | 1.00 | 0.20 | 5 | 8.1s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 4.5s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 1.3s | False |
| 14 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 4.8s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 11.9s | False |
| 16 | easy | factual_lookup | - | - | 0.33 | 5 | 1.9s | True |
| 17 | hard | conceptual | 1 | 0.33 | 0.00 | 5 | 7.7s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 3.0s | False |
| 19 | medium | conceptual | 1 | 0.50 | 0.75 | 5 | 5.5s | False |
| 20 | medium | conceptual | 1 | 1.00 | 1.00 | 5 | 7.1s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 11.3s | False |
| 22 | hard | conceptual | 1 | 1.00 | 0.25 | 5 | 10.1s | False |
| 23 | hard | conceptual | 1 | 1.00 | 0.60 | 5 | 10.7s | False |
| 24 | hard | conceptual | 1 | 1.00 | 0.80 | 5 | 3.0s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 4.6s | False |
| 26 | medium | conceptual | 0 | 0.00 | 0.33 | 5 | 4.9s | False |
| 27 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 8.1s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 4.6s | False |
| 29 | medium | conceptual | 1 | 0.25 | 0.25 | 5 | 8.2s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 6.5s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for the Python package is 3.64. The Python package uses the CMake build system, specifically version 3.23 or later.
