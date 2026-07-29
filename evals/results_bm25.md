# Evaluation Results

**Date:** 2026-07-28 20:38

**Test set:** 30 questions

**Latency probe:** 4.13s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4802 |
| avg_source_precision | 0.2333 |
| avg_hit_rate | 0.6552 |
| avg_mrr | 0.4362 |
| questions_answered | 30 |
| questions_failed | 0 |
| avg_latency_s | 6.3645 |

## RAGAS Scores (judge: `sam860/LFM2:350m`)

> ⚠️ **Self-judged.** No `--judge-model`/`RAGAS_JUDGE_MODEL` was set, so the same model that generated these answers (`sam860/LFM2:350m`) also scored them. This adds self-preference bias, and a model this size is a weak judge to begin with — treat these numbers as indicative at best. The custom keyword recall / source precision metrics above don't use an LLM judge and are more trustworthy.

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.9033 | 29/30 |
| answer_relevancy | 0.0918 | 30/30 |
| context_recall | 0.7381 | 28/30 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0 | 0.00 | 0.67 | 5 | 8.9s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 3.7s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 4.2s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.57 | 5 | 11.3s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 6.8s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 9.5s | False |
| 7 | medium | cross_file_reasoning | 1 | 0.20 | 0.75 | 5 | 4.3s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 6.8s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 3.1s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 0.75 | 5 | 3.5s | False |
| 11 | hard | how_does_it_work | 1 | 0.33 | 0.00 | 5 | 10.3s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.25 | 5 | 11.0s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 1.1s | False |
| 14 | medium | factual_lookup | 1 | 0.33 | 0.25 | 5 | 4.3s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 10.2s | False |
| 16 | easy | factual_lookup | - | - | 1.00 | 5 | 4.3s | True |
| 17 | hard | conceptual | 0 | 0.00 | 0.20 | 5 | 8.3s | False |
| 18 | medium | conceptual | 1 | 1.00 | 0.20 | 5 | 3.3s | False |
| 19 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 2.9s | False |
| 20 | medium | conceptual | 1 | 0.50 | 0.80 | 5 | 4.9s | False |
| 21 | medium | conceptual | 1 | 1.00 | 0.50 | 5 | 8.4s | False |
| 22 | hard | conceptual | 1 | 0.50 | 0.75 | 5 | 10.6s | False |
| 23 | hard | conceptual | 1 | 0.33 | 0.20 | 5 | 7.0s | False |
| 24 | hard | conceptual | 1 | 0.25 | 0.80 | 5 | 2.4s | False |
| 25 | medium | conceptual | 1 | 1.00 | 0.75 | 5 | 5.5s | False |
| 26 | medium | conceptual | 1 | 0.50 | 1.00 | 5 | 5.0s | False |
| 27 | medium | conceptual | 1 | 0.20 | 0.75 | 5 | 9.8s | False |
| 28 | easy | conceptual | 1 | 1.00 | 0.75 | 5 | 5.2s | False |
| 29 | medium | conceptual | 1 | 1.00 | 0.25 | 5 | 5.5s | False |
| 30 | hard | conceptual | 1 | 1.00 | 0.50 | 5 | 8.8s | False |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for the Python package is 3.12. The build system used is `scikit_build_core.build`.

Here's a breakdown:

- **Minimum Python Version:** The documentation explicitly states that the package requires Python 3.12.
- **Build System:** The package utilizes the `scikit_build_core.build` build system, which is part of the C++ build process for Python packages. This system is designed to automate the build process for C++ libraries, including those built with `scikit-build-core`.

This setup ensures that the C++ dependencies are downloaded and used automatically during the Python package build process, without needing manual installation.
