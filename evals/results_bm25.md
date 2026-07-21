# Evaluation Results

**Date:** 2026-07-21 09:17

**Test set:** 16 questions

**Latency probe:** 1.42s (single generation timed before the test set ran; compare `avg_latency_s` only against runs with a similar probe — a high probe means the run was contended)

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.4473 |
| avg_source_precision | 0.1750 |
| avg_hit_rate | 0.4000 |
| avg_mrr | 0.2244 |
| questions_answered | 16 |
| questions_failed | 0 |
| avg_latency_s | 0.8735 |

## RAGAS Scores (judge: `qwen3.5:9b`)

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5865 | 15/16 |
| answer_relevancy | 0.7824 | 16/16 |

## Per-Question Breakdown

| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|-----|----|-----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 0 | 0.00 | 0.67 | 5 | 0.9s | False |
| 2 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 1.0s | False |
| 3 | easy | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.5s | False |
| 4 | medium | cross_file_reasoning | 1 | 0.50 | 0.86 | 5 | 1.5s | False |
| 5 | medium | factual_lookup | 0 | 0.00 | 0.80 | 5 | 0.9s | False |
| 6 | medium | how_does_it_work | 0 | 0.00 | 0.25 | 5 | 1.5s | False |
| 7 | medium | cross_file_reasoning | 1 | 0.20 | 0.75 | 5 | 0.9s | False |
| 8 | hard | factual_lookup | 0 | 0.00 | 0.00 | 5 | 1.0s | False |
| 9 | medium | factual_lookup | 0 | 0.00 | 0.00 | 5 | 0.6s | False |
| 10 | medium | factual_lookup | 0 | 0.00 | 1.00 | 5 | 0.7s | False |
| 11 | hard | how_does_it_work | 1 | 0.33 | 0.00 | 5 | 1.4s | False |
| 12 | hard | cross_file_reasoning | 1 | 1.00 | 0.00 | 5 | 0.5s | False |
| 13 | easy | factual_lookup | 1 | 1.00 | 0.50 | 5 | 0.3s | False |
| 14 | medium | factual_lookup | 1 | 0.33 | 1.00 | 5 | 0.6s | False |
| 15 | hard | cross_file_reasoning | 0 | 0.00 | 0.67 | 5 | 1.2s | False |
| 16 | easy | factual_lookup | - | - | 0.67 | 5 | 0.5s | True |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the `power-grid-model` Python package is **3.12**. The build system used is **CMake** with the following configuration:

```cmake
build_dependencies(
    BUILD_DEPS="${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml"
)
build_system = "scikit_build_core.build"
```

This indicates that CMake is used to build the project, and `pyproject.toml` is the configuration file for defining dependencies and other package metadata.
