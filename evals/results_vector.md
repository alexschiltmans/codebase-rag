# Evaluation Results

**Date:** 2026-07-18 22:40

**Test set:** 16 questions

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.3830 |
| avg_source_precision | 0.1875 |
| questions_answered | 16 |
| questions_failed | 0 |
| avg_latency_s | 0.8937 |

## RAGAS Scores

| Metric | Score |
|--------|-------|
| faithfulness | 0.9048 |
| context_recall | 0.6250 |

## Per-Question Breakdown

| # | Difficulty | Category | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1.00 | 5 | 1.8s | False |
| 2 | easy | factual_lookup | 0.00 | 5 | 0.7s | False |
| 3 | easy | factual_lookup | 0.00 | 5 | 0.9s | False |
| 4 | medium | cross_file_reasoning | 0.43 | 5 | 0.6s | False |
| 5 | medium | factual_lookup | 0.00 | 5 | 0.4s | False |
| 6 | medium | how_does_it_work | 0.50 | 5 | 1.6s | False |
| 7 | medium | cross_file_reasoning | 0.75 | 5 | 1.0s | False |
| 8 | hard | factual_lookup | 0.00 | 5 | 0.4s | False |
| 9 | medium | factual_lookup | 0.75 | 5 | 0.8s | False |
| 10 | medium | factual_lookup | 0.50 | 5 | 1.0s | False |
| 11 | hard | how_does_it_work | 0.20 | 5 | 1.1s | False |
| 12 | hard | cross_file_reasoning | 0.25 | 5 | 1.0s | False |
| 13 | easy | factual_lookup | 0.50 | 5 | 0.2s | False |
| 14 | medium | factual_lookup | 0.25 | 5 | 0.8s | False |
| 15 | hard | cross_file_reasoning | 0.67 | 5 | 1.2s | False |
| 16 | easy | factual_lookup | 0.33 | 5 | 0.9s | True |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required for building the Python package is **3.4**. The Python package uses the CMake project's default Python, which supports both C++ and Python 2.

Here's a breakdown of the details from the context:

- **Python Version Requirement:** The `nbformat` in Document 1 specifies that these dependencies are "build-time only," meaning they are automatically downloaded during the build process when using `uv sync` for source-based builds.

- **Build System:** When building a Python package via CMake, the following steps occur:
  - The CMake project downloads and installs all necessary libraries (including those specified in the dependency list).
  - It then uses these dependencies to create the final Python executable.

This setup ensures that when you build the package from source using `uv sync`, it leverages the built dependencies automatically, avoiding manual installation unless needed for standalone builds.

**License:** The license mentioned is **BSD-3**, which is a permissive free software license. It allows users to use, modify, and distribute the software both commercially and non-commercially, provided they include the original copyright notice and license text in any distributed copies of the software.
