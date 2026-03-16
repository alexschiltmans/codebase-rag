# Evaluation Results: Large Model (qwen3-coder:30b)

**Test set:** 16 questions against the PowerGridModel repository
**LLM:** qwen3-coder:30b (via local Ollama, not Docker, using macOS GPU)
**Retriever:** Vector search (Qdrant, sentence-transformers/all-mpnet-base-v2)
**Top-k:** 5 documents per query

---

## Custom Metrics

| Metric | Score |
|--------|-------|
| Avg keyword recall | 0.4771 |
| Avg source precision | 0.1750 |
| Questions answered | 16 |
| Questions failed | 0 |
| Avg latency | 5.6s |

## RAGAS Scores

RAGAS evaluation was skipped (`--skip-ragas`) due to the high cost of running a 30B judge model locally for structured metric extraction.

---

## Per-Question Breakdown

| # | Question | Difficulty | Category | Keyword Recall | Latency |
|---|----------|-----------|----------|----------------|---------|
| 1 | What types of calculations does PGM support? | easy | factual_lookup | **1.00** | 13.5s |
| 2 | What C++ standard does PGM require? | easy | factual_lookup | 0.00 | 3.8s |
| 3 | What are the third-party C++ dependencies? | easy | factual_lookup | 0.25 | 4.6s |
| 4 | What is the component type hierarchy? | medium | cross_file_reasoning | **1.00** | 6.7s |
| 5 | What calculation methods are available? | medium | factual_lookup | 0.60 | 8.6s |
| 6 | How do you create input data? | medium | how_does_it_work | 0.50 | 7.2s |
| 7 | What error for non-observable state estimation? | medium | cross_file_reasoning | 0.50 | 3.1s |
| 8 | What is the base power constant? | hard | factual_lookup | 0.00 | 3.2s |
| 9 | What winding types are supported? | medium | factual_lookup | 0.00 | 6.0s |
| 10 | What fault types for short circuit? | medium | factual_lookup | **0.75** | 6.1s |
| 11 | How does the core architecture work? | hard | how_does_it_work | 0.20 | 6.0s |
| 12 | What validation error classes exist? | hard | cross_file_reasoning | 0.50 | 5.3s |
| 13 | What license is PGM released under? | easy | factual_lookup | **1.00** | 2.7s |
| 14 | What tap changing strategies exist? | medium | factual_lookup | 0.00 | 5.4s |
| 15 | What happens connecting nodes with different voltages? | hard | cross_file_reasoning | **0.67** | 4.2s |
| 16 | Minimum Python version and build system? | easy | factual_lookup | 0.67 | 3.9s |

### Breakdown by Difficulty

| Difficulty | Avg Keyword Recall | Count |
|-----------|-------------------|-------|
| Easy | 0.58 | 5 |
| Medium | 0.48 | 7 |
| Hard | 0.34 | 4 |

### Breakdown by Category

| Category | Avg Keyword Recall | Count |
|----------|-------------------|-------|
| factual_lookup | 0.43 | 10 |
| cross_file_reasoning | 0.67 | 4 |
| how_does_it_work | 0.35 | 2 |

---

## Comparison: Small Model vs Large Model

| Metric | Small (LFM2 350M) | Large (qwen3-coder 30B) | Delta |
|--------|---------------------|------------------------|-------|
| **Avg keyword recall** | 0.363 | 0.477 | **+31%** |
| **Avg source precision** | 0.150 | 0.175 | +17% |
| **Questions answered** | 16 | 16 | - |
| **Avg latency** | 6.6s | 5.6s | -15% |

### Per-Question Keyword Recall Comparison

| # | Question (abbreviated) | Small | Large | Winner |
|---|----------------------|-------|-------|--------|
| 1 | Calculation types | 1.00 | 1.00 | Tie |
| 2 | C++ standard | 0.00 | 0.00 | Tie |
| 3 | C++ dependencies | 0.00 | **0.25** | Large |
| 4 | Component hierarchy | 0.43 | **1.00** | Large |
| 5 | Calculation methods | 0.60 | 0.60 | Tie |
| 6 | Create input data | 0.50 | 0.50 | Tie |
| 7 | NotObservableError | 0.50 | 0.50 | Tie |
| 8 | Base power constant | 0.00 | 0.00 | Tie |
| 9 | Winding types | 0.00 | 0.00 | Tie |
| 10 | Fault types | 0.00 | **0.75** | Large |
| 11 | Core architecture | 0.20 | 0.20 | Tie |
| 12 | Validation errors | 0.25 | **0.50** | Large |
| 13 | License | 1.00 | 1.00 | Tie |
| 14 | Tap changing strategies | 0.00 | 0.00 | Tie |
| 15 | Voltage conflict | **1.00** | 0.67 | Small |
| 16 | Python version (expected fail) | 0.33 | **0.67** | Large |

**Summary:** Large model wins on 5 questions, small model wins on 1 (Q15), tie on 10.

### Keyword Recall by Difficulty

| Difficulty | Small | Large | Delta |
|-----------|-------|-------|-------|
| Easy | 0.47 | 0.58 | +23% |
| Medium | 0.29 | 0.48 | +66% |
| Hard | 0.36 | 0.34 | -6% |

### Keyword Recall by Category

| Category | Small | Large | Delta |
|----------|-------|-------|-------|
| factual_lookup | 0.29 | 0.43 | +48% |
| cross_file_reasoning | 0.55 | 0.67 | +22% |
| how_does_it_work | 0.35 | 0.35 | 0% |

---

## Conclusions

### 1. Larger model improves answer quality but not retrieval

The 30B model increased average keyword recall by 31% (0.36 → 0.48), with the biggest improvement on **factual lookups** (+48%). This makes sense: the larger model is better at extracting and articulating specific facts from the retrieved context. Source precision was similar (0.15 vs 0.175), confirming that the retrieval pipeline is the same, and model size primarily affects generation quality.

### 2. Retrieval is the bottleneck, not generation

Both models scored 0.00 on the same 4 questions (Q2, Q8, Q9, Q14). These failures share a pattern: the expected answer lives in a specific file type (CMakeLists.txt, common.hpp, enum.py) that the retriever did not surface in the top-5 results. No amount of model intelligence can compensate for missing context. The retrieval pipeline would benefit from:
- Better chunking of build configuration files (CMake, pyproject.toml)
- Boosting code files (`.py`, `.hpp`) that contain dense enum/constant definitions
- Metadata filtering by file type for targeted queries

### 3. Latency improved dramatically

The large model was tested with native macOS GPU (Metal) acceleration (5.6s), while the small model ran in Docker (6.6s). Despite Docker overhead, the 350M model hits comparable latency to the 30B model running natively.

### 4. Cross-file reasoning is the system's strength

Both models performed best on cross-file reasoning (small: 0.55, large: 0.67). The retriever successfully finds relevant documents across multiple files, and both models can synthesize information from multiple chunks. This confirms the approach works for its primary use case.

### 5. Persistent failure cases reveal system limitations

| Question | Root Cause |
|----------|-----------|
| Q2: C++ standard | CMakeLists.txt not chunked/embedded well enough for "C++ standard" queries |
| Q8: Base power constant | `common.hpp` with `base_power_3p = 1e6` not in top-5 results |
| Q9: Winding types | `WindingType` enum values not in retrieved chunks |
| Q14: Tap changing strategies | `TapChangingStrategy` enum not in retrieved chunks |

All four failures are **enum/constant lookups** where the answer is a short list of values in a single file. The embedding model doesn't represent these well, and the chunks containing them don't score high enough to make the top-5.

### 6. Overall assessment

The codebase RAG system works well for:
- **High-level conceptual questions** (calculation types, architecture, licensing)
- **Cross-file reasoning** (error classes, component hierarchies, voltage conflicts)

It struggles with:
- **Specific constant/enum value lookups** (winding types, tap strategies, base power)
- **Build system details** (C++ standard, dependencies from CMakeLists.txt)

The 30B model is the better choice for production use, more accurate on factual lookups and medium-difficulty questions. The 350M model is adequate for basic Q&A and actually matches or beats the large model on hard cross-file reasoning questions (Q15), but misses nuance on factual details.

---

## Failure Cases

### Q16 (expected failure): Minimum Python version and build system

Both models retrieved context from notebook metadata showing Python 3.13.3 (the runtime version, not the minimum requirement) instead of `pyproject.toml` which specifies `>=3.12`. The large model did correctly identify `scikit-build-core` as the build system (keyword recall 0.67 vs 0.33 for small). This confirms the expected failure: the system can confuse runtime version metadata with minimum version specifications.

### Q2: What C++ standard does PGM require?

Both models failed (KR 0.00). The CMakeLists.txt chunk containing `PGM_CXX_STANDARD 23` was not retrieved. Both models acknowledged they couldn't determine the specific C++ standard from the provided context.

### Q8: Base power constant

Both models failed (KR 0.00). The `common.hpp` header containing `base_power_3p = 1e6` was not in retrieved documents. This is a very specific constant in a C++ header file.

### Q9/Q14: Enum value lookups

Both models failed on `WindingType` and `TapChangingStrategy` enum value lookups. The retriever finds descriptive documentation about transformers and tap changers, but not the `enum.py` file containing the actual enum definitions with their values.
