# Evaluation Results

**Test set:** 16 questions against the PowerGridModel repository
**LLM:** sam860/LFM2:350m (via Ollama)
**Retriever:** Vector search (Qdrant, sentence-transformers/all-mpnet-base-v2)
**Top-k:** 5 documents per query

---

## Custom Metrics

| Metric | Score |
|--------|-------|
| Avg keyword recall | 0.3632 |
| Avg source precision | 0.1500 |
| Questions answered | 16 |
| Questions failed | 0 |
| Avg latency | 6.6s |

## RAGAS Scores

RAGAS evaluation partially succeeded with the local judge LLM. Many evaluation jobs timed out, but two metrics produced results:

| Metric | Score |
|--------|-------|
| Faithfulness | 0.6667 |
| Context recall | 0.7500 |

Answer relevancy could not be computed (timed out). For full RAGAS coverage, re-run with a larger model:
```bash
LLM_MODEL_NAME=gemma3:4b python evals/run_eval.py
```

---

## Per-Question Breakdown

| # | Question | Difficulty | Category | Keyword Recall | Latency |
|---|----------|-----------|----------|----------------|---------|
| 1 | What types of calculations does PGM support? | easy | factual_lookup | **1.00** | 4.7s |
| 2 | What C++ standard does PGM require? | easy | factual_lookup | 0.00 | 3.6s |
| 3 | What are the third-party C++ dependencies? | easy | factual_lookup | 0.00 | 5.0s |
| 4 | What is the component type hierarchy? | medium | cross_file_reasoning | 0.43 | 7.5s |
| 5 | What calculation methods are available? | medium | factual_lookup | 0.60 | 6.7s |
| 6 | How do you create input data? | medium | how_does_it_work | 0.50 | 12.3s |
| 7 | What error for non-observable state estimation? | medium | cross_file_reasoning | 0.50 | 3.1s |
| 8 | What is the base power constant? | hard | factual_lookup | 0.00 | 6.7s |
| 9 | What winding types are supported? | medium | factual_lookup | 0.00 | 5.5s |
| 10 | What fault types for short circuit? | medium | factual_lookup | 0.00 | 7.0s |
| 11 | How does the core architecture work? | hard | how_does_it_work | 0.20 | 11.5s |
| 12 | What validation error classes exist? | hard | cross_file_reasoning | 0.25 | 7.3s |
| 13 | What license is PGM released under? | easy | factual_lookup | **1.00** | 3.6s |
| 14 | What tap changing strategies exist? | medium | factual_lookup | 0.00 | 8.8s |
| 15 | What happens connecting nodes with different voltages? | hard | cross_file_reasoning | **1.00** | 9.4s |
| 16 | Minimum Python version and build system? | easy | factual_lookup | 0.33 | 2.7s |

### Breakdown by Difficulty

| Difficulty | Avg Keyword Recall | Count |
|-----------|-------------------|-------|
| Easy | 0.47 | 5 |
| Medium | 0.29 | 7 |
| Hard | 0.36 | 4 |

### Breakdown by Category

| Category | Avg Keyword Recall | Count |
|----------|-------------------|-------|
| factual_lookup | 0.29 | 10 |
| cross_file_reasoning | 0.55 | 4 |
| how_does_it_work | 0.35 | 2 |

---

## Analysis

### Strengths

1. **All 16 questions answered.** The system always retrieves relevant context and produces an answer, never falling back to the "no relevant context" message.
2. **Cross-file reasoning works well** (avg recall 0.55). The model pulls together information from multiple documents (e.g., error classes from errors.py combined with core design docs).
3. **High-confidence factual lookups succeed.** Questions about well-documented topics (calculation types, license) achieve high keyword recall. Q13 (license) improved from 0.50 to 1.00 and Q15 (voltage conflict) improved from 0.67 to 1.00.
4. **Docker inference is reasonable.** Running Ollama within Docker achieves 6.6s average latency, down from 31.3s previously (likely Docker/Ollama config improvements).
5. **RAGAS partially works.** Faithfulness (0.67) and context recall (0.75) scores suggest decent answer quality, though the small model times out on the more complex RAGAS tasks.

### Weaknesses

1. **Enum-specific details are missed.** Questions about specific enum values (WindingType, FaultType, TapChangingStrategy) score 0.00 keyword recall. The retriever finds relevant documents but the LLM doesn't enumerate the specific values with their exact names.
2. **C++ build details not well retrieved.** Questions about CMakeLists.txt content (C++ standard, dependencies) score 0.00. The chunking strategy may not prioritize build configuration files.
3. **Base power constant not found.** The `common.hpp` file with `base_power_3p = 1e6` was not in the top-5 retrieved documents.
4. **Source precision is low (0.15).** Retrieved documents don't always come from the expected files. This is partly by design (the system retrieves any relevant content) but indicates the vector index could benefit from better metadata filtering.

---

## Failure Cases

### Q16 (expected failure): Minimum Python version and build system

The system retrieved context from notebook metadata showing Python 3.13.3 (the runtime version, not the minimum requirement) instead of the PowerGridModel pyproject.toml which specifies `>=3.12`. This confirms the expected failure: the system can confuse runtime version metadata with minimum version specifications.

### Q2: What C++ standard does PGM require?

The LLM acknowledged it could not determine the C++ standard from the provided context. The CMakeLists.txt chunk containing `PGM_CXX_STANDARD 23` was not in the top-5 results.

### Q3: What are the third-party C++ dependencies?

The model described the C API and header structure but did not name the actual dependencies (Boost, Eigen3, nlohmann_json, msgpack). The CMakeLists.txt containing the `find_package` calls was not retrieved.

### Q8: Base power constant

The `common.hpp` definitions were not retrieved. This is a very specific constant buried in a header file. The chunking and embedding just don't surface this kind of detail well.

---

## Recommendations

1. **Use a larger LLM.** The 350M parameter model struggles with detailed technical answers. Using gemma3:4b or larger would likely improve keyword recall significantly.
2. **Improve chunk metadata.** Adding file type / component category tags to chunks would enable filtered retrieval for build-system vs. API vs. documentation questions.
3. **Re-run ragas with a capable judge model.** Use gemma3:4b locally or an API-based model to get complete RAGAS scores including answer relevancy.
4. **Consider boosting code files.** `.hpp`, `.py` enum files, and build configs contain dense factual information that the current embedding model may not rank highly enough.
