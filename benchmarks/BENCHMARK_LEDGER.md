# MAPLE Benchmark Tracking Ledger
=================================

This ledger tracks the progression of End-to-End architectural evaluations across the MAPLE library.

| Date | Run ID / Commit | Models Tested | Embedding Model | Recall Metric | E2E Mating LLM | Latency Context | Fidelity Result |
|---|---|---|---|---|---|---|---|
| 2026-02-21 | pre-refactor | gemini-flash, gemini-flash-lite, gemini-pro | BAAI/bge-small-en-v1.5 | - | NarrativeQA | Top-10 / 5000 chars | 18% (Zero-Shot Baseline) |

*Note: The 18% accuracy achieved on NarrativeQA serves as a standard baseline for zero-shot chunked retrieval on highly abstractive literary datasets. This motivated the pivot to factual, extractive QA datasets like SQuAD and HotpotQA for precise engine validation.*
