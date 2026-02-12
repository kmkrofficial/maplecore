# Benchmarks

Performance evaluation scripts for MAPLE.

## Core Benchmarks
- **`01_recall_enhanced.py`**: Precision/Recall tests on NarrativeQA & HotpotQA. Evaluates the model's ability to retrieve relevant context.
- **`02_latency_scaling.py`**: Latency analysis vs Index Size. Compares Linear, Adaptive, and Hierarchical search strategies.
- **`03_needle_robust.py`**: Robustness tests (Needle-in-a-Haystack) with variable depth and context length.

## Analysis
- **`04_cost_analysis.py`**: Estimates token and storage cost savings compared to full-context inference and RAG.
- **`05_ablation_chunking.py`**: Analyzes the impact of different chunk sizes on retrieval performance.
- **`profiler.py`**: Utility for in-situ hardware monitoring (CPU/RAM/VRAM) during benchmarks.

## Usage
Run benchmarks from the project root:
```bash
python benchmarks/01_recall_enhanced.py
```
