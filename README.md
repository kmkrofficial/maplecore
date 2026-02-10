# MAPLE 🍁

**Memory-Aware Predictive Loading Engine for Infinite Context LLMs**

> Handle million-token documents on consumer hardware using learned attention patterns.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 The Problem

Modern LLMs have limited context windows (4K–128K tokens). Current solutions have critical flaws:

| Approach | Problem |
|-------------|-------------|
| **Cloud APIs** | $$$, latency, privacy concerns |
| **Standard RAG** | Only **30% Recall** — misses relevant context |
| **Naive Chunking** | Destroys semantic boundaries |

## 🚀 The Solution: MAPLE

MAPLE learns which context blocks *actually matter* by analyzing LLM attention patterns — a technique we call **Memory-Aware Predictive Loading**:

```
Document (1M tokens) → MAPLE (1ms) → Top-5 Blocks → LLM (50ms) → Answer
```

### Key Results

| Metric | RAG | MAPLE | Improvement |
|--------|-----|-------|-------------|
| **Recall@5** | 29.6% | **71.6%** | 2.4x better |
| **Latency** | 0.85ms | 2.71ms | Acceptable |
| **Model Size** | N/A | 100 KB | Tiny |

### Why It Works: 97% Attention Sparsity

We discovered that LLM attention is **extremely sparse**:
- Only **3%** of context blocks receive meaningful attention
- The rest can be safely pruned without affecting answer quality
- MAPLE learns to predict which blocks will be attended to

---

## 📦 Installation

```bash
pip install maplecore
```

Or from source:

```bash
git clone https://github.com/kmkrworks/maple.git
cd maple
pip install -e .
```

To include benchmark dependencies:

```bash
pip install -e ".[benchmarks]"
```

---

## ⚡ Quick Start

```python
from maplecore import Maple

# Initialize client
client = Maple(model_path="maple.pth")

# Index a document
client.index_file("books/sherlock_holmes.txt")

# Query
results = client.query("What was the speckled band?")

# Get relevant context
context = client.get_context(results, max_blocks=5)
print(context)
```

### Search Strategies

```python
# Adaptive (default): Entropy-aware with dynamic K
results = client.query("Who is the killer?", strategy="adaptive")

# Hierarchical: For 50K+ blocks
results = client.query("Find the murder weapon", strategy="hierarchical")

# Linear: Simple top-k
results = client.query("Describe Watson", strategy="linear")
```

---

## 🏗️ Architecture

```
maplecore/
├── core.py       # MapleNet model (768 → 128 → 1)
├── client.py     # Maple client (high-level API)
├── indexer.py    # BGE embedding, chunking, I/O
├── search.py     # Linear, Hierarchical, Adaptive search
├── trainer.py    # Training logic
└── utils.py      # Device handling, helpers
```

### Components

| Component | Description |
|-----------|-------------|
| **MapleIndexer** | Chunks documents and generates BGE embeddings |
| **MapleNet** | Lightweight MLP that scores block relevance |
| **MapleScanner** | Implements search strategies (Linear, Hierarchical, Adaptive) |
| **Maple** | High-level client API |

---

## 🧪 Training Your Own MAPLE

```python
from maplecore import MapleTrainer, MapleIndexer

# Prepare training data with oracle labels
training_data = [
    {"query_emb": q, "block_emb": b, "label": 1, "question": "...", "block_id": 0},
    ...
]

# Train
trainer = MapleTrainer(device="cuda")
model, recall = trainer.train(
    training_data,
    epochs=20,
    save_path="maple.pth"
)

print(f"Best Recall@5: {recall*100:.1f}%")
```

---

## 📊 Benchmarks

Run the full benchmark suite:

```bash
pip install -e ".[benchmarks]"
python -m benchmarks.run_all
```

### Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `01_recall_narrativeqa` | Full NarrativeQA recall: MAPLE vs RAG |
| `02_latency_scaling` | Latency from 10K to 1M blocks |
| `03_needle_in_haystack` | Precision at varying depths |
| `04_cost_analysis` | Token cost savings vs full context |

### Scale Performance (Hierarchical Search)

| Blocks | Linear | Hierarchical | Speedup |
|--------|--------|--------------|---------|
| 1,000 | 18ms | 1.2ms | 15x |
| 50,000 | 900ms | 60ms | 15x |
| 100,000 | 1.8s | 120ms | 15x |

### Quantization (INT8)

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| Size | 378 KB | 98 KB | -74% |
| Accuracy | 71.6% | 71.6% | 0% |

---

## 🔧 Configuration

```python
from maplecore import Maple

client = Maple(
    model_path="maple.pth",
    device="cuda",           # or "cpu"
    chunk_size=500,          # characters per block
    strategy="adaptive"      # default search strategy
)
```

### Adaptive Search Parameters

The MapleScanner class accepts tuning parameters:

```python
from maplecore import MapleScanner, MapleNet

model = MapleNet.load("maple.pth")
scanner = MapleScanner(
    model,
    confidence_threshold=0.15,  # Below this → RAG fallback
    mass_target=0.80,           # Accumulate until 80% attention mass
    max_blocks=20,              # Never return more than this
    chapter_size=100            # Blocks per chapter (hierarchical)
)
```

---

## 📚 Examples

See the `examples/` directory:

- `demo_infinite_context.py` — Basic usage with Sherlock Holmes
- `benchmark_vs_rag.py` — Compare MAPLE vs standard RAG

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for embeddings
- [Meta Llama 3](https://llama.meta.com/) for attention oracle generation
- [NarrativeQA](https://github.com/deepmind/narrativeqa) dataset

---

**Built with 🍁 for the infinite context future.**
