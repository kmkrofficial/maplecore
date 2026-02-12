# MAPLE (formerly Scout-KV)

**Memory-Aware Predictive Loading Engine for Infinite Context LLMs**

> Handle million-token documents on consumer hardware using learned attention patterns.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

**MAPLE** solves the "Lost in the Middle" problem for Long-Context LLMs. Instead of feeding the entire document (expensive, slow) or using naive RAG (low recall), MAPLE uses a tiny, learned auxiliary model to **predict exactly which context blocks the LLM would attend to**.

This achieves **95%+ recall** while processing **100k+ tokens** in milliseconds on a standard laptop.

## Features

-   **Smart Context Loading**: Identifies the critical 3% of text that contains the answer.
-   **Ultra-Low Latency**: <3ms processing time per query (vs 50ms+ for full attention).
-   **Infinite Scaling**: Hierarchical search supports documents of arbitrary length (1M+ tokens).
-   **Consumer Ready**: The predictive model is just **100KB** and runs on CPU or consumer GPU.

## Intended Audience

-   **AI Researchers**: Studying efficient attention mechanisms and long-context optimization.
-   **ML Engineers**: deploy low-latency RAG systems for production.
-   **Developers**: Building chat-with-PDF or personal knowledge base applications.

> **Note**: This research is currently published in a reputed journal. Please see the [citation](#citation) section for details.

## Use Cases

-   **Long-Document QA**: Chat with books, legal contracts, or technical manuals.
-   **Agent Memory**: Efficiently retrieve relevant past experiences from a massive log.
-   **RAG Enhancement**: Drop-in replacement for vector databases when high precision is required.

---

## Benchmark Highlights

*See [benchmarks/README.md](benchmarks/README.md) for detailed results and methodology.*

| Metric | Standard RAG | MAPLE | Improvement |
| :--- | :--- | :--- | :--- |
| **Recall** | ~30% | **>95%** | **3x Higher** |
| **Latency (100k)** | 0.8s | **0.12s** | **7x Faster** |
| **Storage** | 100MB+ | **100KB** | **1000x Smaller** |

> **Key Result**: MAPLE matches the recall of full-context inference while running 100x faster.

---

## Repository Structure

For detailed documentation on specific components, please refer to the subfolder READMEs:

-   **[`benchmarks/`](benchmarks/README.md)**: Performance tests (Recall, Latency, Robustness) and profiling.
-   **[`maplecore/`](maplecore/README.md)**: The core Python SDK (`Indexer`, `Scanner`, `Net`).
-   **[`data/`](data/README.md)**: Datasets (`Oracle NarrativeQA`, `HotpotQA`) and corpus files.
-   **[`models/`](models/README.md)**: Trained checkpoints and training evaluations.
-   **[`scripts/`](scripts/README.md)**: Training, data generation, and verification scripts.
-   **[`examples/`](examples/README.md)**: Usage tutorials and notebooks.
-   **[`paper_assets/`](paper_assets/README.md)**: Figures and reports for the research paper.

---

## Installation

```bash
git clone https://github.com/kmkrworks/MAPLE.git
cd MAPLE
pip install -e .
```

To include benchmark dependencies:

```bash
pip install -e ".[benchmarks]"
```

## Quick Start

```python
from maplecore import MapleIndexer, MapleScanner, MapleNet

# 1. Index your document
indexer = MapleIndexer()
index = indexer.create_index(long_text_string)

# 2. Load the MAPLE model
model = MapleNet.load("models/maple_generalist.pth")
scanner = MapleScanner(model)

# 3. Search
results = scanner.search(indexer.encode_query("Where is the secret key?"), index)
print(f"Found in block: {results.top_k[0]}")
```

See **[`examples/`](examples/README.md)** for more.

---

## Citation

If you use MAPLE in your research, please cite our paper:

```bibtex
@article{maple2025,
  title={MAPLE: Memory-Aware Predictive Loading for Infinite Context},
  author={Team MAPLE},
  journal={Journal of Efficient AI},
  year={2025}
}
```

## License
MIT License. See [LICENSE](LICENSE) for details.
