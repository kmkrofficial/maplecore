# MAPLE

**Memory-Aware Predictive Loading Engine** — an experiment in *learned* context
selection for long-document LLM question answering.

> **Status: research experiment (v0.2.0-beta → v0.3 in progress).** There is no
> paper and no validated result yet. Numbers from earlier versions were affected
> by bugs and are archived, not claimed. See [docs/GAPS.md](docs/GAPS.md) for
> the known-issues tracker.

## Idea

Instead of sending a whole document to an LLM, or retrieving blocks by
embedding similarity, MAPLE trains a tiny MLP (`MapleNet`, ~100K parameters)
to predict **which blocks a teacher LLM would attend to** when answering a
question:

1. **Oracle labels.** A teacher (Llama-3-8B-Instruct, 4-bit) reads a question
   plus 500-character blocks. The final token's attention is summed per block;
   the top-5 blocks become positive labels.
2. **Training.** `MapleNet` scores `[query_embedding ‖ block_embedding]`
   (BGE-small, 384 + 384 dims) and is trained with BCE on those labels.
3. **Search.** Blocks are scored linearly, or hierarchically: mean-pooled
   "chapters" of 100 blocks are scored first, then blocks inside the best
   chapters. An adaptive mode widens the search when confidence is low.

The hypothesis under test, and how it is measured, is written down in
[docs/METRICS.md](docs/METRICS.md).

## Install

```bash
git clone https://github.com/kmkrofficial/maplecore.git
cd maplecore
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# NVIDIA GPU: install a CUDA build of PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -e .                                        # SDK only
pip install -e ".[benchmarks,parsers,production,training,dev]"   # everything
```

Python ≥ 3.10. Tested on Windows 11 with Python 3.12, torch 2.11 (CUDA 12.8)
and an RTX 4090 Laptop (16 GB). `requirements-lock.txt` pins the exact versions used.

## Quick start

A trained checkpoint is required (see *Reproducing* below; no weights are
published yet).

```python
from maplecore import Maple

client = Maple(model_path="models/maple_generalist.pth")   # device auto-detected
client.index_file("book.txt")            # .txt / .pdf / .docx / .html
result = client.query("Who stole the jewel?", k=5)

print(result.block_ids, result.strategy_used, f"{result.latency_ms:.1f} ms")
print(client.get_context(result))        # concatenated text of the top blocks
```

Lower-level API:

```python
from maplecore import MapleIndexer, MapleNet, MapleScanner

indexer = MapleIndexer()                              # BAAI/bge-small-en-v1.5
index = indexer.create_index(text, chunk_size=500)
scanner = MapleScanner(MapleNet.load("models/maple_generalist.pth"))
result = scanner.search(indexer.encode_query("..."), index, k=5, strategy="adaptive")
```

## Reproducing the experiment

```bash
# 1. Data (see docs/DATA.md). Needs `hf auth login` + accepted Llama-3 licence for the teacher.
python scripts/download_data.py
python scripts/download_data.py --only teacher

# 2. Oracle labels (GPU)
python scripts/generate_oracle.py        --split train      --max-samples 2000
python scripts/generate_oracle.py        --split test       --max-samples 500
python scripts/generate_oracle_hotpot.py --split train      --max-samples 2000
python scripts/generate_oracle_hotpot.py --split validation --max-samples 500

# 3. Train
python scripts/train_generalist.py --seed 42

# 4. Evaluate
python -m benchmarks.run_all                 # 01-05, 07
python benchmarks/06_e2e_generation.py       # needs GEMINI_API_KEY in .env
```

## Repository layout

| Path | Contents |
|---|---|
| [`maplecore/`](maplecore/README.md) | SDK: indexer, `MapleNet`, scanner, trainer, client, ONNX runner, multi-hop orchestrator |
| [`scripts/`](scripts/README.md) | Data download, oracle generation, training |
| [`benchmarks/`](benchmarks/README.md) | Evaluation suite and shared metrics |
| [`examples/`](examples/README.md) | Small usage and verification scripts |
| [`tests/`](tests/) | Unit tests (`pytest`); no downloads needed |
| [`docs/`](docs/) | [GAPS](docs/GAPS.md) · [METRICS](docs/METRICS.md) · [DATA](docs/DATA.md) · [archive](docs/archive/) |

## Development

```bash
pytest            # unit tests, CPU-only, no model downloads
```

## License

MIT — see [LICENSE](LICENSE).
