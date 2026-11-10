# MAPLE

**Memory-Aware Predictive Loading Engine** — an experiment in *learned* context
selection for long-document LLM question answering.

> **Status: research experiment (v0.3.0).** There is no
> paper and no validated result yet. Numbers from earlier versions were affected
> by bugs and are archived, not claimed (see
> [docs/METRICS.md §10](docs/METRICS.md#10-numbers-that-must-not-be-reused)).

## Idea

Instead of sending a whole document to an LLM, or retrieving blocks by
embedding similarity alone, MAPLE learns **which blocks a teacher LLM attends
to** when it answers a question, and uses that to pick context:

1. **Oracle labels.** A teacher (Llama-3-8B-Instruct, 4-bit) reads a question
   plus 500-character blocks. The final token's attention is summed per block
   (last layer, and the mean over all layers), giving a soft relevance
   distribution over the blocks.
2. **Students.** Two ways of learning that distribution:
   * a small scorer over frozen BGE-small embeddings (`MapleResidualNet`: starts
     exactly at cosine similarity and learns a residual), trained with a
     listwise KL loss over each question's blocks, optionally with distractor
     blocks from other documents;
   * **MAPLE-Embed**: the sentence encoder itself fine-tuned so that its
     cosine ranking matches the teacher (optionally only the top layers, with a
     KL term that keeps it close to the base encoder).
3. **Search.** `rerank` (default): one matrix-vector product keeps the 200 most
   similar blocks, then the scorer orders only those. `linear` scores every
   block; `hierarchical` / `adaptive` score mean-pooled "chapters" first.

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

# 2. Oracle labels (GPU, ~2 h for both teacher budgets)
bash scripts/run_oracle_all.sh

# 3. Students
bash scripts/run_gate.sh gate_a              # scorer variants, 3 seeds x 2 budgets, + evaluation
bash scripts/run_embed.sh                    # MAPLE-Embed (full fine-tune), 3 seeds
bash scripts/run_embed_r.sh                  # MAPLE-Embed, top-2 layers + KL to the base encoder

# 4. Evaluate (results in benchmarks/results/, every run logged under runs/)
python benchmarks/01_recall_enhanced.py --embedding-model <encoder>      # held-out evidence retrieval
python benchmarks/08_beir_rerank.py --rankers bm25 dense hybrid cross_encoder maple:main --model main=<ckpt>
python benchmarks/09_multihop_rerank.py --rankers bm25 dense hybrid maple:main --model main=<ckpt>
python benchmarks/02_latency_scaling.py --devices cuda cpu --model <ckpt>
python benchmarks/11_efficiency.py --rankers bm25 dense hybrid cross_encoder maple:main --model main=<ckpt>
python benchmarks/10_e2e_longbench.py --rankers bm25 dense maple:main --model main=<ckpt>   # reader-fixed QA
```

On Windows, GPU jobs should run one at a time: a process that outgrows the
GPU's memory is paged into system RAM and slows down many times instead of
failing. `scripts/train_biencoder.py --max-gpu-mem-frac 0.8` caps the
allocator; `scripts/gpu_mem_sampler.ps1` logs dedicated vs shared GPU memory.

## Repository layout

| Path | Contents |
|---|---|
| [`maplecore/`](maplecore/README.md) | SDK: indexer, `MapleNet`, scanner, trainer, client, ONNX runner, multi-hop orchestrator |
| [`scripts/`](scripts/README.md) | Data download, oracle generation, training |
| [`benchmarks/`](benchmarks/README.md) | Evaluation suite and shared metrics |
| [`examples/`](examples/README.md) | Small usage and verification scripts |
| [`tests/`](tests/) | Unit tests (`pytest`); no downloads needed |
| [`docs/`](docs/) | [METRICS](docs/METRICS.md) · [DATA](docs/DATA.md) · [RELATED_WORK](docs/RELATED_WORK.md) · [archive](docs/archive/) |
| `docs/private/`, `scratch/` | Local-only notes and scratch work — gitignored, never committed |

## Development

```bash
pytest            # unit tests, CPU-only, no model downloads
```

## License

MIT — see [LICENSE](LICENSE).
