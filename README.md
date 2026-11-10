# MAPLE

**Memory-Aware Predictive Loading Engine** — an experiment in *learned* context
selection for long-document LLM question answering.

> **Status: research experiment, v0.3.0 — negative result.** MAPLE does not
> select context better than plain dense retrieval. Numbers from earlier versions
> were affected by bugs and are archived, not claimed (see
> [docs/METRICS.md §10](docs/METRICS.md#10-numbers-that-must-not-be-reused)).

## Summary: the goal, how it was attempted, and whether it worked

**Goal.** Long-document question answering cannot send a whole document to an LLM,
so a few pieces have to be selected. The usual way, dense retrieval, picks the
pieces whose embeddings look most like the question, and misses pieces that matter
but look different. MAPLE set out to select context **better than dense retrieval
at almost the same cost**, by learning *where an LLM itself looks* when it answers.

**How.**
1. A teacher LLM (Llama-3-8B-Instruct, 4-bit) reads a question plus the document in
   500-character blocks; the attention its final token pays to each block becomes a
   soft relevance label (5,000 questions from HotpotQA and NarrativeQA: 4,000 for
   training, 1,000 held out, each labelled at two context budgets).
2. Cheap students learn to predict those labels: small scorers over frozen
   BGE-small embeddings (7 variants: residual-on-cosine, listwise KL loss, distractor
   training, human-label control), and the embedding model itself fine-tuned on the
   labels (full fine-tune, cross-document negatives, and a regularised version that
   trains only the top layers and stays close to the base model).
3. Each student was compared with BM25, dense retrieval, hybrid retrieval and a
   cross-encoder on held-out sets, long documents, BEIR, multi-hop QA and end-to-end
   answers from a fixed reader, with pass/fail rules written before each run,
   3 seeds, significance tests, and published baselines reproduced first.

**Did it work? No.**
* The idea is half right: the teacher's attention *does* find the evidence better
  than dense retrieval (HotpotQA 85.9 vs 82.8, NarrativeQA 77.3 vs 71.7
  evidence recall@5). But using it directly needs the 8B model at query time.
* No cheap student learned it. The best scorers trained on attention stay below
  dense retrieval (78.4 vs 82.8); fine-tuning the embedding model gains 1-2.5 points
  on the training datasets and loses up to 18 points elsewhere (BEIR, multi-hop QA).
  The regularised version is the best trade-off (+1.1 in-domain, −0.8 on BEIR) and
  still fails its pre-set bar.
* End to end, with the same reader, answers built from MAPLE's selection are not
  significantly different from dense retrieval's (LongBench F1 52.4-53.3 vs 51.5,
  *p* ≥ 0.17).
* It is fast (4.55 ms per query over 1M blocks on a GPU), but about 1.5 ms slower
  than the dense retrieval it fails to beat.
* Why: the teacher finds a passage such as "Adriana Trigiani… based in Greenwich
  Village" for "where is the director of *Big Stone Gap* based?" because it has just
  read the passage that names the director. A student that scores each block on its
  own cannot make that connection; models that read passages together
  (cross-encoders, multi-hop retrievers) are where the signal stays usable.
* The earlier v0.2.0 claims (98% recall, >90% needle, 98% SQuAD) came from bugs and
  benchmarks too easy to test retrieval; measured properly, that version was the
  weakest of all (63% vs 82.8% for dense, collapsing to 0.8% on 5,000-block pools).

Full tables are in [Results](#results).

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

## Results

Encoder BAAI/bge-small-en-v1.5 for every dense row and every MAPLE student; teacher
Llama-3-8B-Instruct (4-bit); means over seeds 42/43/44 unless stated; *p* is a
paired bootstrap against dense retrieval. Protocols in [docs/METRICS.md](docs/METRICS.md).

**Held-out evidence retrieval** (HotpotQA validation 500 questions, ~11 blocks each;
NarrativeQA test 500, ~11 blocks; teacher budget 1,536 tokens; evidence_recall@5, %):

| method | HotpotQA | NarrativeQA |
|---|---|---|
| BM25 | 73.4 | 57.3 |
| dense retrieval | 82.8 | 71.7 |
| teacher attention, mean over layers (not a deployable method: needs the 8B model) | 85.9 | 77.3 |
| MAPLE v1 (original MLP) | 63.1 | 58.4 |
| best MAPLE scorer trained on teacher attention | 78.4 | 65.1 |
| MAPLE scorer trained on human evidence labels | 83.7 (*p* 0.17) | 70.1 |
| MAPLE-Embed, regularised (top 2 encoder layers + KL to the base encoder) | 83.9 (*p* 0.01-0.11) | 71.4 |

**Out of domain** (BEIR SciFact / NFCorpus / FiQA / TREC-COVID, re-ranking BM25's top
100, macro nDCG@10; multi-hop: 1,000 questions each from MuSiQue / 2Wiki / HotpotQA,
re-ranking a dense top 20, Recall@5):

| method | BEIR | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|
| BM25 | 44.9 | 49.6 | 68.8 | 80.1 |
| dense retrieval | **53.9** | 50.6 | **70.8** | 85.9 |
| hybrid (BM25 + dense, RRF) | 52.0 | **52.4** | 70.7 | **86.6** |
| MAPLE scorer, human labels (seed 42) | 52.0 | 47.7 | 69.5 | 83.7 |
| MAPLE-Embed, full fine-tune | 51.6 | 33.2 | 53.5 | 79.5 |
| MAPLE-Embed, regularised | 53.1 | 49.8 | 70.2 | 86.0 |

On BEIR, our BM25 first stage reproduces published BM25 within 0.2-1.7 points; the
best published LLM re-ranker in our comparison, ICR with Llama-3.1-8B (Chen et al.,
ICLR 2025), reports 55.4 on the same four datasets.

**End to end, reader fixed** (Llama-3-8B-Instruct, 2,000-token context, LongBench v1,
200 questions per task, F1):

| method | HotpotQA | NarrativeQA |
|---|---|---|
| whole context (7,500 tokens) | 46.8 | 22.4 |
| dense retrieval | 51.5 | 17.6 |
| hybrid | 54.7 | 19.4 |
| cross-encoder (bge-reranker-base) | 53.5 | 21.4 |
| MAPLE scorer, human labels | 53.3 | 17.7 |
| MAPLE-Embed, regularised | 52.4 | 19.4 |

No selection method differs significantly from dense retrieval (*p* ≥ 0.17 for MAPLE).
Our whole-context row matches the published Llama-3-8B baseline (InfLLM, NeurIPS 2024:
47.38 / 19.85).

**Speed** (RTX 4090 Laptop; search only, query encoded): the `rerank` strategy takes
4.55 ms at 1M blocks on the GPU and 4.81 ms at 100k blocks on the CPU, about 1.5 ms
more than plain cosine similarity; bge-reranker-base needs ~237 ms for 100 candidates.

**What this means.** The teacher's attention is a better relevance signal than
embedding similarity, but a student that scores each block on its own cannot learn
it: the teacher finds a passage like "Adriana Trigiani… based in Greenwich Village"
for "where is the director of *Big Stone Gap* based?" because it has read the
passage naming the director in the same context. Fine-tuning the encoder gains
1-2.5 points in-domain and loses as much or more elsewhere, with human labels as
well. MAPLE is fast, but it does not beat dense retrieval.

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
