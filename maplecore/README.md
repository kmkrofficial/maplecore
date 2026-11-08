# maplecore (SDK)

| Module | Public objects | Purpose |
|---|---|---|
| `client.py` | `Maple` | High-level API: `index_text` / `index_file` → `query` → `get_context`; `save_index` / `load_index`. Default strategy: `rerank` |
| `indexer.py` | `MapleIndexer`, `Index`, `Block` | Fixed-size character chunking (optional overlap), sentence-transformer encoding, index persistence (records the encoder name), cached chapter and unit-norm embeddings |
| `core.py` | `MapleNet` | v1 MLP `2·dim → 128 → 1`; `load()` infers dimensions from the checkpoint |
| `models.py` | `MapleResidualNet`, `save_model`, `load_model`, `export_to_onnx` | Residual scorer that starts exactly at cosine similarity; self-describing v2 checkpoints (also reads v1 state dicts) |
| `listwise.py` | `QueryGroup`, `listwise_kl_loss`, `ListwiseTrainer` | Listwise KL training over each question's whole block list, best-epoch restore, early stopping |
| `search.py` | `MapleScanner`, `SearchResult`, `SearchStrategy` | `rerank`, `linear`, `hierarchical`, `adaptive` search (see below) |
| `trainer.py` | `MapleTrainer` | v1 pairwise BCE training with per-group split and Recall@5 model selection |
| `onnx_runner.py` | `MapleONNXRunner` | ONNX Runtime CPU inference; drop-in model for `MapleScanner` |
| `orchestrator.py` | `RecursiveOrchestrator` | Experimental multi-hop Search → Reason → Search loop around any `llm(prompt) -> str` callable |
| `context_manager.py` | `ContextSynthesizer` | Deduplicated (SHA-256), size-capped context window for the orchestrator |
| `parsers.py` | `extract_text_from_file` | `.pdf` (PyMuPDF), `.docx` (python-docx), `.html` (BeautifulSoup), plain text |
| `utils.py` | `get_device`, `resolve_device`, `setup_logging` | Device selection (`None`/`"auto"` → CUDA if a device is visible, else CPU) |

## Search strategies

* **rerank** (default) — one matrix-vector product against the cached unit-norm embeddings keeps the
  `rerank_candidates` (200) most similar blocks; the MAPLE scorer ranks only those. Equals `linear` when the
  candidates cover the index.
* **linear** — score all `N` blocks.
* **hierarchical** — with `ceil(N / C)` chapters (`C = 100`, the last may be partial, so every block is reachable),
  score mean-pooled chapter embeddings, then the blocks of the top `fast_chapters` (5) chapters.
* **adaptive** — as hierarchical, but widen to `fallback_chapters` (50) when the best chapter score is below
  `confidence_threshold` (0.15) or, if `entropy_threshold` is set, when chapter scores are flat.

Caveat: scorers are trained on single-block embeddings; scoring mean-pooled chapter embeddings (hierarchical /
adaptive) is an untested approximation whose cost benchmark 07 measures.
