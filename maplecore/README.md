# maplecore (SDK)

| Module | Public objects | Purpose |
|---|---|---|
| `client.py` | `Maple` | High-level API: `index_text` / `index_file` → `query` → `get_context`; `save_index` / `load_index` |
| `indexer.py` | `MapleIndexer`, `Index`, `Block` | Fixed-size character chunking (optional overlap), sentence-transformer encoding, index persistence (records the encoder name), cached chapter embeddings |
| `core.py` | `MapleNet` | MLP `2·dim → 128 → 1`; `load()` infers dimensions from the checkpoint; `export_to_onnx()` |
| `search.py` | `MapleScanner`, `SearchResult`, `SearchStrategy` | `linear`, `hierarchical`, `adaptive` search (see below) |
| `trainer.py` | `MapleTrainer` | BCE training with per-group train/val split, Recall@5 model selection, seeding |
| `onnx_runner.py` | `MapleONNXRunner` | ONNX Runtime CPU inference; drop-in model for `MapleScanner` |
| `orchestrator.py` | `RecursiveOrchestrator` | Experimental multi-hop Search → Reason → Search loop around any `llm(prompt) -> str` callable |
| `context_manager.py` | `ContextSynthesizer` | Deduplicated (SHA-256), size-capped context window for the orchestrator |
| `parsers.py` | `extract_text_from_file` | `.pdf` (PyMuPDF), `.docx` (python-docx), `.html` (BeautifulSoup), plain text |
| `utils.py` | `get_device`, `resolve_device`, `setup_logging` | Device selection (`None`/`"auto"` → CUDA if available, else CPU) |

## Search strategies

With `N` blocks and `chapter_size = C` (default 100), there are `ceil(N / C)`
chapters; the last may be partial, so every block is reachable.

* **linear** — score all `N` blocks.
* **hierarchical** — score mean-pooled chapter embeddings, then the blocks in the
  top `fast_chapters` (default 5) chapters.
* **adaptive** — as hierarchical, but widen to `fallback_chapters` (default 50)
  when the best chapter score is below `confidence_threshold` (default 0.15) or,
  if `entropy_threshold` is set, when chapter scores are too flat. Returns blocks
  until `mass_target` (0.8) of candidate score mass is covered, at least `k`,
  at most `max(k, max_blocks)`.

If an index has no more chapters than the fast path would scan, hierarchical and
adaptive scan everything (`metadata["action"]` says which path ran).

Caveat: `MapleNet` is trained on single-block embeddings; scoring mean-pooled
chapter embeddings with it is an untested approximation. Benchmark 07 measures
the resulting recall gap against linear search.
