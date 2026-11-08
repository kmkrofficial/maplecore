# MAPLE — Known Gaps (tracker)

Status of the project as of 2026-11-08, when work resumed after the v0.2.0-beta
commits (Feb 2026). MAPLE is an **experiment**, not a published paper. Each gap
has an ID so commits and notes can reference it.

Status legend: `open` · `fixed` · `partly fixed` · `in progress` · `blocked`. Last updated 2026-11-08.

## A. Environment & reproducibility

| ID | Gap | Status |
|---|---|---|
| A1 | No working Python environment on this machine (system Python 3.13 has no torch). | fixed — `.venv` (Python 3.12, torch 2.11+cu128) |
| A2 | No trained weights (`models/` is gitignored and empty). Nothing in the repo runs end-to-end. | open — needs oracle data, then `train_generalist.py` |
| A3 | No oracle data (`data/` is gitignored and empty). | in progress — public datasets downloaded; oracle generation pending |
| A4 | Oracle teacher `meta-llama/Meta-Llama-3-8B-Instruct` is gated on Hugging Face; needs the user's HF login. | fixed — HF login done; access confirmed |
| A5 | No lockfile / pinned versions; `requirements.txt` and `setup.py` disagree (onnx is a hard dep in one, optional in the other). | in progress — extras aligned; `requirements-lock.txt` added |
| A6 | No unit tests, no CI. | partly fixed — pytest suite added; no CI yet |
| A7 | Many scripts must be run from the repo root because of relative paths (`models/...`, `data/...`) and `sys.path` hacks. | open |

## B. Correctness bugs in the SDK (`maplecore/`)

| ID | Gap | Status |
|---|---|---|
| B1 | Hierarchical/adaptive search use `num_blocks // chapter_size` chapters, so the trailing `num_blocks % chapter_size` blocks are **never searchable**. The needle benchmark hides this by padding text with dots. | fixed — `ceil` chapters; regression test |
| B2 | `_score_blocks` deletes `logits`, which is undefined on the ONNX path → `UnboundLocalError`. | fixed |
| B3 | `MapleONNXRunner.predict` uses `.squeeze()`, giving a 0-d array for batch size 1 → `torch.cat` fails. | fixed |
| B4 | `MapleTrainer` keeps "best" weights via `state_dict().copy()` (shallow) → actually returns the last epoch. | fixed — `deepcopy`; regression test |
| B5 | `MapleNet.load`, `MapleScanner`, `MapleIndexer`, `MapleTrainer` default to `device="cuda"` → crash on CPU-only machines. | fixed — `resolve_device()` |
| B6 | Adaptive search computes entropy but never uses it; the confidence threshold is applied to mean-pooled chapter embeddings the MLP was never trained on. | partly fixed — optional `entropy_threshold`; chapter-approximation gap measured by benchmark 07 |
| B7 | `ContextSynthesizer` dedups with Python `hash()` (not stable across processes); release notes claim SHA256. | fixed — SHA-256 |
| B8 | Orchestrator loop `while hops <= max_hops` allows `max_hops + 1` LLM calls before the fallback call. | fixed — at most `max_hops` searches |
| B9 | Chapter embeddings are recomputed on every query (O(N) Python loop) instead of at index time. | fixed — cached per chapter size |
| B10 | `Index` persistence does not store the embedding model name/dim → silent mismatch when loading with a different encoder. | fixed — encoder name saved, mismatch warning |
| B11 | Fixed-width character chunking splits words/sentences; no overlap option. | partly fixed — `overlap` option (default 0 keeps oracle compatibility) |

| B12 | `MapleNet.load` inferred `input_dim` but not `hidden_dim` → any non-128 checkpoint failed to load. (found 2026-11-08) | fixed |

## C. Training / data pipeline

| ID | Gap | Status |
|---|---|---|
| C1 | Oracle covers only the first ~1,536 tokens (~12 blocks) of each NarrativeQA document — not the long-document setting MAPLE is pitched for. | in progress — attention hooks allow longer `--max-tokens`; benchmark 07 evaluates full stories |
| C2 | Oracle label = last-layer, head-averaged attention of the final token. Known to be noisy (attention sinks, recency bias); never validated against answer location. | in progress — all-layer scores + gold evidence recorded; METRICS §3 decision rule |
| C3 | `train_generalist.py` assigns positives the wrong `block_id` (`blocks[i]` indexed by position in `pos_embs`) and all negatives `-1`, so validation Recall@5 is meaningless/inflated. | fixed — one row per in-context block with real ids |
| C4 | Train/val split is by question string; synthetic needle questions are near-identical, so needle samples leak across splits. | fixed — `group` key per sample |
| C5 | Synthetic needle training uses the same template ("secret passkey") as the needle benchmark → test leakage. | fixed — disjoint train / held-out needle templates |
| C6 | No held-out test split; best-epoch selection uses the same val set that is reported. | fixed — held-out oracle splits (NarrativeQA test, HotpotQA validation) |
| C7 | Three different training scripts (`train_maple`, `train_generalist`, `train_needle_finetune`) write three different weight files; benchmarks load different ones (`maple_generalist.pth`, `maple_v1.pth`, `maple_bge_small_en_v1.5.pth`). | fixed — single canonical trainer; `train_needle_finetune.py` removed |
| C8 | `generate_oracle_hotpot.py` used `trust_remote_code=True` for a dataset that no longer needs it (newer `datasets` rejects it). | fixed |

## D. Benchmarks / evaluation

| ID | Gap | Status |
|---|---|---|
| D1 | `run_all.py` imports modules that no longer exist (`01_recall_narrativeqa`, `03_needle_in_haystack`). | fixed |
| D2 | `04_cost_analysis.py` imports `download_gutenberg_book`, which does not exist. | fixed |
| D3 | E2E SQuAD eval: contexts are ~700 chars (~2 blocks), so "top-5" sends the whole paragraph — it measures the LLM, not retrieval. | fixed — `--haystack-chars` padding |
| D4 | E2E HotpotQA eval: ~10 paragraphs → top-5 is ~half the context. Retrieval is barely stressed. | fixed — `--haystack-chars` padding |
| D5 | Recall benchmark on NarrativeQA evaluates against the same oracle-style labels on ~12-block candidate sets — not long-context. | fixed — benchmark 01 scores against gold evidence; benchmark 07 for long pools |
| D6 | Needle benchmark: one sample per (length, depth) cell, one fixed phrasing, pads text to chapter boundaries. | fixed |
| D7 | No standard IR metrics (MRR, nDCG), no confidence intervals, no seeds recorded. | fixed — MRR, nDCG, bootstrap CIs, paired p-values, seeds |
| D8 | No answer-containment metric (does retrieved context contain the gold answer string?). | fixed — `answer_in@k` |
| D9 | Latency benchmarks use synthetic random embeddings; no end-to-end (encode + search) latency. | partly fixed — 07 reports per-query latency on real embeddings; encode+search total still open |
| D10 | E2E benchmark hardcodes preview Gemini model names and a 4-bit Llama that needs the gated download. | partly fixed — `--gemini-models`, `--with-llama`; see D12 |

| D11 | HotpotQA long-context distractors are random paragraphs from other questions — topically unrelated and easy (dense recall barely drops with pool size). Need similarity-mined distractors. | open |
| D12 | `google.generativeai` (used by benchmark 06) is end-of-life; migrate to `google.genai`. | open |
| D13 | Teacher attention: only ~19% of the final token's last-layer attention lands on context blocks in a smoke test (rest on BOS / prompt) — supports C2. | open |

## E. Documentation & claims

| ID | Gap | Status |
|---|---|---|
| E1 | Headline numbers disagree: README ">95% recall", package docstring "71% Recall@5", final report "98% Recall@1". None are reproducible from the repo. | fixed — claims removed; archived |
| E2 | README claims publication in "a reputed journal" with a placeholder citation; no paper exists. | fixed |
| E3 | Needle ">90% success" is attributed to "spatial dimension relationships"; the actual change was benchmark padding + matching synthetic training data. | fixed — archived with warning |
| E4 | Docs reference files that don't exist: `scanner.py`, `model.py`, `quickstart.py`, `demo_notebook.ipynb`, `verify_linear.py`, `LICENSE`, `data/README.md`, `models/README.md`, paper figures. | fixed — docs rewritten; LICENSE added |
| E5 | README quick start is broken (`results.top_k`, default CUDA device). | fixed |
| E6 | `PR_DESCRIPTION.md` and `final_report.md` are stale artifacts in the repo root. | fixed — moved to `docs/archive/` |
| E7 | Inconsistent naming: "Memory-Aware Predictive Loading Engine" vs "Model-Adaptive Projection of Latent Embeddings"; repo URL `kmkrworks/MAPLE` vs `kmkrofficial`. | fixed |

## F. Missing features (product side)

| ID | Gap | Status |
|---|---|---|
| F1 | No multi-document index / incremental add. | open |
| F2 | Orchestrator only lives inside the benchmark; `Maple` client has no `answer()` / LLM integration. | open |
| F3 | No published weights (e.g. on Hugging Face Hub) so the SDK works out of the box. | open |
