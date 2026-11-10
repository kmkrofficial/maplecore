# Scripts

Run from the repository root. Data layout and schemas: [docs/DATA.md](../docs/DATA.md).

## Data and labels
| Script | Purpose |
|---|---|
| `download_data.py` | Datasets, Gutenberg books, encoders, BEIR, re-rankers and (with `--only teacher`) the gated teacher LLM; writes `data/MANIFEST.json`, fails loudly on partial downloads |
| `oracle_common.py` | Teacher loading, prompt construction, fast last-row attention capture (batched, optional per-head mass and calibration pass), `HeadStore`, evidence helpers |
| `generate_oracle.py` / `generate_oracle_hotpot.py` | NarrativeQA / HotpotQA oracle labels → `data/oracle/*.json`; `--save-heads`, `--calibration-query`, `--attn-impl {fast,eager}`, `--resume` |
| `run_oracle_all.sh` | Regenerates every oracle file for both teacher budgets |
| `compare_oracles.py` | Sample-by-sample agreement of two oracle files (fast vs eager attention) |
| `select_heads.py` | QRHead-style selection of query-focused heads on a train split |
| `targets.py` | Soft training targets (last / all-layer / calibrated / heads / top-5 / gold) |
| `embedding_cache.py` | Encodes oracle files once per encoder (`data/cache/`) |

## Training
| Script | Purpose |
|---|---|
| `train_v2.py` | **Canonical trainer**: residual scorer, listwise KL on soft targets, optional cross-document distractors, from cached embeddings |
| `train_biencoder.py` | MAPLE-Embed: distils teacher attention into the sentence encoder itself |
| `train_generalist.py` / `train_maple.py` | v0.2-era pairwise (BCE) trainers, kept for the v1 baseline and `run_matrix.py` |

## Experiments and tracking
| Script | Purpose |
|---|---|
| `run_gate.sh GATE` | Trains a grid of variants (3 seeds x 2 teacher budgets), evaluates each, aggregates (`gate_a`) |
| `run_experiment.sh` | The original v1 train + benchmark pipeline |
| `runlog.py` | Run registry: logs, config, metrics, per-item outputs, checksums, model cards (`python scripts/runlog.py list`) |
