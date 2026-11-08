# Benchmarks

Metric definitions: [docs/METRICS.md](../docs/METRICS.md). Shared implementation:
[`metrics.py`](metrics.py). Results go to `benchmarks/results/` (gitignored).
Run from the repository root.

| Script | Question (METRICS §) | Needs | Output |
|---|---|---|---|
| `01_recall_enhanced.py` | Label validity, imitation, evidence on held-out oracle sets (§3–5) | held-out oracle files; model optional | `retrieval_quality.json` |
| `02_latency_scaling.py` | Search latency vs index size, real embeddings (§7) | model | `latency_results.json` |
| `02_latency_extreme.py` | Latency up to 10M blocks, synthetic embeddings (§7) | — | `latency_extreme.json` |
| `03_needle_robust.py` | Needle-in-a-haystack, held-out templates, N trials/cell (§6) | model optional | `needle_robustness.json`, heatmaps |
| `04_cost_analysis.py` | Token reduction vs full context (§7) | — | `cost_savings*.json` |
| `05_ablation_chunking.py` | Chunk size at fixed context budget (§6) | model optional | `ablation_chunking.json` |
| `06_e2e_generation.py` | Downstream EM / F1 / judge with padded contexts (§8) | `GEMINI_API_KEY`; model | `e2e_metrics_*.json` |
| `07_long_context.py` | Evidence retrieval vs pool size; NarrativeQA full stories (§6–7) | model optional | `long_context*.json` |
| `run_all.py` | Runs 01–05 and 07 | — | — |
| `run_matrix.py` | Train + evaluate across encoders, logs to W&B | oracle train files, `WANDB_API_KEY` | W&B runs |

"Model optional" scripts run the BM25 / dense baselines when no checkpoint exists.

```bash
python -m benchmarks.run_all
python -m benchmarks.run_all --only 07
python benchmarks/07_long_context.py --datasets hotpotqa --pool-sizes 10 100 1000 5000
```
