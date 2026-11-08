# Benchmarks

Metric definitions: [docs/METRICS.md](../docs/METRICS.md). Shared code: [`metrics.py`](metrics.py) (recall, MRR, nDCG,
EM/F1, bootstrap CIs) and [`rankers.py`](rankers.py) (one interface for BM25, dense, hybrid RRF, cross-encoder, MAPLE
and teacher-attention rows). Results go to `benchmarks/results/` (gitignored); every call is also recorded in the run
registry (`runs/`, see [docs/DATA.md](../docs/DATA.md)). Run from the repository root.

| Script | Question | Needs | Output |
|---|---|---|---|
| `01_recall_enhanced.py` | Retrieval quality on held-out oracle sets: imitation (oracle recall) and dataset evidence (evidence recall, all-evidence, answer-in-context) | held-out oracle files; model optional | `retrieval_quality*.json` |
| `02_latency_scaling.py` | Search latency vs index size for linear / hierarchical / adaptive / **rerank** and a cosine reference, per device | model | `latency_results.json` |
| `02_latency_extreme.py` | Latency up to 10M blocks, synthetic embeddings | — | `latency_extreme.json` |
| `03_needle_robust.py` | Needle-in-a-haystack, held-out templates, N trials per cell | model optional | `needle_robustness.json`, heatmaps |
| `04_cost_analysis.py` | Token reduction vs full context | — | `cost_savings*.json` |
| `05_ablation_chunking.py` | Chunk size at a fixed context budget | model optional | `ablation_chunking.json` |
| `06_e2e_generation.py` | Legacy end-to-end benchmark with Gemini readers | `GEMINI_API_KEY` | `e2e_metrics_*.json` |
| `07_long_context.py` | Evidence retrieval vs pool size (HotpotQA + distractors) and on whole NarrativeQA stories | model optional | `long_context*.json` |
| `08_beir_rerank.py` | **BEIR top-100 re-ranking** (nDCG@10), with a published-BM25 sanity check | BEIR data | `beir_rerank.json` |
| `09_multihop_rerank.py` | **Multi-hop re-ranking** on HippoRAG's MuSiQue / 2Wiki / HotpotQA sets (Recall@2, Recall@5, All-Recall@5) | HippoRAG_2 data | `multihop_rerank.json` |
| `10_e2e_longbench.py` | **Reader-fixed end-to-end QA**: LongBench HotpotQA/NarrativeQA (F1) and NarrativeQA full stories (ROUGE-L, BLEU, METEOR) | Llama-3-8B, LongBench | `e2e_reader_fixed.json` |
| `11_efficiency.py` | **Latency, throughput and memory of every method** vs pool size, plus the quality-latency Pareto frontier | — | `efficiency.json`, `pareto.png` |
| `compare_published.py` | Our numbers next to published ones, only where the protocol matches | results of 08-10 | `docs/private/results/published_comparison.md` |
| `aggregate_results.py` | Tables across seeds/variants (`--root benchmarks/results/<gate>`) | results | `SUMMARY.md` / `.json` |
| `run_all.py` | Runs the classic suite (01-05, 07) | — | — |
| `run_matrix.py` | Train + evaluate across encoders, logs to W&B | `WANDB_API_KEY` | W&B runs |

"Model optional" scripts run the baselines when no checkpoint is given. Benchmarks 08-10 report a `protocol_ok` /
`protocol_mismatch` flag: if our reproduction of a paper's own baseline is off, the numbers are not compared with
papers.

```bash
python benchmarks/08_beir_rerank.py --rankers bm25 dense hybrid cross_encoder maple:main --model main=models/x.pt
python benchmarks/09_multihop_rerank.py --rankers bm25 dense hybrid cross_encoder
python benchmarks/10_e2e_longbench.py --tasks longbench_hotpotqa --limit 50
python benchmarks/11_efficiency.py --rankers bm25 dense hybrid maple:main --model main=models/x.pt
```
