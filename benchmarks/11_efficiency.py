#!/usr/bin/env python3
"""
Benchmark 11: efficiency across methods
=======================================

Per-query latency (p50 / p95), throughput, peak memory and a quality-latency Pareto frontier for every ranker, over
pool sizes 20 ... 100,000 blocks, on GPU and CPU. GPU work is bracketed by ``cuda.synchronize``. A ranker whose first
query takes longer than ``--max-seconds-per-query`` at some pool size is recorded as ``"infeasible"`` and skipped for
larger pools (e.g. a cross-encoder at 100k blocks).

``total_ms`` = query encoding + ranking (the cost a user pays per question); ``rank_ms`` = ranking only.
Quality for the Pareto plot comes from ``--quality-file`` ({ranker name: evidence recall}), e.g. built from the
benchmark-01 results.

Usage:
    python benchmarks/11_efficiency.py --rankers bm25 dense hybrid cross_encoder maple:main --model main=models/x.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import BGE_MODEL_NAME, RESULTS_DIR
from benchmarks.rankers import build_rankers

logger = logging.getLogger(__name__)

POOL_SIZES = (20, 100, 1_000, 10_000, 100_000)
DEFAULT_RANKERS = ("bm25", "dense", "hybrid", "cross_encoder", "maple:main")


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_calls(fn: Callable[[], object], n: int = 100, sync: bool = True, device: str = "cuda") -> Dict[str, float]:
    """Wall-clock p50 / p95 (ms) of ``n`` calls; GPU-synchronised when ``sync``."""
    samples = []
    for _ in range(n):
        if sync:
            _sync(device)
        t0 = time.perf_counter()
        fn()
        if sync:
            _sync(device)
        samples.append((time.perf_counter() - t0) * 1000)
    return {"p50": float(np.percentile(samples, 50)), "p95": float(np.percentile(samples, 95))}


def pareto_frontier(points: Dict[str, Tuple[float, float]]) -> List[str]:
    """Names not dominated by another point; points are (latency, quality): lower latency and higher quality win.
    Sorted by latency, then name."""
    names = []
    for a, (la, qa) in points.items():
        dominated = any((lb <= la and qb >= qa) and (lb < la or qb > qa) for b, (lb, qb) in points.items() if b != a)
        if not dominated:
            names.append(a)
    return sorted(names, key=lambda n: (points[n][0], n))


def synthetic_pool(n: int, dim: int = 384, seed: int = 0) -> Tuple[List[str], torch.Tensor]:
    """Deterministic stand-in pool (used by tests): short texts and random embeddings."""
    g = torch.Generator().manual_seed(seed)
    texts = [f"document {i} discusses topic{i % 97} with several filler words about subject{i % 13}" for i in range(n)]
    return texts, torch.randn(n, dim, generator=g)


class ScannerRanker:
    """MAPLE's own search strategies (rerank / linear / ...) behind the ranker interface; the Index is built once."""

    def __init__(self, model, strategy: str, name: str):
        self.model, self.strategy, self.name = model, strategy, name
        self._scanner = self._index = None

    def prepare(self, texts, embs) -> None:
        from maplecore import MapleScanner
        from maplecore.indexer import Block, Index

        device = str(embs.device).split(":")[0]
        self._index = Index(blocks=[Block(i, t) for i, t in enumerate(texts)], embeddings=embs)
        import copy
        self._scanner = MapleScanner(copy.deepcopy(self.model), device=device)

    def rank(self, question, texts, q_emb, block_embs, k=None, sample=None):
        return self._scanner.search(q_emb, self._index, k=k or 5, strategy=self.strategy).block_ids


def run_one(ranker, pool_sizes: Sequence[int], device: str, n_queries: int = 100, max_seconds_per_query: float = 10.0,
            pool_factory: Optional[Callable[[int], Tuple[List[str], torch.Tensor]]] = None,
            encode_query: Optional[Callable[[str], torch.Tensor]] = None, k: int = 5) -> dict:
    """Time ``ranker`` on growing pools. Returns {str(pool): stats | "infeasible"}; stops at the first infeasible pool."""
    pool_factory = pool_factory or synthetic_pool
    out: dict = {}
    questions = [f"What is the significance of topic{i % 97} in subject{i % 13}?" for i in range(n_queries)]
    for n in pool_sizes:
        texts, embs = pool_factory(n)
        embs = embs.to(device)
        if hasattr(ranker, "prepare"):
            ranker.prepare(texts, embs)
        q_embs = [encode_query(q) if encode_query else None for q in questions[:2]]
        q0 = q_embs[0] if q_embs[0] is not None else embs[0]

        ranker.rank(questions[0], texts, q0, embs, k=k)  # untimed warm-up: lazy model loading, CUDA kernels
        _sync(device)
        t0 = time.perf_counter()  # feasibility probe
        ranker.rank(questions[0], texts, q0, embs, k=k)
        _sync(device)
        if time.perf_counter() - t0 > max_seconds_per_query:
            out[str(n)] = "infeasible"
            break

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rank_ms = time_calls(lambda: ranker.rank(questions[1 % len(questions)], texts, q0, embs, k=k), n_queries,
                             device=device)
        enc_ms = (time_calls(lambda: encode_query(questions[0]), min(n_queries, 50), device=device)
                  if encode_query else {"p50": 0.0, "p95": 0.0})
        total = {"p50": rank_ms["p50"] + enc_ms["p50"], "p95": rank_ms["p95"] + enc_ms["p95"]}
        try:
            import psutil
            ram_mb = psutil.Process().memory_info().rss / 2 ** 20
        except Exception:
            ram_mb = None
        out[str(n)] = {"rank_ms": rank_ms, "encode_ms": enc_ms, "total_ms": total,
                       "qps": 1000.0 / max(total["p50"], 1e-9),
                       "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                        if device == "cuda" and torch.cuda.is_available() else None),
                       "ram_mb": ram_mb}
    return out


def _real_pool_factory(indexer, max_pool: int):
    """Real text blocks (Gutenberg, 500 chars) with real BGE embeddings, built once for the largest pool."""
    from benchmarks.data_loader import build_large_corpus

    text = build_large_corpus(max_pool * 500)
    texts = [b.text for b in indexer.chunk_text(text, 500)][:max_pool]
    embs = indexer.encode_texts(texts, show_progress=True).cpu()
    return lambda n: (texts[:n], embs[:n])


def run(rankers: Sequence[str] = DEFAULT_RANKERS, pool_sizes: Sequence[int] = POOL_SIZES,
        devices: Sequence[str] = ("cuda", "cpu"), n_queries: int = 100, max_seconds_per_query: float = 10.0,
        model_paths: Optional[Dict[str, Path]] = None, quality_file: Optional[Path] = None,
        embedding_model: str = BGE_MODEL_NAME, pareto_pool: int = 1000) -> dict:
    from maplecore import MapleIndexer, load_model

    devices = [d for d in devices if d != "cuda" or torch.cuda.is_available()]
    model_paths = model_paths or {}
    objs = list(build_rankers(list(rankers), model_paths))
    for key, path in model_paths.items():  # MAPLE's own two-stage search, timed through the scanner
        objs.append(ScannerRanker(load_model(path, device="cpu"), "rerank", f"maple_rerank:{key}"))
    indexer = MapleIndexer(model_name=embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
    factory = _real_pool_factory(indexer, max(pool_sizes))
    enc = lambda q: indexer.encode_query(q)  # noqa: E731

    results: dict = {"config": {"pool_sizes": list(pool_sizes), "devices": devices, "n_queries": n_queries,
                                "max_seconds_per_query": max_seconds_per_query}, "rankers": {}}
    for r in objs:
        results["rankers"][r.name] = {}
        for dev in devices:
            if dev == "cuda" and r.name == "bm25":
                continue
            logger.info(f"{r.name} on {dev}")
            res = run_one(r, pool_sizes, dev, n_queries, max_seconds_per_query, factory,
                          lambda q, d=dev: enc(q).to(d))
            results["rankers"][r.name][dev] = res
            print(f"  {r.name:<28}{dev:<6}" + "  ".join(
                f"{n}:{v if isinstance(v, str) else format(v['total_ms']['p50'], '.2f') + 'ms'}" for n, v in res.items()))

    if quality_file:
        quality = json.loads(Path(quality_file).read_text(encoding="utf-8"))
        dev = devices[0]
        pts = {}
        for name, q in quality.items():
            lat = results["rankers"].get(name, {}).get(dev, {}).get(str(pareto_pool))
            if isinstance(lat, dict):
                pts[name] = (lat["total_ms"]["p50"], q)
        results["pareto"] = {"device": dev, "pool": pareto_pool, "points": pts, "frontier": pareto_frontier(pts)}
        _plot_pareto(pts, results["pareto"]["frontier"], RESULTS_DIR / "pareto.png", dev, pareto_pool)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "efficiency.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {out}")
    return results


def _plot_pareto(points: dict, frontier: list, path: Path, device: str, pool: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (lat, q) in points.items():
        ax.scatter(lat, q * 100, s=70, color="#d62728" if name in frontier else "#7f7f7f")
        ax.annotate(name, (lat, q * 100), textcoords="offset points", xytext=(6, 4), fontsize=8)
    fr = sorted((points[n] for n in frontier))
    ax.plot([p[0] for p in fr], [p[1] * 100 for p in fr], color="#d62728", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(f"p50 query latency, ms ({device}, pool={pool:,} blocks, encode + rank)")
    ax.set_ylabel("evidence recall@5 (%)")
    ax.set_title("Quality vs latency (red = Pareto frontier)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Efficiency across methods")
    ap.add_argument("--rankers", nargs="+", default=list(DEFAULT_RANKERS))
    ap.add_argument("--model", action="append", default=[], metavar="KEY=PATH")
    ap.add_argument("--pool-sizes", type=int, nargs="+", default=list(POOL_SIZES))
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--max-seconds-per-query", type=float, default=10.0)
    ap.add_argument("--quality-file", type=Path, default=None)
    ap.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    a = ap.parse_args()
    paths = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.model}
    from scripts.runlog import record_outputs, start_run
    with start_run("bench", "11_efficiency", vars(a)) as _run:
        _res = run(a.rankers, a.pool_sizes, a.devices, a.n_queries, a.max_seconds_per_query, paths, a.quality_file,
                   a.embedding_model)
        record_outputs(_run, RESULTS_DIR / "efficiency.json", RESULTS_DIR / "pareto.png")
