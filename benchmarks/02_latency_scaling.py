#!/usr/bin/env python3
"""
Benchmark 02: Latency Scaling
==============================

Measures search latency across increasing index sizes:
  [10K, 50K, 100K, 500K, 1M] blocks

Tests Linear, Hierarchical, and Adaptive strategies.

Output: benchmarks/results/latency_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleIndexer, MapleScanner, load_model
from maplecore.indexer import Index
from benchmarks.config import (
    RESULTS_DIR, MODEL_PATH, DEFAULT_DEVICE, CHUNK_SIZE,
    SCALING_BLOCK_COUNTS,
)
from benchmarks.data_loader import build_large_corpus
from benchmarks.profiler import HardwareMonitor, wrap_result

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_WARMUP = 3
NUM_TRIALS = 20


def _create_synthetic_index(indexer: MapleIndexer, num_blocks: int) -> Index:
    """
    Create a synthetic index with the given number of blocks.

    Uses real BGE embeddings from Gutenberg text for authenticity.
    """
    # Estimate chars needed (CHUNK_SIZE chars per block)
    target_chars = num_blocks * CHUNK_SIZE
    logger.info(f"Building corpus for {num_blocks:,} blocks ({target_chars:,} chars)...")

    text = build_large_corpus(target_chars)

    # Chunk the text
    blocks = indexer.chunk_text(text, chunk_size=CHUNK_SIZE)
    blocks = blocks[:num_blocks]

    if len(blocks) < num_blocks:
        logger.warning(f"Only got {len(blocks)} blocks, target was {num_blocks}")

    # Generate embeddings
    logger.info(f"Encoding {len(blocks):,} blocks...")
    embeddings = indexer.encode_blocks(blocks)

    return Index(
        blocks=blocks,
        embeddings=embeddings,
        chunk_size=CHUNK_SIZE,
    )


def _benchmark_strategy(
    scanner: MapleScanner,
    query_emb: torch.Tensor,
    index: Index,
    strategy: str,
    k: int = 5,
) -> float:
    """Run a single strategy benchmark, return median latency in ms."""
    # Warmup
    for _ in range(NUM_WARMUP):
        scanner.search(query_emb, index, k=k, strategy=strategy)

    # Timed runs
    latencies = []
    for _ in range(NUM_TRIALS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        scanner.search(query_emb, index, k=k, strategy=strategy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return float(sorted(latencies)[NUM_TRIALS // 2])  # median


STRATEGIES = ("linear", "hierarchical", "adaptive", "rerank")


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _percentiles(samples: List[float]) -> dict:
    samples = sorted(samples)
    return {"p50": samples[len(samples) // 2], "p95": samples[min(len(samples) - 1, int(0.95 * len(samples)))]}


def _time(fn, device: str, trials: int) -> dict:
    out = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out.append((time.perf_counter() - t0) * 1000)
    return _percentiles(out)


def run(devices: Optional[Sequence[str]] = None, counts: Optional[Sequence[int]] = None,
        model_path: Optional[Path] = None, trials: int = NUM_TRIALS) -> dict:
    """Search latency vs index size for every strategy, on each device. Embeddings are encoded once per size."""
    print("=" * 70)
    print("BENCHMARK 02: Latency Scaling")
    print("=" * 70)

    devices = [d for d in (devices or ["cuda", "cpu"]) if d != "cuda" or torch.cuda.is_available()]
    counts = list(counts or SCALING_BLOCK_COUNTS)
    model = load_model(model_path or MODEL_PATH, device="cpu")
    indexer = MapleIndexer(device=DEFAULT_DEVICE)  # encode on the GPU when there is one; indexes are moved per device
    query_emb = indexer.encode_query("What is the significance of the red herring?")

    metrics = []
    with HardwareMonitor(interval=0.5) as mon:
        for n in counts:
            print(f"--- Scale N={n:,} ---")
            base = _create_synthetic_index(indexer, n)
            for dev in devices:
                index = Index(blocks=base.blocks, embeddings=base.embeddings.to(dev), chunk_size=base.chunk_size)
                scanner = MapleScanner(copy.deepcopy(model), device=dev)
                q = query_emb.to(dev)
                for strat in STRATEGIES:                       # warm-up: kernels, chapter embeddings, normalised copy
                    scanner.search(q, index, k=5, strategy=strat)
                timings = {st: _time(lambda st=st: scanner.search(q, index, k=5, strategy=st), dev, trials)
                           for st in STRATEGIES}
                normed = index.normalized_embeddings()
                qn = torch.nn.functional.normalize(q.float(), dim=0)
                timings["dense"] = _time(lambda: torch.topk(normed @ qn, 5), dev, trials)  # cosine reference
                for st, t in timings.items():
                    print(f"  {dev:<4} {st:<12} p50={t['p50']:8.2f} ms  p95={t['p95']:8.2f} ms")
                metrics.append({
                    "n_blocks": n, "device": dev, "latency_ms": timings,
                    "linear_time_ms": timings["linear"]["p50"], "adaptive_time_ms": timings["adaptive"]["p50"],
                    "hierarchical_time_ms": timings["hierarchical"]["p50"], "rerank_time_ms": timings["rerank"]["p50"],
                    "speedup": timings["linear"]["p50"] / max(timings["adaptive"]["p50"], 1e-9),
                    "rerank_speedup": timings["linear"]["p50"] / max(timings["rerank"]["p50"], 1e-9),
                })
                index = scanner = normed = None  # free the per-device copy before the next device
                if dev == "cuda":
                    torch.cuda.empty_cache()
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_output = wrap_result(metrics, mon)
    out_file = RESULTS_DIR / "latency_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {out_file}")
    print("=" * 70)
    return final_output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Latency scaling of the search strategies")
    ap.add_argument("--devices", nargs="+", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--counts", type=int, nargs="+", default=None)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--trials", type=int, default=NUM_TRIALS)
    a = ap.parse_args()
    from scripts.runlog import record_outputs, start_run
    with start_run("bench", "02_latency", vars(a)) as _run:
        run(a.devices, a.counts, a.model, a.trials)
        record_outputs(_run, RESULTS_DIR / 'latency_results.json')
