#!/usr/bin/env python3
"""
Benchmark 02: Latency Scaling
==============================

Measures search latency across increasing index sizes:
  [10K, 50K, 100K, 500K, 1M] blocks

Tests Linear, Hierarchical, and Adaptive strategies.

Output: results/latency_scaling.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleNet, MapleIndexer, MapleScanner
from maplecore.indexer import Block, Index
from benchmarks.config import (
    RESULTS_DIR, MODEL_PATH, DEFAULT_DEVICE, CHUNK_SIZE,
    SCALING_BLOCK_COUNTS,
)
from benchmarks.data_loader import build_large_corpus

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_WARMUP = 3
NUM_TRIALS = 10


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


def run():
    """Run the latency scaling benchmark."""
    print("=" * 70)
    print("BENCHMARK 02: Latency Scaling")
    print("=" * 70)

    # ---- Load components ----
    indexer = MapleIndexer(device=DEFAULT_DEVICE)
    model = MapleNet.load(MODEL_PATH, device=DEFAULT_DEVICE)
    scanner = MapleScanner(model, device=DEFAULT_DEVICE)

    # Prepare a query
    query_emb = indexer.encode_query("What is the significance of the red herring?")

    strategies = ["linear", "hierarchical", "adaptive"]
    results = {
        "block_counts": [],
        "strategies": {s: [] for s in strategies},
    }

    for target_blocks in SCALING_BLOCK_COUNTS:
        print(f"\n--- {target_blocks:,} blocks ---")

        index = _create_synthetic_index(indexer, target_blocks)
        actual = index.num_blocks
        results["block_counts"].append(actual)

        for strategy in strategies:
            latency = _benchmark_strategy(scanner, query_emb, index, strategy)
            results["strategies"][strategy].append(latency)
            print(f"  {strategy:<15} {latency:>8.2f} ms")

        # Free memory
        del index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Save results ----
    output_path = RESULTS_DIR / "latency_scaling.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved -> {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run()
