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

    device = DEFAULT_DEVICE
    print(f"Running Latency Benchmark on {device}...")
    
    # Init Model (Outside loop to save time)
    model = MapleNet.load(str(MODEL_PATH), device=device)
    indexer = MapleIndexer(device=device)
    scanner = MapleScanner(model, device=device)
    
    counts = SCALING_BLOCK_COUNTS # Use predefined scaling counts
    metrics = []
    
    for n in counts:
        print(f"--- Scale N={n} ---")
        
        with HardwareMonitor(interval=0.1) as mon:
            # Create Index
            index = _create_synthetic_index(indexer, n) # Use existing synthetic index creation
            
            # Prepare a query
            query_emb = indexer.encode_query("What is the significance of the red herring?")

            # Linear Search Profile
            start = time.perf_counter()
            for _ in range(NUM_TRIALS): # Use NUM_TRIALS for consistency
                scanner.search(query_emb, index, k=5, strategy="linear")
            lin_time = (time.perf_counter() - start) / NUM_TRIALS * 1000
            
            # Adaptive Search Profile
            start = time.perf_counter()
            for _ in range(NUM_TRIALS): # Use NUM_TRIALS for consistency
                scanner.search(query_emb, index, k=5, strategy="adaptive")
            adp_time = (time.perf_counter() - start) / NUM_TRIALS * 1000
            
        stats = mon.get_stats()
        
        metrics.append({
            "n_blocks": n,
            "linear_time_ms": lin_time,
            "adaptive_time_ms": adp_time,
            "speedup": lin_time / adp_time if adp_time > 0 else 0.0,
            "hardware_usage": stats
        })

        # Free memory
        del index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Wrap Result
    # Use empty monitor for global stats since we did per-scale
    dummy_mon = HardwareMonitor() 
    final_output = wrap_result(metrics, dummy_mon)
    
    out_file = RESULTS_DIR / "latency_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {out_file}")
    print("=" * 70)

    return final_output


if __name__ == "__main__":
    run()
