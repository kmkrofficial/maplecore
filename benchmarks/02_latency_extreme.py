#!/usr/bin/env python3
"""
Benchmark 02 (Extreme): Latency Scaling to 10M Blocks
=======================================================

Simulates search latency for index sizes up to 10 MILLION blocks.
To fit in memory, we use:
  - Synthetic random embeddings (float16)
  - No text storage (only metadata)
  - Comparison of Linear vs Hierarchical vs Adaptive search

Output:
  - results/latency_extreme.json
  - results/latency_extreme_loglog.png
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleNet, MapleScanner
from maplecore.indexer import Block, Index
from benchmarks.config import RESULTS_DIR, DEFAULT_DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 10k, 100k, 1M, 5M, 10M
SCALING_COUNTS = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
EMBED_DIM = 384
NUM_TRIALS = 5
WARMUP = 2

# We'll use a dummy model for the scanner since we're testing search logic speed,
# not model inference speed (which is constant per query).
# However, Adaptive strategy uses the model to score reduced candidates.

class SyntheticIndex(Index):
    """
    A lightweight Index that generates random embeddings.
    """
    def __init__(self, num_blocks: int, dim: int = EMBED_DIM):
        self._num_blocks = num_blocks
        self._dim = dim
        self.chunk_size = 500
        self.source_path = "synthetic"
        
        # We don't store actual blocks to save RAM
        self.blocks = [Block(id=i, text="") for i in range(min(num_blocks, 100))] # Dummy

        logger.info(f"Allocating {num_blocks:,} x {dim} float16 tensor...")
        # Use float16 to save RAM (10M * 384 * 2 bytes = 7.6 GB)
        self.embeddings = torch.randn(num_blocks, dim, dtype=torch.float16)
        
        # Normalize for cosine similarity
        self.embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=1)

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def embedding_dim(self) -> int:
        return self._dim


def benchmark(scanner: MapleScanner, index: SyntheticIndex, strategy: str) -> float:
    """Measure median latency (ms) for a search strategy."""
    
    # Random query vector
    query_emb = torch.randn(EMBED_DIM, dtype=torch.float16)
    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)
    
    # If device is GPU, move query/index if they fit (likely index behaves as CPU for large scale)
    # For this benchmark, we keep index on CPU to simulate RAG-vector-db scenario 
    # where total index > VRAM. Scanner handles device movement if needed.
    
    # Warmup
    for _ in range(WARMUP):
        try:
            scanner.search(query_emb, index, k=5, strategy=strategy)
        except Exception:
            pass # Ignore warmup errors

    latencies = []
    for _ in range(NUM_TRIALS):
        start = time.perf_counter()
        scanner.search(query_emb, index, k=5, strategy=strategy)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    
    return np.median(latencies)


def plot_loglog(results: dict, output_path: Path):
    """Generate Log-Log plot of Latency vs Index Size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = results["counts"]
    strategies = results["strategies"]
    colors = {"linear": "#e74c3c", "hierarchical": "#3498db", "adaptive": "#2ecc71"}
    markers = {"linear": "o", "hierarchical": "s", "adaptive": "^"}

    for name, latencies in strategies.items():
        ax.loglog(counts, latencies, color=colors.get(name, "gray"), 
                  marker=markers.get(name, "o"), linewidth=2, label=name.capitalize())

    ax.set_xlabel("Index Size (blocks)", fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_title("Search Latency Scaling (Log-Log)", fontsize=15, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=12)
    
    # Annotate 10M points
    for name, latencies in strategies.items():
        if latencies:
            ax.text(counts[-1], latencies[-1], f"{latencies[-1]:.1f}ms", 
                    fontsize=9, ha="left", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved -> {output_path}")


def run():
    print("=" * 70)
    print("BENCHMARK 02 (EXTREME): 10 Million Block Latency")
    print("=" * 70)
    
    # Use CPU for large index storage simulation
    # (MapleScanner will use GPU for small batch model inference if available)
    # Note: MapleScanner logic currently assumes index.embeddings is likely on same device as query.
    # We will force CPU usage for this test to ensure we test SYSTEM RAM limits, not VRAM.
    model_device = "cpu" # Force CPU for FAIR comparison at 10M scale (won't fit 10M in VRAM)
    
    # Load dummy model (needed for Adaptive strategy's re-ranking)
    # We use a small initialized model to avoid loading weights
    model = MapleNet(input_dim=768, hidden_dim=128)
    model.eval()
    
    scanner = MapleScanner(model, device=model_device)
    
    strategies = ["linear", "hierarchical", "adaptive"]
    results = {
        "counts": [],
        "strategies": {s: [] for s in strategies}
    }
    
    for count in SCALING_COUNTS:
        print(f"\n--- Index Size: {count:,} blocks ---")
        
        try:
            index = SyntheticIndex(count)
            results["counts"].append(count)
            
            for strategy in strategies:
                latency = benchmark(scanner, index, strategy)
                results["strategies"][strategy].append(latency)
                print(f"  {strategy:<15} {latency:>8.2f} ms")
                
            del index
            import gc; gc.collect()
            
        except MemoryError:
            logger.error(f"OOM at {count:,} blocks!")
            break

    # Save
    json_path = RESULTS_DIR / "latency_extreme.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    # Plot
    png_path = RESULTS_DIR / "latency_extreme_loglog.png"
    plot_loglog(results, png_path)
    
    print("\n" + "=" * 70)
    print(f"Saved -> {json_path}")
    print(f"Chart -> {png_path}")
    print("=" * 70)

if __name__ == "__main__":
    run()
