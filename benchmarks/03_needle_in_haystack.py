#!/usr/bin/env python3
"""
Benchmark 03: Needle in a Haystack
====================================

Inserts a known "needle" (random UUID fact) at various depths [0%–100%]
within contexts of varying size, then queries MAPLE to check if the
needle block is retrieved.

Output: results/needle_heatmap.json
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleNet, MapleIndexer, MapleScanner
from maplecore.indexer import Index
from benchmarks.config import (
    RESULTS_DIR, MODEL_PATH, DEFAULT_DEVICE, CHUNK_SIZE,
    NEEDLE_DEPTHS, NEEDLE_CONTEXT_SIZES,
)
from benchmarks.data_loader import build_large_corpus

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_TRIALS = 5  # Repeat each (depth, size) combo for robustness


def _inject_needle(text: str, needle: str, depth: float) -> str:
    """Insert needle at a specific depth percentage in the text."""
    pos = int(len(text) * depth)
    # Find nearest whitespace boundary
    while pos < len(text) and not text[pos].isspace():
        pos += 1
    return text[:pos] + f"\n\n{needle}\n\n" + text[pos:]


def run():
    """Run the needle-in-a-haystack benchmark."""
    print("=" * 70)
    print("BENCHMARK 03: Needle in a Haystack")
    print("=" * 70)

    # ---- Load components ----
    indexer = MapleIndexer(device=DEFAULT_DEVICE)
    model = MapleNet.load(MODEL_PATH, device=DEFAULT_DEVICE)
    scanner = MapleScanner(model, device=DEFAULT_DEVICE)

    # Build a large base corpus
    max_size = max(NEEDLE_CONTEXT_SIZES)
    logger.info(f"Building base corpus ({max_size:,} chars)...")
    base_corpus = build_large_corpus(max_size)

    # Results: heatmap[context_size_idx][depth_idx] = accuracy
    heatmap = []

    for size_idx, context_size in enumerate(NEEDLE_CONTEXT_SIZES):
        print(f"\n--- Context: {context_size:,} chars ---")
        row = []

        haystack = base_corpus[:context_size]

        for depth_idx, depth in enumerate(NEEDLE_DEPTHS):
            hits = 0

            for trial in range(NUM_TRIALS):
                # Generate unique needle
                secret_uuid = str(uuid.uuid4())
                needle_text = (
                    f"The secret verification code is {secret_uuid}. "
                    f"Remember this code: {secret_uuid}."
                )

                # Inject needle
                modified_text = _inject_needle(haystack, needle_text, depth)

                # Index
                blocks = indexer.chunk_text(modified_text, chunk_size=CHUNK_SIZE)
                embeddings = indexer.encode_blocks(blocks, show_progress=False)
                index = Index(blocks=blocks, embeddings=embeddings, chunk_size=CHUNK_SIZE)

                # Query
                query = f"What is the secret verification code?"
                query_emb = indexer.encode_query(query)
                result = scanner.search(query_emb, index, k=5, strategy="adaptive")

                # Check if any retrieved block contains the UUID
                for bid in result.block_ids:
                    if secret_uuid in blocks[bid].text:
                        hits += 1
                        break

            accuracy = hits / NUM_TRIALS
            row.append(accuracy)

            marker = "+" if accuracy >= 0.8 else ("~" if accuracy >= 0.4 else "-")
            print(f"  Depth {depth:>5.0%}: {accuracy:.0%} {marker}")

        heatmap.append(row)

    # ---- Save results ----
    results = {
        "depths": NEEDLE_DEPTHS,
        "context_sizes": NEEDLE_CONTEXT_SIZES,
        "heatmap": heatmap,
        "num_trials": NUM_TRIALS,
        "chunk_size": CHUNK_SIZE,
    }

    output_path = RESULTS_DIR / "needle_heatmap.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved -> {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run()
