#!/usr/bin/env python3
"""
Benchmark 03 (Robust): Needle in a Haystack
=============================================

Improved needle-in-a-haystack test with:
  1. Chunk-boundary-safe insertion (padding to prevent splitting)
  2. Variable needle types (UUID, natural language facts)
  3. PassKey-style heatmap visualization (a la Llama-3 tech report)

Output:
  - results/needle_robust.json
  - results/needle_heatmap_robust.png
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleNet, MapleIndexer, MapleScanner
from maplecore.indexer import Index
from benchmarks.config import DATA_DIR, RESULTS_DIR
from benchmarks.data_loader import build_large_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500
MODEL_PATH = Path("models/maple_v1.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRIALS = 5

DEPTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONTEXT_SIZES = [5_000, 10_000, 25_000, 50_000, 100_000]

# ---------------------------------------------------------------------------
# Needle Types
# ---------------------------------------------------------------------------
NEEDLE_TYPES = {
    "uuid": {
        "template": "The secret verification code is {value}. Remember this code: {value}.",
        "query": "What is the secret verification code?",
        "gen_value": lambda: str(uuid.uuid4()),
    },
    "city": {
        "template": "The secret city is {value}. The hidden capital that nobody knows about is {value}.",
        "query": "What is the secret city?",
        "gen_value": lambda: "Atlantis",
    },
    "date": {
        "template": "The meeting is scheduled for {value}. Mark your calendar: the important date is {value}.",
        "query": "When is the meeting scheduled?",
        "gen_value": lambda: "March 15, 2048",
    },
    "passkey": {
        "template": "The pass key is {value}. I repeat: the pass key is {value}.",
        "query": "What is the pass key?",
        "gen_value": lambda: str(np.random.randint(10000, 99999)),
    },
}


# ===================================================================
# Chunk-Boundary-Safe Insertion
# ===================================================================

def inject_needle_safe(
    text: str,
    needle: str,
    depth: float,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[str, int]:
    """
    Insert needle at a specific depth, ensuring it lands entirely
    within a single chunk (not split across chunk boundaries).

    Strategy:
      1. Compute target character position from depth.
      2. Find the nearest chunk boundary BEFORE that position.
      3. Pad with whitespace so the needle starts at the chunk boundary.
      4. Insert the needle (guaranteed to be within one chunk).

    Args:
        text: Haystack text.
        needle: Needle text to insert.
        depth: Depth percentage (0.0 = start, 1.0 = end).
        chunk_size: Expected chunk size for alignment.

    Returns:
        (modified_text, expected_chunk_id) where expected_chunk_id is
        the chunk index where the needle should appear.
    """
    if len(needle) >= chunk_size:
        raise ValueError(
            f"Needle ({len(needle)} chars) must be shorter than "
            f"chunk_size ({chunk_size} chars)"
        )

    # Target position in the text
    target_pos = int(len(text) * depth)

    # Align to the START of a chunk boundary
    chunk_start = (target_pos // chunk_size) * chunk_size

    # We want the needle centered within the chunk
    # Compute how many padding chars we need before the needle
    pad_before = max(0, (chunk_size - len(needle)) // 4)  # small top pad
    insert_pos = chunk_start + pad_before

    # Clamp to text bounds
    insert_pos = min(insert_pos, len(text))

    # Build the modified text with the needle embedded
    # Add newline separators to visually isolate the needle
    padded_needle = f"\n\n{needle}\n\n"
    modified = text[:insert_pos] + padded_needle + text[insert_pos:]

    # Calculate which chunk the needle should land in
    expected_chunk_id = insert_pos // chunk_size

    return modified, expected_chunk_id


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_needle(
    indexer: MapleIndexer,
    scanner: MapleScanner,
    base_corpus: str,
    context_size: int,
    depth: float,
    needle_type: str = "uuid",
    k: int = 5,
) -> Tuple[bool, str]:
    """
    Run a single needle-in-a-haystack trial.

    Returns:
        (found, needle_value) where found=True if the needle block
        was retrieved in top-k.
    """
    haystack = base_corpus[:context_size]
    needle_cfg = NEEDLE_TYPES[needle_type]

    # Generate needle
    value = needle_cfg["gen_value"]()
    needle_text = needle_cfg["template"].format(value=value)
    query = needle_cfg["query"]

    # Inject safely
    modified_text, expected_chunk = inject_needle_safe(
        haystack, needle_text, depth, CHUNK_SIZE
    )

    # Index
    blocks = indexer.chunk_text(modified_text, chunk_size=CHUNK_SIZE)
    embeddings = indexer.encode_blocks(blocks, show_progress=False)
    index = Index(blocks=blocks, embeddings=embeddings, chunk_size=CHUNK_SIZE)

    # Query
    query_emb = indexer.encode_query(query)
    result = scanner.search(query_emb, index, k=k, strategy="adaptive")

    # Check if any retrieved block contains the needle value
    found = False
    for bid in result.block_ids:
        if bid < len(blocks) and value in blocks[bid].text:
            found = True
            break

    return found, value


# ===================================================================
# Heatmap Visualization
# ===================================================================

def plot_passkey_heatmap(
    heatmap: List[List[float]],
    depths: List[float],
    context_sizes: List[int],
    needle_type: str,
    output_path: Path,
):
    """
    Generate a PassKey-style heatmap (Llama-3 tech report style).

    X-axis: Needle depth (0%..100%)
    Y-axis: Context size
    Color:  Retrieval accuracy (0..1)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    data = np.array(heatmap)
    im = ax.imshow(
        data,
        cmap="RdYlGn",
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"{d:.0%}" for d in depths], fontsize=10)
    ax.set_yticks(range(len(context_sizes)))
    ax.set_yticklabels([f"{s//1000}K" for s in context_sizes], fontsize=10)

    ax.set_xlabel("Needle Depth", fontsize=13)
    ax.set_ylabel("Context Size (chars)", fontsize=13)
    ax.set_title(
        f"MAPLE Needle-in-a-Haystack  (type: {needle_type})",
        fontsize=15,
        fontweight="bold",
    )

    # Annotate cells
    for i in range(len(context_sizes)):
        for j in range(len(depths)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Retrieval Accuracy", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap saved -> {output_path}")


# ===================================================================
# Main
# ===================================================================

def run(needle_type: str = "uuid"):
    """Run the robust needle-in-a-haystack benchmark."""
    print("=" * 70)
    print("BENCHMARK 03 (ROBUST): Needle in a Haystack")
    print(f"  Needle Type:    {needle_type}")
    print(f"  Depths:         {len(DEPTHS)} ({DEPTHS[0]:.0%}..{DEPTHS[-1]:.0%})")
    print(f"  Context Sizes:  {[f'{s//1000}K' for s in CONTEXT_SIZES]}")
    print(f"  Trials/Cell:    {NUM_TRIALS}")
    print(f"  Chunk Size:     {CHUNK_SIZE}")
    print("=" * 70)

    # ---- Load components ----
    indexer = MapleIndexer(device=DEVICE)
    model = MapleNet.load(str(MODEL_PATH), device=DEVICE)
    scanner = MapleScanner(model, device=DEVICE)

    # ---- Build corpus ----
    max_size = max(CONTEXT_SIZES) + CHUNK_SIZE * 2  # extra margin
    logger.info(f"Building base corpus ({max_size:,} chars)...")
    base_corpus = build_large_corpus(max_size)

    # ---- Run trials ----
    heatmap = []

    for size_idx, context_size in enumerate(CONTEXT_SIZES):
        print(f"\n--- Context: {context_size//1000}K chars ---")
        row = []

        for depth_idx, depth in enumerate(DEPTHS):
            hits = 0

            for trial in range(NUM_TRIALS):
                found, _ = evaluate_needle(
                    indexer, scanner, base_corpus,
                    context_size, depth, needle_type,
                )
                if found:
                    hits += 1

            accuracy = hits / NUM_TRIALS
            row.append(accuracy)

            marker = "+" if accuracy >= 0.8 else ("~" if accuracy >= 0.4 else "-")
            print(f"  Depth {depth:>5.0%}: {accuracy:.0%} {marker}")

        heatmap.append(row)

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "needle_type": needle_type,
        "needle_query": NEEDLE_TYPES[needle_type]["query"],
        "depths": DEPTHS,
        "context_sizes": CONTEXT_SIZES,
        "heatmap": heatmap,
        "num_trials": NUM_TRIALS,
        "chunk_size": CHUNK_SIZE,
    }

    json_path = RESULTS_DIR / "needle_robust.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- Plot heatmap ----
    png_path = RESULTS_DIR / "needle_heatmap_robust.png"
    plot_passkey_heatmap(heatmap, DEPTHS, CONTEXT_SIZES, needle_type, png_path)

    print("\n" + "=" * 70)
    print(f"Saved  -> {json_path}")
    print(f"Chart  -> {png_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Robust Needle-in-a-Haystack")
    parser.add_argument(
        "--needle-type", type=str, default="uuid",
        choices=list(NEEDLE_TYPES.keys()),
        help="Type of needle to insert",
    )
    args = parser.parse_args()
    run(needle_type=args.needle_type)
