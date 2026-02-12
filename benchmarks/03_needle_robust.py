#!/usr/bin/env python3
"""
Benchmark 03: Robustness (Needle-in-a-Haystack)
===============================================

Tests the retrieval precision of MAPLE across varying context lengths.
Inserts a specific "needle" (fact/UUID) into a large "haystack" (filler text)
at various depths and checks if the correct block is retrieved in the Top-5.

Key Features:
- Safe Insertion: Pads needle with newlines to avoid chunk splitting.
- Dual Types: UUID (exact) vs Fact (semantic).
- Visualization: Heatmap of success rates.
"""

import argparse
import json
import logging
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import DATA_DIR, RESULTS_DIR, NEEDLE_CONTEXT_SIZES, NEEDLE_DEPTHS, MODEL_PATH
from maplecore import MapleScanner, MapleIndexer, MapleNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use slightly smaller steps if strict 100k is too slow, but user asked for 100k
CONTEXT_LENGTHS = [10_000, 25_000, 50_000, 100_000] 
DEPTHS = [0, 25, 50, 75, 100] # Percentages

HAYSTACK_FILE = DATA_DIR / "haystack_corpus.txt" # We will generate or load this

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_haystack(length: int) -> str:
    """
    Generate a haystack of roughly `length` characters using Alice in Wonderland text.
    """
    alices_adventures = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: "
        "once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "
        "'and what is the use of a book,' thought Alice 'without pictures or conversation?' "
        "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), "
        "whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her. "
        "There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, "
        "'Oh dear! Oh dear! I shall be too late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, "
        "but at the time it all seemed quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-POCKET, "
        "and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, "
        "and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge. "
    )
    multiplier = (length // len(alices_adventures)) + 1
    return (alices_adventures * multiplier)[:length]

def insert_safe_needle(haystack: str, needle: str, depth_percent: int) -> Tuple[str, int]:
    """
    Insert needle at specific depth with newline padding.
    """
    if depth_percent == 0:
        insert_idx = 0
    elif depth_percent == 100:
        insert_idx = len(haystack)
    else:
        insert_idx = int(len(haystack) * (depth_percent / 100))
        
    # Find nearest space to avoid cutting words
    while insert_idx < len(haystack) and haystack[insert_idx] != " ":
        insert_idx += 1
        
    # Pad with newlines to force chunk boundaries or at least distinct semantics
    padded_needle = f"\n\n{needle}\n\n"
    
    new_text = haystack[:insert_idx] + padded_needle + haystack[insert_idx:]
    return new_text, insert_idx

def get_needle_pair(type_: str) -> Tuple[str, str]:
    """Return (question, needle_text) for a given type."""
    if type_ == "uuid":
        secret_code = str(uuid.uuid4())
        needle = f"The secret passkey is {secret_code}."
        question = "What is the secret passkey?"
        return question, needle
    else:
        # Fact
        city = random.choice(["San Francisco", "New York", "London", "Tokyo", "Berlin"])
        day = random.choice(["Monday", "Tuesday", "Friday"])
        time_ = f"{random.randint(1, 12)} PM"
        needle = f"The Project MAPLE secret meeting is held in {city} every {day} at {time_}."
        question = "Where and when is the secret meeting?"
        return question, needle

# ---------------------------------------------------------------------------
# Benchmark Logic
# ---------------------------------------------------------------------------

def run_test(
    scanner: MapleScanner,
    context_len: int,
    depth: int,
    needle_type: str = "fact"
) -> bool:
    """
    Run a single needle test.
    
    Returns: True if needle block is in Top-5 results.
    """
    # 1. Prepare Data
    haystack = generate_haystack(context_len)
    question, needle = get_needle_pair(needle_type)
    final_text, _ = insert_safe_needle(haystack, needle, depth)
    
    # 2. Index
    # We use temporary index
    from maplecore.indexer import MapleIndexer
    
    # We need to construct an Index object manually or usage scanner's internal?
    # Scanner takes `Index` object.
    # MapleIndexer creates `Index`.
    
    # Use indexer from scanner? Scanner doesn't enforce indexer, it takes Index.
    # We can assume `scanner` has access to embedding model?
    # Wait, `MapleScanner` uses `self.model` (the MLP).
    # `MapleIndexer` uses BGE.
    
    # We need to embed the text first.
    indexer = MapleIndexer(device=scanner.device)
    index = indexer.create_index(final_text, chunk_size=200)
    
    # 3. Search
    # We want to use "adaptive" search as per user request
    query_emb = indexer.encode_query(question)
    
    results = scanner.search(
        query_emb, 
        index, 
        k=5, 
        strategy="adaptive"
    )
    
    # 4. Verify
    # Check if any of the result blocks contain the needle text
    found = False
    for block_id in results.block_ids:
        # Access block text from index
        if 0 <= block_id < len(index.blocks):
            block_text = index.blocks[block_id].text
            if needle in block_text:
                found = True
                break
            
    # Cleanup to save RAM?
    del index.embeddings
    del index
    torch.cuda.empty_cache()
    
    return found

def plot_heatmap(results: List[Dict], output_path: Path):
    """
    Generate heatmap: Length vs Depth.
    Differs slightly from request (X=Depth, Y=Length).
    """
    # Aggregate data
    # Structure: [(len, depth, success), ...]
    
    lengths = sorted(list(set(r["length"] for r in results)))
    depths = sorted(list(set(r["depth"] for r in results)))
    
    grid = np.zeros((len(lengths), len(depths)))
    
    for r in results:
        l_idx = lengths.index(r["length"])
        d_idx = depths.index(r["depth"])
        grid[l_idx, d_idx] = 1.0 if r["success"] else 0.0
        
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        grid, 
        xticklabels=depths, 
        yticklabels=[f"{l//1000}k" for l in lengths],
        cmap="RdYlGn", 
        vmin=0, 
        vmax=1,
        annot=True,
        fmt=".1f",
        linewidths=.5
    )
    
    ax.set_xlabel("Depth (%)")
    ax.set_ylabel("Context Length (chars)")
    ax.set_title("Needle-in-a-Haystack Robustness (Success Rate)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Heatmap saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=CONTEXT_LENGTHS)
    parser.add_argument("--type", type=str, choices=["uuid", "fact"], default="fact")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Scanner
    logger.info(f"Loading MAPLE Generalist from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error("Model not found! Run training first.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapleNet.load(str(MODEL_PATH), device=device)
    
    scanner = MapleScanner(
        model=model,
        device=device
    )
    
    results = []
    
    # 2. Run Matrix
    logger.info(f"Starting Benchmark (Type: {args.type})")
    
    from benchmarks.profiler import HardwareMonitor, wrap_result
    
    all_results = [] # Renamed to avoid conflict with `results` from `run_test`
    
    with HardwareMonitor(interval=0.1) as mon:
        for length in args.lengths:
            for depth in DEPTHS:
                logger.info(f"Testing Length={length:,}, Depth={depth}%...")
                success = run_test(scanner, length, depth, args.type)
                
                all_results.append({
                    "length": length,
                    "depth": depth,
                    "success": success,
                    "type": args.type
                })
                
                status = "PASSED" if success else "FAILED"
                logger.info(f"  -> {status}")
            
    # 3. Save Results
    final_output = wrap_result(all_results, mon)
    
    json_path = RESULTS_DIR / "needle_robustness.json"
        
    # 4. Plot
    plot_path = RESULTS_DIR / "needle_heatmap.png"
    plot_heatmap(all_results, plot_path) # Use all_results for plotting

if __name__ == "__main__":
    main()
