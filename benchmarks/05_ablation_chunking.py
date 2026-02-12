#!/usr/bin/env python3
"""
Benchmark 05: Chunk Size Ablation
===================================

Investigates the impact of chunk size on:
  1. Recall@5 (Accuracy)
  2. Index Size (RAM Usage)

Tested sizes: 128, 256, 512 tokens (approx)
Converts chars to tokens using dist ~4 chars/token approximation
for chunking logic (500 chars ~ 125 tokens).

Output:
  - results/ablation_chunking.json
  - results/ablation_tradeoff.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleIndexer, MapleNet, MapleScanner
from maplecore.utils import _cosine_similarity as cosine_similarity
from benchmarks.config import DATA_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Approx mapping: 1 token ~ 4 characters
CHUNK_SIZES_TOKENS = [128, 256, 512]
CHUNK_SIZES_CHARS = [500, 1000, 2000] # roughly corresponding sizes
MODEL_PATH = Path("models/maple_v1.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 100 # Smaller set for speed

def _load_data():
    """Load NarrativeQA samples (subset)."""
    oracle_path = DATA_DIR / "oracle_data.json"
    if not oracle_path.exists():
        logger.error("oracle_data.json missing")
        return []
        
    with open(oracle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return data["samples"][:MAX_SAMPLES]

def evaluate_chunk_size(samples: list, chunk_chars: int, indexer: MapleIndexer, model: MapleNet):
    """Calculate average Recall@5 for a specific chunk size."""
    hits = 0
    total = 0
    total_blocks = 0
    
    # We need to re-chunk the original full context
    # But oracle data only gives us TOP blocks and contexts.
    # We should ideally use the full document from data_loader, 
    # but for ablation we can stick to valid samples where we have the text.
    
    # Since we need to measure RECALL, we need ground truth.
    # The oracle ground truth is specific to the chunking used during oracle generation (500 chars).
    # CHANGING the chunk size invalidates the oracle labels (block IDs will shift).
    
    # SOLUTION:
    # Instead of using pre-computed oracle block IDs, we must check if the retrieved block
    # contains the *answer span* or match the ground truth text content.
    # However, NarrativeQA is extensive.
    
    # ALTERNATIVE:
    # Use the "Needle" approach but with varying chunk sizes? 
    # Or just re-run the recall test assuming the "ground truth" logic 
    # can be proxied by "does the retrieved block align with the original top blocks?"
    
    # Better approach for this ablation given constraints:
    # Use the Oracle Top-1 block text as the "Query" (Passage Retrieval)
    # OR: Use the same Question, but re-scan the document.
    # Ground truth: The text span that was in the oracle's top block.
    # If a retrieved block overlaps significantly (>50%) with the Oracle Top Block text, count as hit.
    
    for s in samples:
        question = s["question"]
        
        # Reconstruct rough document from available blocks (imperfect but viable)
        # OR better: skip samples where we don't have full doc.
        # Let's assume we use the 'all_block_texts' from oracle as "document"
        doc_text = "\n".join(s["all_block_texts"].values())
        
        # Identified "Golden Span" from oracle (the text of the top-1 block)
        if not s["top_5_block_ids"]: continue
        gold_id = str(s["top_5_block_ids"][0])
        gold_text = s["all_block_texts"].get(gold_id, "")
        if not gold_text: continue
        
        # 1. Chunk document with NEW size
        blocks = indexer.chunk_text(doc_text, chunk_size=chunk_chars)
        total_blocks += len(blocks)
        
        if len(blocks) < 5: continue
        
        # 2. Encode
        query_emb = indexer.encode_query(question)
        block_embs = indexer.encode_blocks(blocks, show_progress=False)
        
        # 3. Score (MAPLE)
        with torch.no_grad():
            query_exp = query_emb.unsqueeze(0).expand(len(blocks), -1)
            combined = torch.cat([query_exp, block_embs], dim=1)
            scores = torch.sigmoid(model(combined))
            
        top_indices = torch.argsort(scores, descending=True)[:5].tolist()
        
        # 4. Check hit (overlap with golden text)
        is_hit = False
        for idx in top_indices:
            # Simple overlap check: is a significant part of gold text in this block?
            # or vice versa.
            retrieved_text = blocks[idx].text
            
            # Intersection over Union of words?
            s1 = set(gold_text.split())
            s2 = set(retrieved_text.split())
            intersection = len(s1 & s2)
            if intersection / len(s1) > 0.5: # 50% of gold words found
                is_hit = True
                break
        
        if is_hit:
            hits += 1
        total += 1
        
    avg_blocks = total_blocks / max(1, total)
    recall = hits / max(1, total) * 100
    
    # Estimate VRAM: Blocks * 384 dims * 4 bytes (float32)
    vram_est_mb = (avg_blocks * 384 * 4) / 1024 / 1024 
    
    return recall, avg_blocks, vram_est_mb


def plot_tradeoff(results: dict, output_path: Path):
    """Plot Recall vs Chunk Size and RAM Cost."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sizes = results["sizes_chars"]
    recalls = results["recalls"]
    vram_costs = results["vram_per_doc_mb"]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Chunk Size (chars)', fontsize=12)
    ax1.set_ylabel('Recall@5 (%)', color=color, fontsize=12)
    ax1.plot(sizes, recalls, color=color, marker='o', linewidth=2, label="Recall")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Embedding RAM / Doc (MB)', color=color, fontsize=12)
    ax2.plot(sizes, vram_costs, color=color, marker='s', linestyle="--", label="RAM Usage")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Chunk Size Ablation: Accuracy vs Efficiency", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved -> {output_path}")


def run():
    print("=" * 70)
    print("BENCHMARK 05: Chunk Size Ablation")
    print("=" * 70)
    
    samples = _load_data()
    if not samples: return
    
    indexer = MapleIndexer(device=DEVICE)
    model = MapleNet.load(str(MODEL_PATH), device=DEVICE)
    
    results = {
        "sizes_chars": [],
        "recalls": [],
        "avg_blocks": [],
        "vram_per_doc_mb": []
    }
    
    for size_char in CHUNK_SIZES_CHARS:
        print(f"\n--- Testing Chunk Size: {size_char} chars ---")
        recall, blocks, vram = evaluate_chunk_size(samples, size_char, indexer, model)
        
        results["sizes_chars"].append(size_char)
        results["recalls"].append(recall)
        results["avg_blocks"].append(blocks)
        results["vram_per_doc_mb"].append(vram)
        
        print(f"  Recall@5: {recall:.1f}%")
        print(f"  Avg Blocks: {blocks:.1f}")
        print(f"  Est RAM:  {vram:.2f} MB/doc")

    # Save
    json_path = RESULTS_DIR / "ablation_chunking.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    png_path = RESULTS_DIR / "ablation_tradeoff.png"
    plot_tradeoff(results, png_path)
    
    print("\n" + "=" * 70)
    print(f"Saved -> {json_path}")
    print(f"Chart -> {png_path}")
    print("=" * 70)

if __name__ == "__main__":
    run()
