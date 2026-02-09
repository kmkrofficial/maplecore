#!/usr/bin/env python3
"""
run_hierarchical_demo.py
========================
Scout-KV Phase 9: Hierarchical Scale Demo

Demonstrates hierarchical retrieval for massive contexts:
- 50,000 blocks (~6.4M tokens)
- 2-layer search: Chapters → Blocks
- Target: ~40x speedup over linear scan
"""

import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# =============================================================================
# ScoutBGE Model
# =============================================================================

class ScoutBGE(nn.Module):
    """Scout model for BGE embeddings."""
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# =============================================================================
# Utility Functions
# =============================================================================

def load_sherlock_text():
    """Load Sherlock Holmes book."""
    cache_file = Path("sherlock_holmes.txt")
    
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        import requests
        url = "https://www.gutenberg.org/files/1661/1661-0.txt"
        print(f"Downloading from {url}...")
        response = requests.get(url, timeout=30)
        text = response.text
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
    
    # Clean Gutenberg headers
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    if end != -1:
        text = text[start:end]
    else:
        text = text[start:]
    
    return text.strip()


def chunk_text(text, block_chars=500):
    """Chunk text into blocks (~128 tokens each)."""
    blocks = []
    for i in range(0, len(text), block_chars):
        block = text[i:i+block_chars].strip()
        if block:
            blocks.append(block)
    return blocks


def scout_inference_batch(model, query_emb, block_embs, device="cuda"):
    """
    Batched Scout inference for speed.
    Returns scores for all blocks.
    """
    model = model.to(device)
    query_emb = query_emb.to(device)
    block_embs = block_embs.to(device)
    
    with torch.no_grad():
        # Expand query to match all blocks
        num_blocks = block_embs.shape[0]
        query_expanded = query_emb.unsqueeze(0).expand(num_blocks, -1)
        
        # Concatenate [query, block] for all blocks
        combined = torch.cat([query_expanded, block_embs], dim=1)
        
        # Batch inference
        scores = model(combined)
    
    return scores


def get_top_k(scores, k=5):
    """Get top-k indices from scores tensor."""
    _, top_ids = torch.topk(scores, k)
    return top_ids.cpu().tolist()


# =============================================================================
# Main Demo
# =============================================================================

def main():
    print("="*70)
    print("Scout-KV Phase 9: Hierarchical Scale Demo")
    print("="*70)
    
    # =========================================================================
    # Setup
    # =========================================================================
    print("\n[1] Loading models...")
    
    # Load BGE
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    print("  ✓ BGE loaded")
    
    # Load Scout
    scout = ScoutBGE(input_dim=768, hidden_dim=128, dropout=0.3)
    scout.load_state_dict(torch.load("scout_bge.pth", weights_only=True))
    scout.eval()
    scout = scout.cuda()
    print("  ✓ ScoutBGE loaded")
    
    # =========================================================================
    # Data Simulation: Create 50,000 blocks
    # =========================================================================
    print("\n[2] Simulating massive context...")
    
    # Load and chunk Sherlock Holmes
    text = load_sherlock_text()
    base_blocks = chunk_text(text, block_chars=500)
    print(f"  Base blocks: {len(base_blocks)}")
    
    # Encode base blocks
    print("  Encoding base blocks with BGE...")
    base_embeddings = bge.encode(base_blocks, convert_to_tensor=True, 
                                  device="cuda", batch_size=32, show_progress_bar=True)
    
    # Duplicate to create 50,000 blocks (reuse embeddings to save RAM)
    DUPLICATE_FACTOR = 50
    TARGET_BLOCKS = len(base_blocks) * DUPLICATE_FACTOR
    
    # Stack embeddings (memory efficient - just expand reference)
    all_embeddings = base_embeddings.repeat(DUPLICATE_FACTOR, 1)
    
    print(f"  ✓ Simulated {all_embeddings.shape[0]:,} blocks (~{all_embeddings.shape[0] * 128:,} tokens)")
    
    # =========================================================================
    # Build Hierarchical Index
    # =========================================================================
    print("\n[3] Building hierarchical index...")
    
    CHAPTER_SIZE = 100  # blocks per chapter
    num_chapters = all_embeddings.shape[0] // CHAPTER_SIZE
    
    # Create chapter embeddings by averaging block embeddings
    chapter_embeddings = []
    for i in range(num_chapters):
        start = i * CHAPTER_SIZE
        end = start + CHAPTER_SIZE
        chapter_emb = all_embeddings[start:end].mean(dim=0)
        chapter_embeddings.append(chapter_emb)
    
    chapter_embeddings = torch.stack(chapter_embeddings)
    print(f"  ✓ Created {chapter_embeddings.shape[0]} chapter embeddings")
    
    # =========================================================================
    # Query Encoding
    # =========================================================================
    print("\n[4] Encoding query...")
    
    QUESTION = "What was the speckled band?"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    query_emb = bge.encode(QUERY_PREFIX + QUESTION, convert_to_tensor=True, device="cuda")
    print(f"  Query: \"{QUESTION}\"")
    
    # =========================================================================
    # Linear Search (Baseline)
    # =========================================================================
    print("\n[5] Running LINEAR search (baseline)...")
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    linear_scores = scout_inference_batch(scout, query_emb, all_embeddings)
    linear_top5 = get_top_k(linear_scores, k=5)
    
    torch.cuda.synchronize()
    linear_time = (time.perf_counter() - start_time) * 1000
    
    print(f"  ✓ Scanned {all_embeddings.shape[0]:,} blocks in {linear_time:.1f} ms")
    print(f"  Top-5 blocks: {linear_top5}")
    
    # =========================================================================
    # Hierarchical Search
    # =========================================================================
    print("\n[6] Running HIERARCHICAL search...")
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    # Layer 1: Scout on chapters
    chapter_scores = scout_inference_batch(scout, query_emb, chapter_embeddings)
    top5_chapters = get_top_k(chapter_scores, k=5)
    
    layer1_time = (time.perf_counter() - start_time) * 1000
    
    # Layer 2: Scout on blocks within top-5 chapters
    candidate_blocks = []
    candidate_indices = []
    for chap_id in top5_chapters:
        start = chap_id * CHAPTER_SIZE
        end = start + CHAPTER_SIZE
        candidate_blocks.append(all_embeddings[start:end])
        candidate_indices.extend(range(start, end))
    
    candidate_embeddings = torch.cat(candidate_blocks, dim=0)
    
    block_scores = scout_inference_batch(scout, query_emb, candidate_embeddings)
    top5_relative = get_top_k(block_scores, k=5)
    
    # Map back to global indices
    hierarchical_top5 = [candidate_indices[i] for i in top5_relative]
    
    torch.cuda.synchronize()
    hierarchical_time = (time.perf_counter() - start_time) * 1000
    
    print(f"  Layer 1: {len(chapter_embeddings)} chapters → Top-5: {top5_chapters}")
    print(f"  Layer 2: {candidate_embeddings.shape[0]} blocks → Top-5: {hierarchical_top5}")
    print(f"  ✓ Total time: {hierarchical_time:.1f} ms")
    
    # =========================================================================
    # Results
    # =========================================================================
    speedup = linear_time / hierarchical_time
    
    # Check if results overlap
    linear_set = set(b % len(base_blocks) for b in linear_top5)
    hier_set = set(b % len(base_blocks) for b in hierarchical_top5)
    overlap = len(linear_set & hier_set)
    
    print("\n" + "="*70)
    print("HIERARCHICAL SEARCH RESULTS")
    print("="*70)
    print(f"{'Method':<18} | {'Items Scanned':<15} | {'Latency (ms)':<12} | {'Speedup':<10}")
    print("-"*70)
    print(f"{'Linear Scan':<18} | {all_embeddings.shape[0]:<15,} | {linear_time:<12.1f} | {'1.0x':<10}")
    print(f"{'Hierarchical':<18} | {len(chapter_embeddings) + candidate_embeddings.shape[0]:<15,} | {hierarchical_time:<12.1f} | {speedup:<10.1f}x")
    print("="*70)
    
    print(f"\n[Verification]")
    print(f"  Linear Top-5 (mod base): {sorted(linear_set)}")
    print(f"  Hierarchical Top-5 (mod base): {sorted(hier_set)}")
    print(f"  Overlap: {overlap}/5 blocks match")
    
    if speedup >= 10:
        print(f"\n✓ SUCCESS! Hierarchical search is {speedup:.0f}x faster!")
    else:
        print(f"\n⚠ Speedup is {speedup:.1f}x (expected ~40x for larger contexts)")
    
    print("="*70)


if __name__ == "__main__":
    main()
