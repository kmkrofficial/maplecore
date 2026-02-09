#!/usr/bin/env python3
"""
benchmark_suite.py
==================
Scout-KV Phase 7: Benchmark Suite

Compares Scout-KV against Standard RAG Baseline:
- Recall@5 comparison
- Latency comparison
- Sparsity visualization

Outputs:
- benchmark_recall.png
- benchmark_latency.png
- sparsity_heatmap.png
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# ScoutBGE Model Definition
# =============================================================================

class ScoutBGE(nn.Module):
    """Scout model for BGE embeddings (384 dim)."""
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
# Benchmark Functions
# =============================================================================

def compute_cosine_similarity(query_emb, block_embs):
    """Compute cosine similarity between query and all blocks."""
    # Normalize
    query_norm = query_emb / query_emb.norm()
    block_norms = block_embs / block_embs.norm(dim=1, keepdim=True)
    
    # Dot product = cosine similarity
    similarities = torch.matmul(block_norms, query_norm)
    return similarities


def rag_baseline(query_emb, block_embs, k=5):
    """
    Standard RAG: Pick top-k blocks by cosine similarity.
    Returns: top-k block IDs and time taken
    """
    start = time.time()
    
    sims = compute_cosine_similarity(query_emb, block_embs)
    _, top_ids = torch.topk(sims, k)
    
    elapsed = time.time() - start
    return top_ids.tolist(), elapsed


def scout_inference(scout, query_emb, block_embs, k=5):
    """
    Scout-KV: Pick top-k blocks by MLP logits.
    Returns: top-k block IDs and time taken
    """
    start = time.time()
    
    with torch.no_grad():
        scores = []
        for block_emb in block_embs:
            combined = torch.cat([query_emb, block_emb], dim=0).unsqueeze(0)
            logit = scout(combined)
            scores.append(logit.item())
        
        scores_tensor = torch.tensor(scores)
        _, top_ids = torch.topk(scores_tensor, k)
    
    elapsed = time.time() - start
    return top_ids.tolist(), elapsed


def compute_recall(predicted, ground_truth, k=5):
    """
    Compute Recall@K.
    How many of the ground truth items are in the predicted set?
    """
    pred_set = set(predicted[:k])
    truth_set = set(ground_truth[:k])
    
    if len(truth_set) == 0:
        return 0.0
    
    hits = len(pred_set & truth_set)
    return hits / len(truth_set)


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark():
    """Run the full benchmark suite."""
    print("="*70)
    print("Scout-KV Phase 7: Benchmark Suite")
    print("="*70)
    
    # Load oracle data
    print("\nLoading oracle_data.json...")
    with open("oracle_data.json", "r", encoding="utf-8") as f:
        oracle_data = json.load(f)
    
    samples = oracle_data["samples"]
    print(f"Loaded {len(samples)} samples")
    
    # Load BGE model
    print("\nLoading BGE model...")
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    print("✓ BGE loaded")
    
    # Load Scout model
    print("\nLoading ScoutBGE model...")
    scout = ScoutBGE(input_dim=768, hidden_dim=128, dropout=0.3)
    scout.load_state_dict(torch.load("scout_bge.pth", weights_only=True))
    scout.eval()
    print("✓ ScoutBGE loaded")
    
    # Load NarrativeQA for context
    print("\nLoading NarrativeQA for context...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    oracle_lookup = {s["question"]: s for s in samples}
    
    # Benchmark results
    rag_recalls = []
    scout_recalls = []
    rag_latencies = []
    scout_latencies = []
    
    BLOCK_SIZE = 128
    MAX_TOKENS = 2048
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    print("\nRunning benchmark...")
    matched = 0
    
    for ds_sample in tqdm(dataset, total=len(samples)):
        if matched >= len(samples):
            break
            
        try:
            context = ds_sample["document"]["text"]
            question = ds_sample["question"]["text"]
        except:
            continue
            
        if question not in oracle_lookup:
            continue
        
        oracle = oracle_lookup[question]
        ground_truth = oracle["top_5_block_ids"]
        
        # Truncate context
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        if len(context_tokens) > MAX_TOKENS:
            context_tokens = context_tokens[:MAX_TOKENS]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        
        # Chunk into blocks
        num_blocks = (len(context_tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        blocks = []
        for j in range(num_blocks):
            start = j * BLOCK_SIZE
            end = min((j + 1) * BLOCK_SIZE, len(context_tokens))
            block_tokens = context_tokens[start:end]
            blocks.append(tokenizer.decode(block_tokens, skip_special_tokens=True))
        
        # Skip if mismatch (blocks < 5)
        if len(blocks) < 5:
            continue
        
        # Encode with BGE
        query_emb = bge.encode(QUERY_PREFIX + question, convert_to_tensor=True, device="cuda")
        block_embs = bge.encode(blocks, convert_to_tensor=True, device="cuda", batch_size=32)
        
        # RAG Baseline
        rag_top5, rag_time = rag_baseline(query_emb, block_embs, k=5)
        rag_recall = compute_recall(rag_top5, ground_truth, k=5)
        rag_recalls.append(rag_recall)
        rag_latencies.append(rag_time * 1000)  # Convert to ms
        
        # Scout-KV
        scout_top5, scout_time = scout_inference(scout, query_emb.cpu(), block_embs.cpu(), k=5)
        scout_recall = compute_recall(scout_top5, ground_truth, k=5)
        scout_recalls.append(scout_recall)
        scout_latencies.append(scout_time * 1000)
        
        matched += 1
    
    print(f"\n✓ Benchmarked {matched} samples")
    
    # Calculate averages
    avg_rag_recall = np.mean(rag_recalls) * 100
    avg_scout_recall = np.mean(scout_recalls) * 100
    avg_rag_latency = np.mean(rag_latencies)
    avg_scout_latency = np.mean(scout_latencies)
    
    # ==========================================================================
    # Visualization 1: Recall Comparison
    # ==========================================================================
    print("\nGenerating benchmark_recall.png...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Standard RAG\n(Cosine Similarity)', 'Scout-KV\n(Learned MLP)']
    recalls = [avg_rag_recall, avg_scout_recall]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(methods, recalls, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Recall@5 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Scout-KV vs RAG: Recall Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('benchmark_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Visualization 2: Latency Comparison
    # ==========================================================================
    print("Generating benchmark_latency.png...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    latencies = [avg_rag_latency, avg_scout_latency]
    
    bars = ax.bar(methods, latencies, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Scout-KV vs RAG: Latency Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f} ms', ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('benchmark_latency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Visualization 3: Sparsity Heatmap (Sample #0)
    # ==========================================================================
    print("Generating sparsity_heatmap.png...")
    
    sample_0 = samples[0]
    block_scores = sample_0["all_block_scores"]
    num_blocks = sample_0["num_blocks"]
    top5_ids = set(sample_0["top_5_block_ids"])
    
    # Convert to array
    scores = [block_scores.get(str(i), 0.0) for i in range(num_blocks)]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Bar colors: highlight top-5
    bar_colors = ['#4ECDC4' if i in top5_ids else '#95A5A6' for i in range(num_blocks)]
    
    bars = ax.bar(range(num_blocks), scores, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Block ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Attention Sparsity Visualization (Sample #0)\nTop-5 Blocks: {sorted(top5_ids)}', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Top-5 Blocks'),
        Patch(facecolor='#95A5A6', edgecolor='black', label='Other Blocks')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('sparsity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Summary Table
    # ==========================================================================
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Metric':<25} {'RAG Baseline':<20} {'Scout-KV':<20}")
    print("-"*70)
    print(f"{'Recall@5':<25} {avg_rag_recall:.1f}%{'':<15} {avg_scout_recall:.1f}%")
    print(f"{'Avg Latency':<25} {avg_rag_latency:.2f} ms{'':<12} {avg_scout_latency:.2f} ms")
    print(f"{'Samples Tested':<25} {len(rag_recalls):<20} {len(scout_recalls):<20}")
    print("="*70)
    
    # Winner
    if avg_scout_recall > avg_rag_recall:
        print("\n✓ Scout-KV wins on Recall!")
    elif avg_scout_recall < avg_rag_recall:
        print("\n⚠ RAG baseline has higher recall (Scout needs more training)")
    else:
        print("\n= Tie on Recall")
    
    print("\nCharts saved:")
    print("  - benchmark_recall.png")
    print("  - benchmark_latency.png")
    print("  - sparsity_heatmap.png")
    print("="*70)


if __name__ == "__main__":
    run_benchmark()
