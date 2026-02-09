#!/usr/bin/env python3
"""
run_quantization_experiment.py
==============================
Scout-KV Phase 8: INT8 Quantization Experiment

Compares FP32 vs INT8 quantized ScoutBGE model:
- Model size reduction
- CPU latency improvement
- Recall@5 accuracy preservation
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
from tqdm import tqdm

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


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
# Utility Functions
# =============================================================================

def get_file_size_mb(path):
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def run_inference(model, query_emb, block_embs, device="cpu"):
    """Run Scout inference on all blocks, return top-5 and time."""
    model = model.to(device)
    query_emb = query_emb.to(device)
    block_embs = block_embs.to(device)
    
    start = time.perf_counter()
    
    with torch.no_grad():
        scores = []
        for block_emb in block_embs:
            combined = torch.cat([query_emb, block_emb], dim=0).unsqueeze(0)
            logit = model(combined)
            scores.append(logit.item())
    
    elapsed = time.perf_counter() - start
    
    # Get top-5
    indexed = [(i, s) for i, s in enumerate(scores)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    top5 = [idx for idx, _ in indexed[:5]]
    
    return top5, elapsed


def compute_recall(predicted, ground_truth, k=5):
    """Compute Recall@K."""
    pred_set = set(predicted[:k])
    truth_set = set(ground_truth[:k])
    
    if len(truth_set) == 0:
        return 0.0
    
    hits = len(pred_set & truth_set)
    return hits / len(truth_set)


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("="*70)
    print("Scout-KV Phase 8: INT8 Quantization Experiment")
    print("="*70)
    
    # =========================================================================
    # Setup
    # =========================================================================
    print("\n[Setup] Loading models and data...")
    
    # Load FP32 model
    model_fp32 = ScoutBGE(input_dim=768, hidden_dim=128, dropout=0.3)
    model_fp32.load_state_dict(torch.load("scout_bge.pth", weights_only=True))
    model_fp32.eval()
    
    fp32_path = "scout_bge.pth"
    fp32_size = get_file_size_mb(fp32_path)
    print(f"  FP32 model size: {fp32_size:.4f} MB")
    
    # =========================================================================
    # Quantization
    # =========================================================================
    print("\n[Quantization] Converting FP32 → INT8...")
    
    # Dynamic quantization targets Linear layers
    model_int8 = quant.quantize_dynamic(
        model_fp32,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )
    
    # Save quantized model
    int8_path = "scout_bge_int8.pth"
    torch.save(model_int8.state_dict(), int8_path)
    int8_size = get_file_size_mb(int8_path)
    print(f"  INT8 model size: {int8_size:.4f} MB")
    print(f"  Size reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
    
    # =========================================================================
    # Load Validation Data
    # =========================================================================
    print("\n[Data] Loading oracle_data.json and BGE embeddings...")
    
    with open("oracle_data.json", "r", encoding="utf-8") as f:
        oracle_data = json.load(f)
    
    samples = oracle_data["samples"]
    print(f"  Loaded {len(samples)} samples")
    
    # Load BGE
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")  # CPU for fair comparison
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    oracle_lookup = {s["question"]: s for s in samples}
    
    # =========================================================================
    # Benchmarking Loop
    # =========================================================================
    print("\n[Benchmark] Running 50 samples (CPU)...")
    
    BLOCK_SIZE = 128
    MAX_TOKENS = 2048
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    fp32_recalls = []
    int8_recalls = []
    fp32_latencies = []
    int8_latencies = []
    
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
        
        if len(blocks) < 5:
            continue
        
        # Encode with BGE (CPU)
        query_emb = bge.encode(QUERY_PREFIX + question, convert_to_tensor=True)
        block_embs = bge.encode(blocks, convert_to_tensor=True, batch_size=32)
        
        # FP32 inference
        fp32_top5, fp32_time = run_inference(model_fp32, query_emb, block_embs, device="cpu")
        fp32_recalls.append(compute_recall(fp32_top5, ground_truth))
        fp32_latencies.append(fp32_time * 1000)  # ms
        
        # INT8 inference
        int8_top5, int8_time = run_inference(model_int8, query_emb, block_embs, device="cpu")
        int8_recalls.append(compute_recall(int8_top5, ground_truth))
        int8_latencies.append(int8_time * 1000)  # ms
        
        matched += 1
    
    # =========================================================================
    # Results
    # =========================================================================
    avg_fp32_recall = np.mean(fp32_recalls) * 100
    avg_int8_recall = np.mean(int8_recalls) * 100
    avg_fp32_latency = np.mean(fp32_latencies)
    avg_int8_latency = np.mean(int8_latencies)
    
    size_delta = int8_size - fp32_size
    latency_delta = avg_int8_latency - avg_fp32_latency
    recall_delta = avg_int8_recall - avg_fp32_recall
    
    print("\n" + "="*70)
    print("QUANTIZATION RESULTS")
    print("="*70)
    print(f"{'Metric':<15} | {'FP32 (Original)':<18} | {'INT8 (Quantized)':<18} | {'Delta':<10}")
    print("-"*70)
    print(f"{'Size (MB)':<15} | {fp32_size:<18.4f} | {int8_size:<18.4f} | {size_delta:+.4f}")
    print(f"{'Latency (ms)':<15} | {avg_fp32_latency:<18.3f} | {avg_int8_latency:<18.3f} | {latency_delta:+.3f}")
    print(f"{'Recall@5 (%)':<15} | {avg_fp32_recall:<18.1f} | {avg_int8_recall:<18.1f} | {recall_delta:+.1f}")
    print("="*70)
    
    # Analysis
    print("\n[Analysis]")
    
    if int8_size < fp32_size:
        print(f"  ✓ Size: {(1-int8_size/fp32_size)*100:.1f}% smaller")
    else:
        print(f"  ⚠ Size: INT8 is larger (quantization overhead for tiny model)")
    
    if avg_int8_latency < avg_fp32_latency:
        speedup = avg_fp32_latency / avg_int8_latency
        print(f"  ✓ Latency: {speedup:.2f}x faster")
    else:
        print(f"  ⚠ Latency: INT8 is slower (overhead for tiny model on CPU)")
    
    if abs(recall_delta) < 2:
        print(f"  ✓ Accuracy: Preserved (delta < 2%)")
    else:
        print(f"  ⚠ Accuracy: Significant change detected")
    
    print("\n" + "="*70)
    print(f"Quantized model saved: {int8_path}")
    print("="*70)


if __name__ == "__main__":
    main()
