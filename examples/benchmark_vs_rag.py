#!/usr/bin/env python3
"""
Benchmark: MAPLE vs Standard RAG
==================================

This example compares MAPLE's learned retrieval against
standard RAG (cosine similarity) on the same dataset.
"""

import json
import logging
import time
from collections import defaultdict

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from maplecore import Maple, MapleNet, MapleIndexer
from maplecore.utils import _cosine_similarity as cosine_similarity


def main():
    print("="*70)
    print("MAPLE vs RAG: Recall Benchmark")
    print("="*70)
    
    # Load oracle data (ground truth)
    print("\n[1] Loading oracle data...")
    with open("oracle_data.json", "r", encoding="utf-8") as f:
        oracle_data = json.load(f)
    
    samples = oracle_data["samples"]
    print(f"    Loaded {len(samples)} samples")
    
    # Initialize components
    print("\n[2] Loading models...")
    indexer = MapleIndexer(device="cuda")
    model = MapleNet.load("maple.pth", device="cuda")
    
    # Load NarrativeQA
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print("\n[3] Loading NarrativeQA dataset...")
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    oracle_lookup = {s["question"]: s for s in samples}
    
    # Benchmark
    print("\n[4] Running benchmark...")
    
    BLOCK_SIZE = 128
    MAX_TOKENS = 2048
    
    rag_recalls = []
    maple_recalls = []
    matched = 0
    
    for ds_sample in dataset:
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
        ground_truth = set(oracle["top_5_block_ids"][:5])
        
        # Truncate and chunk
        tokens = tokenizer.encode(context, add_special_tokens=False)[:MAX_TOKENS]
        blocks = []
        for i in range(0, len(tokens), BLOCK_SIZE):
            block_tokens = tokens[i:i+BLOCK_SIZE]
            blocks.append(tokenizer.decode(block_tokens, skip_special_tokens=True))
        
        if len(blocks) < 5:
            continue
        
        # Encode
        query_emb = indexer.encode_query(question)
        block_embs = indexer.model.encode(blocks, convert_to_tensor=True, device="cuda")
        
        # RAG: Cosine similarity
        sims = cosine_similarity(query_emb, block_embs)
        _, rag_top5 = torch.topk(sims, 5)
        rag_pred = set(rag_top5.tolist())
        rag_recall = len(rag_pred & ground_truth) / 5
        rag_recalls.append(rag_recall)
        
        # MAPLE: Learned MLP
        with torch.no_grad():
            query_exp = query_emb.unsqueeze(0).expand(len(blocks), -1)
            combined = torch.cat([query_exp, block_embs], dim=1)
            scores = torch.sigmoid(model(combined))
        
        _, maple_top5 = torch.topk(scores, 5)
        maple_pred = set(maple_top5.tolist())
        maple_recall = len(maple_pred & ground_truth) / 5
        maple_recalls.append(maple_recall)
        
        matched += 1
        if matched % 10 == 0:
            print(f"    Processed {matched}/{len(samples)} samples")
    
    # Results
    avg_rag = np.mean(rag_recalls) * 100
    avg_maple = np.mean(maple_recalls) * 100
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Method':<20} | {'Recall@5':<15}")
    print("-"*40)
    print(f"{'RAG (Cosine Sim)':<20} | {avg_rag:.1f}%")
    print(f"{'MAPLE (MLP)':<20} | {avg_maple:.1f}%")
    print("-"*40)
    print(f"{'Improvement':<20} | {avg_maple/avg_rag:.1f}x")
    print("="*70)


if __name__ == "__main__":
    main()
