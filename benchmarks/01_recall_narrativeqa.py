#!/usr/bin/env python3
"""
Benchmark 01: Recall@K on NarrativeQA
=======================================

Loads the full NarrativeQA test set and compares:
  - MAPLE (Adaptive) vs Standard RAG (Cosine Similarity)

Output: results/recall_curve.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import Maple, MapleNet, MapleIndexer
from maplecore.utils import _cosine_similarity as cosine_similarity
from benchmarks.config import (
    DATA_DIR, RESULTS_DIR, MODEL_PATH, DEFAULT_DEVICE,
    BLOCK_SIZE_TOKENS, MAX_TOKENS, RECALL_K_VALUES,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(max_samples: int = 500):
    """Run the NarrativeQA recall benchmark."""
    print("=" * 70)
    print("BENCHMARK 01: Recall@K — MAPLE vs Standard RAG")
    print("=" * 70)

    # ---- Load components ----
    logger.info("Loading MAPLE model and indexer...")
    indexer = MapleIndexer(device=DEFAULT_DEVICE)
    model = MapleNet.load(MODEL_PATH, device=DEFAULT_DEVICE)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # ---- Load NarrativeQA ----
    from benchmarks.data_loader import load_narrativeqa
    dataset = load_narrativeqa(split="test", streaming=True)

    # ---- Load oracle data ----
    oracle_path = Path("oracle_data.json")
    if not oracle_path.exists():
        logger.error("oracle_data.json not found. Run training pipeline first.")
        return

    with open(oracle_path, "r", encoding="utf-8") as f:
        oracle_data = json.load(f)

    samples = oracle_data["samples"]
    oracle_lookup = {s["question"]: s for s in samples}
    logger.info(f"Loaded {len(samples)} oracle samples")

    # ---- Run benchmark ----
    # Per-K tracking
    maple_hits = defaultdict(list)   # k -> list of recall values
    rag_hits = defaultdict(list)
    matched = 0

    for ds_sample in dataset:
        if matched >= min(max_samples, len(samples)):
            break

        try:
            context = ds_sample["document"]["text"]
            question = ds_sample["question"]["text"]
        except Exception:
            continue

        if question not in oracle_lookup:
            continue

        oracle = oracle_lookup[question]
        ground_truth = set(oracle["top_5_block_ids"][:5])

        # Tokenize & chunk
        tokens = tokenizer.encode(context, add_special_tokens=False)[:MAX_TOKENS]
        blocks = []
        for i in range(0, len(tokens), BLOCK_SIZE_TOKENS):
            block_tokens = tokens[i:i + BLOCK_SIZE_TOKENS]
            blocks.append(tokenizer.decode(block_tokens, skip_special_tokens=True))

        if len(blocks) < 5:
            continue

        # Encode
        query_emb = indexer.encode_query(question)
        block_embs = indexer.model.encode(blocks, convert_to_tensor=True, device=DEFAULT_DEVICE)

        # RAG: Cosine similarity
        sims = cosine_similarity(query_emb, block_embs)
        rag_sorted = torch.argsort(sims, descending=True).tolist()

        # MAPLE: Learned MLP
        with torch.no_grad():
            query_exp = query_emb.unsqueeze(0).expand(len(blocks), -1)
            combined = torch.cat([query_exp, block_embs], dim=1)
            scores = torch.sigmoid(model(combined))
        maple_sorted = torch.argsort(scores, descending=True).tolist()

        # Calculate recall at each K
        for k in RECALL_K_VALUES:
            rag_pred = set(rag_sorted[:k])
            maple_pred = set(maple_sorted[:k])
            denom = min(len(ground_truth), k)

            rag_hits[k].append(len(rag_pred & ground_truth) / denom)
            maple_hits[k].append(len(maple_pred & ground_truth) / denom)

        matched += 1
        if matched % 25 == 0:
            logger.info(f"  Processed {matched} samples...")

    # ---- Results ----
    maple_recalls = [np.mean(maple_hits[k]) * 100 for k in RECALL_K_VALUES]
    rag_recalls = [np.mean(rag_hits[k]) * 100 for k in RECALL_K_VALUES]

    results = {
        "k_values": RECALL_K_VALUES,
        "maple_recalls": maple_recalls,
        "rag_recalls": rag_recalls,
        "num_samples": matched,
    }

    output_path = RESULTS_DIR / "recall_curve.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'K':<6} | {'RAG':<12} | {'MAPLE':<12} | {'Δ':<10}")
    print("-" * 45)
    for k, rag_r, maple_r in zip(RECALL_K_VALUES, rag_recalls, maple_recalls):
        print(f"{k:<6} | {rag_r:>8.1f}%   | {maple_r:>8.1f}%   | {maple_r - rag_r:>+6.1f}pp")
    print("=" * 70)
    print(f"Saved -> {output_path}")

    return results


if __name__ == "__main__":
    run()
