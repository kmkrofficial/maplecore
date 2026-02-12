#!/usr/bin/env python3
"""
Train Generalist MAPLE Model
============================

Trains a single Sticky-LLM adapter on mixed data from:
1. NarrativeQA (Literary/Book domain)
2. HotpotQA (Encyclopedic/Wikipedia domain)

Goal: Prevent catastrophic forgetting and overfitting to one structure.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import DATA_DIR
from maplecore import MapleIndexer, MapleTrainer
from maplecore.trainer import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
NARRATIVE_DATA = DATA_DIR / "oracle_data.json"
HOTPOT_DATA = DATA_DIR / "oracle_hotpotqa.json"
MODEL_OUTPUT = "models/maple_generalist.pth"

def load_and_merge_data():
    """Load both datasets and merge them."""
    data = []
    
    # 1. NarrativeQA
    if NARRATIVE_DATA.exists():
        with open(NARRATIVE_DATA, "r", encoding="utf-8") as f:
            nq_data = json.load(f)
            # Handle list vs dict format
            if isinstance(nq_data, dict) and "samples" in nq_data:
                nq_data = nq_data["samples"]
            
            logger.info(f"Loaded {len(nq_data)} NarrativeQA samples.")
            for item in nq_data:
                item["domain"] = "narrative_qa" # Ensure tag
            data.extend(nq_data)
    else:
        logger.warning(f"{NARRATIVE_DATA} not found! Training only on HotpotQA?")

    # 2. HotpotQA
    if HOTPOT_DATA.exists():
        with open(HOTPOT_DATA, "r", encoding="utf-8") as f:
            hq_data = json.load(f)
            logger.info(f"Loaded {len(hq_data)} HotpotQA samples.")
            # ...
            for item in hq_data:
                item["domain"] = "hotpot_qa" # Ensure tag
            data.extend(hq_data)
    else:
        logger.warning(f"{HOTPOT_DATA} not found!")
        
    if not data:
        raise ValueError("No training data found!")
        
    # Shuffle to mix domains in batches
    random.shuffle(data)
    logger.info(f"Total training samples: {len(data)}")
    
    return data

def prepare_dataset(raw_samples, indexer: MapleIndexer) -> Dataset:
    """Convert raw oracle samples to Trainer Dataset."""
    dataset_items = []
    
    # Pre-embed logic (conceptually similar to train_maple.py)
    # But since MapleTrainer expects (query_emb, pos_embs, neg_embs),
    # we can just prepare those lists.
    # Wait, MapleTrainer.train() takes a list of (query, pos, neg) tuples?
    # No, let's check MapleTrainer signature.
    # It takes a DataLoader or list.
    
    # We'll use the same logic as train_maple.py:
    # 1. Embed query
    # 2. Embed "pos" (top-5 blocks)
    # 3. Embed "neg" (random other blocks)
    
    logger.info("Encoding samples for training...")
    
    for sample in raw_samples:
        query_text = sample["question"]
        
        # Normalize schema
        if "top_5_block_ids" in sample: 
            # NarrativeQA Schema
            label_ids = set(sample["top_5_block_ids"])
            # Convert all_block_texts dict to list of {id, text}
            blocks = []
            for bid_str, text in sample["all_block_texts"].items():
                blocks.append({"id": int(bid_str), "text": text})
        else:
            # HotpotQA Schema (default)
            blocks = sample["blocks"] # List of {id, text}
            label_ids = set(sample["labels"])
        
        # 1. Embed Query
        query_emb = indexer.encode_query(query_text) # (dim,)
        
        # 2. Embed Blocks (Batch)
        # Note: blocks might not be sorted by ID, but that's fine for sets
        block_texts = [b["text"] for b in blocks]
        if not block_texts: continue
        
        block_embs = indexer.model.encode(block_texts, convert_to_tensor=True)
        
        # 3. Split Positive/Negative
        pos_embs = []
        neg_embs = []
        
        for i, block in enumerate(blocks):
            if block["id"] in label_ids:
                pos_embs.append(block_embs[i])
            else:
                neg_embs.append(block_embs[i])
                
        # 4. Create Triples/Pairs
        # Simplest approach: One entry per Positive, with 1 Negative
        if not pos_embs: continue
        if not neg_embs: continue # Rare case
        
        # Balance: If we have 5 pos, pick 5 negs
        num_pairs = min(len(pos_embs), len(neg_embs))
        # Or use all positives and random negatives
        
        for i in range(len(pos_embs)):
            p_emb = pos_embs[i]
            # Pick a random negative
            n_emb = random.choice(neg_embs)
            
            # Positive
            dataset_items.append({
                "question": query_text,
                "query_emb": query_emb,
                "block_emb": p_emb,
                "label": 1.0,
                "block_id": blocks[i]["id"] # Use normalized block ID
            })
            # Negative
            dataset_items.append({
                "question": query_text,
                "query_emb": query_emb,
                "block_emb": n_emb,
                "label": 0.0,
                "block_id": -1 # Dummy ID for negative
            })
            
    logger.info(f"Created {len(dataset_items)} training pairs.")
    return dataset_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # 1. Load merged data
    raw_data = load_and_merge_data()
    
    # 2. Initialize Indexer (for embeddings)
    indexer = MapleIndexer(device=args.device) # Loads BGE
    
    # 3. Prepare Tensors
    train_data = prepare_dataset(raw_data, indexer)
    
    # 4. Train
    logger.info("Starting training...")
    trainer = MapleTrainer(
        input_dim=indexer.model.get_sentence_embedding_dimension() * 2,
        hidden_dim=128,
        device=args.device
    )
    
    model_path = Path(MODEL_OUTPUT)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    trainer.train(
        training_data=train_data,
        epochs=args.epochs,
        batch_size=32,
        lr=1e-3,
        val_split=0.2,
        save_path=str(model_path)
    )
    
    logger.info(f"Model saved to {model_path}")
    
    # 6. Plot Training Curve
    if hasattr(trainer, "history_"):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(trainer.history_["loss"], label="Train Loss")
        plt.title("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(trainer.history_["val_recall"], label="Val Recall")
        plt.title("Recall")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("benchmarks/results/training_generalist_curve.png")
        logger.info("Training curve saved.")

if __name__ == "__main__":
    from benchmarks.profiler import HardwareMonitor, wrap_result
    import json
    
    with HardwareMonitor(interval=0.1) as mon:
        train()

        
    # Save metrics (Only hardware for now as main() returns nothing)
    metrics = {"status": "completed"}
    final_output = wrap_result(metrics, mon)
    
    out_path = Path("models/training_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Training profile saved to {out_path}")
