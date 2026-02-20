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
import gc
import json
import logging
import random
import sys
import uuid
import requests
from tqdm import tqdm
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

# Config for Needle
FRANKENSTEIN_URL = "https://www.gutenberg.org/files/84/84-0.txt"
CHUNK_SIZE = 200
NUM_NEEDLE_SAMPLES = 1000

def get_frankenstein_text() -> str:
    """Download or load Frankenstein text."""
    path = DATA_DIR / "frankenstein.txt"
    if not path.exists():
        logger.info("Downloading Frankenstein from Gutenberg...")
        try:
            resp = requests.get(FRANKENSTEIN_URL)
            resp.raise_for_status()
            text = resp.text
            start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
            end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
            if start != -1 and end != -1:
                text = text[start:end]
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            return "Alice " * 10000 
            
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_needle_data(indexer: MapleIndexer) -> list:
    """Generate synthetic training data formatted for MapleTrainer."""
    text = get_frankenstein_text()
    blocks = indexer.chunk_text(text, chunk_size=CHUNK_SIZE)
    logger.info("Encoding blocks for mining...")
    blocks = blocks[:2000]
    block_embs = indexer.encode_blocks(blocks, show_progress=True)
    
    dataset_items = []
    
    logger.info(f"Generating {NUM_NEEDLE_SAMPLES} synthetic samples...")
    for _ in tqdm(range(NUM_NEEDLE_SAMPLES)):
        secret = str(uuid.uuid4())
        needle_text = f"The secret passkey is {secret}."
        query = "What is the secret passkey?"
        
        query_emb = indexer.encode_query(query)
        
        target_idx = random.randint(0, len(blocks)-1)
        base_block = blocks[target_idx]
        
        style = random.choice(["append", "prepend", "alone", "padded"])
        if style == "append":
            injected_text = base_block.text + f" {needle_text}"
        elif style == "prepend":
            injected_text = f"{needle_text} " + base_block.text
        elif style == "padded":
            injected_text = f"\n\n{needle_text}\n\n"
        else:
            injected_text = needle_text
            
        target_emb = indexer.model.encode(injected_text, convert_to_tensor=True)
        
        # Positive
        dataset_items.append({
            "question": query,
            "query_emb": query_emb,
            "block_emb": target_emb,
            "label": 1.0,
            "block_id": base_block.id if hasattr(base_block, "id") else -1,
            "domain": "synthetic_needle"
        })
        
        # Hard Negative 1 (wrong needle)
        wrong_secret = str(uuid.uuid4())
        wrong_text = base_block.text + f" The secret passkey is {wrong_secret}."
        wrong_emb = indexer.model.encode(wrong_text, convert_to_tensor=True)
        
        dataset_items.append({
            "question": query,
            "query_emb": query_emb,
            "block_emb": wrong_emb,
            "label": 0.0,
            "block_id": -1,
            "domain": "synthetic_needle"
        })
        
        # Hard Negative 2 (same block, no needle)
        dataset_items.append({
            "question": query,
            "query_emb": query_emb,
            "block_emb": block_embs[target_idx],
            "label": 0.0,
            "block_id": -1,
            "domain": "synthetic_needle"
        })
        
        # Hard Negative 3 (adjacent block prior)
        if target_idx > 0:
            dataset_items.append({
                "question": query,
                "query_emb": query_emb,
                "block_emb": block_embs[target_idx - 1],
                "label": 0.0,
                "block_id": -1,
                "domain": "synthetic_needle"
            })
            
        # Hard Negative 4 (adjacent block after)
        if target_idx < len(block_embs) - 1:
            dataset_items.append({
                "question": query,
                "query_emb": query_emb,
                "block_emb": block_embs[target_idx + 1],
                "label": 0.0,
                "block_id": -1,
                "domain": "synthetic_needle"
            })
        
        # Hard Negative 5 and 6 (random blocks)
        for _ in range(2):
            neg_idx = random.randint(0, len(blocks)-1)
            if abs(neg_idx - target_idx) > 10:
                dataset_items.append({
                    "question": query,
                    "query_emb": query_emb,
                    "block_emb": block_embs[neg_idx],
                    "label": 0.0,
                    "block_id": -1,
                    "domain": "synthetic_needle"
                })
                
    return dataset_items

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
                "block_id": blocks[i]["id"], # Use normalized block ID
                "domain": sample.get("domain", "unknown")
            })
            # Negative
            dataset_items.append({
                "question": query_text,
                "query_emb": query_emb,
                "block_emb": n_emb,
                "label": 0.0,
                "block_id": -1, # Dummy ID for negative
                "domain": sample.get("domain", "unknown")
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
    
    # Run garbage collection after preparing the first dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Generate needle data
    needle_data = generate_needle_data(indexer)
    train_data.extend(needle_data)
    
    # Shuffle entire dataset
    random.shuffle(train_data)
    
    # Run garbage collection before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
        main()

        
    # Save metrics (Only hardware for now as main() returns nothing)
    metrics = {"status": "completed"}
    final_output = wrap_result(metrics, mon)
    
    out_path = Path("models/training_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Training profile saved to {out_path}")
