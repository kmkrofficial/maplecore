#!/usr/bin/env python3
"""
Synthetic Needle Fine-Tuning
============================
Fine-tunes MAPLE Generalist on synthetic "Needle-in-a-Haystack" data
to fix the "Narrative Bias" issue.
"""

import logging
import random
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleNet, MapleIndexer
from benchmarks.config import MODEL_PATH, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Config
FRANKENSTEIN_URL = "https://www.gutenberg.org/files/84/84-0.txt"
OUTPUT_MODEL = Path("models/maple_robust.pth")
NUM_SAMPLES = 500
EPOCHS = 2
LR = 1e-5
CHUNK_SIZE = 200 # Matching benchmark

class NeedleDataset(Dataset):
    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        self.samples = samples # (query_emb, block_emb, label)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def get_frankenstein_text() -> str:
    """Download or load Frankenstein text."""
    path = DATA_DIR / "frankenstein.txt"
    if not path.exists():
        logger.info("Downloading Frankenstein from Gutenberg...")
        try:
            resp = requests.get(FRANKENSTEIN_URL)
            resp.raise_for_status()
            text = resp.text
            # Simple cleanup
            start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
            end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
            if start != -1 and end != -1:
                text = text[start:end]
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            # Fallback to local Alice if download fails (unlikely in env, but safe)
            return "Alice " * 10000 
            
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_data(indexer: MapleIndexer) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Generate synthetic training data."""
    text = get_frankenstein_text()
    blocks = indexer.chunk_text(text, chunk_size=CHUNK_SIZE)
    # Encode all blocks (we need embeddings for hard negatives)
    logger.info("Encoding blocks for mining...")
    # For speed, just take first 2000 blocks
    blocks = blocks[:2000]
    block_embs = indexer.encode_blocks(blocks, show_progress=True).cpu()
    
    samples = []
    
    logger.info(f"Generating {NUM_SAMPLES} synthetic samples...")
    for _ in tqdm(range(NUM_SAMPLES)):
        # 1. Create Needle pair
        secret = str(uuid.uuid4())[:8]
        needle_text = f"The secret password is {secret}."
        query = "What is the secret password?"
        
        # 2. Embed Query
        query_emb = indexer.encode_query(query).cpu() # [dim]
        
        # 3. Select a random block as base
        target_idx = random.randint(0, len(blocks)-1)
        base_block = blocks[target_idx]
        
        # 4. Inject needle into block text (simulate insertion)
        # We need to re-embed this modify block
        # Insert randomly? Or append?
        # User requested "insert ... into a block".
        # We'll just replace the block text partially or append
        injected_text = base_block.text + f" {needle_text}"
        # Re-embed target
        target_emb = indexer.model.encode(injected_text, convert_to_tensor=True, device="cpu")
        
        # POSITIVE sample
        samples.append((query_emb, target_emb, 1.0))
        
        # HARD NEGATIVE 1: Block with WRONG Needle
        # Inject a different secret
        wrong_secret = str(uuid.uuid4())[:8]
        wrong_text = base_block.text + f" The secret password is {wrong_secret}."
        wrong_emb = indexer.model.encode(wrong_text, convert_to_tensor=True, device="cpu")
        samples.append((query_emb, wrong_emb, 0.0))
        
        # HARD NEGATIVE 2: Pure Block (No needle)
        # Use random neighbor or same block? Same block is best hard negative.
        # But we don't have embedding of same block? Yes we do in `blocks` but need to find index.
        # `base_block` came from `blocks[target_idx]`.
        # But `block_embs` corresponds to `blocks`.
        samples.append((query_emb, block_embs[target_idx], 0.0))

        # RANDOM NEGATIVES (Far away)
        for _ in range(2):
            neg_idx = random.randint(0, len(blocks)-1)
            if abs(neg_idx - target_idx) > 10:
                 samples.append((query_emb, block_embs[neg_idx], 0.0))
                 
    return samples

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Init
    indexer = MapleIndexer(device=device)
    
    # 2. Data
    raw_samples = generate_data(indexer)
    dataset = NeedleDataset(raw_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Model (Load Generalist)
    logger.info(f"Loading generalist model: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error("Generalist model not found!")
        return
        
    model = MapleNet.load(str(MODEL_PATH), device=device)
    model.train()
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    # 5. Loop
    logger.info(f"Starting Fine-Tuning (Epochs={EPOCHS}, LR={LR})")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        
        for query_embs, doc_embs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            query_embs = query_embs.to(device)
            doc_embs = doc_embs.to(device)
            labels = labels.to(device).float()
            
            # Forward
            combined = torch.cat([query_embs, doc_embs], dim=1)
            logits = model(combined)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
    # 6. Save
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_MODEL)
    logger.info(f"Robust model saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    train()
