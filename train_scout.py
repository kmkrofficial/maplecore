#!/usr/bin/env python3
"""
train_scout.py
==============
Scout-KV Phase 3: Scout Model Training

Trains a lightweight Scout model to predict which context blocks
are most relevant to a query, based on the oracle attention data.

Model Architecture:
    Input: Concatenated [query_embedding, block_embedding] = 8192
    Hidden: 1024 (ReLU, Dropout 0.3)
    Output: 1 (Logit for binary classification)

Training:
    - 80/20 Train/Val split
    - BCEWithLogitsLoss
    - AdamW (lr=1e-4)
    - 20 epochs, batch size 32

Validation Metric:
    - Recall@5: Are the true top-5 blocks in the predicted top-5?
"""

import json
import sys
import gc
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


# =============================================================================
# Model Definition
# =============================================================================

class ScoutModel(nn.Module):
    """
    Lightweight Scout model for predicting block relevance.
    
    Input: Concatenated query + block embeddings (8192)
    Output: Single logit (binary classification)
    """
    def __init__(self, input_dim=8192, hidden_dim=1024, dropout=0.3):
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
        return x.squeeze(-1)  # [batch] logits


# =============================================================================
# Dataset Definition
# =============================================================================

class ScoutDataset(Dataset):
    """
    Dataset yielding (query_vector, block_vector, binary_label) triplets.
    
    Data is flattened: 50 samples × ~17 blocks = ~850 training points.
    Label: 1 if block is in the Top-5 for its sample, 0 otherwise.
    """
    def __init__(self, data_list):
        """
        Args:
            data_list: List of dicts with keys:
                - query_embedding: [4096] tensor
                - block_embedding: [4096] tensor
                - label: 0 or 1 (binary)
                - question: str (for grouping in validation)
                - block_id: int
        """
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Concatenate query and block embeddings (ensure float32)
        query = item["query_embedding"].float()
        block = item["block_embedding"].float()
        combined = torch.cat([query, block], dim=0)
        label = torch.tensor(item["label"], dtype=torch.float32)
        return combined, label, item["question"], item["block_id"]


def collate_fn(batch):
    """Custom collate to handle question strings."""
    embeddings = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    questions = [b[2] for b in batch]
    block_ids = [b[3] for b in batch]
    return embeddings, labels, questions, block_ids


# =============================================================================
# Data Loading & Preparation
# =============================================================================

def load_and_prepare_data():
    """
    Load training_dataset.pt and oracle_data.json.
    Create binary labels and compute query embeddings.
    
    Returns: List of training examples with all necessary fields.
    """
    print("Loading data...")
    
    # Load pre-computed block embeddings
    block_data = torch.load("training_dataset.pt", weights_only=False)
    print(f"  Loaded {len(block_data)} block embeddings")
    
    # Load oracle data for ground truth top-5
    with open("oracle_data.json", "r", encoding="utf-8") as f:
        oracle_data = json.load(f)
    
    # Create lookup: question -> top_5_block_ids
    top5_lookup = {}
    for sample in oracle_data["samples"]:
        top5_lookup[sample["question"]] = set(sample["top_5_block_ids"])
    
    print(f"  Loaded {len(top5_lookup)} oracle samples with top-5 labels")
    
    # Group block data by question
    question_blocks = defaultdict(list)
    for item in block_data:
        question_blocks[item["question"]].append(item)
    
    # Compute query embeddings
    # For simplicity, we'll use the MEAN of all block embeddings as the "query"
    # This is a reasonable proxy since the prompt includes the question
    # A more sophisticated approach would re-embed just the question text
    print("  Computing query embeddings (mean of block embeddings)...")
    
    query_embeddings = {}
    for question, blocks in question_blocks.items():
        # Stack all block embeddings and compute mean
        all_embeddings = torch.stack([b["embedding"] for b in blocks])
        query_embeddings[question] = all_embeddings.mean(dim=0)
    
    # Build final training data with binary labels
    print("  Creating training examples with binary labels...")
    
    training_examples = []
    
    for question, blocks in question_blocks.items():
        if question not in top5_lookup:
            print(f"  Warning: No oracle data for question '{question[:30]}...'")
            continue
            
        top5_ids = top5_lookup[question]
        query_emb = query_embeddings[question]
        
        for block in blocks:
            # Binary label: 1 if in top-5, 0 otherwise
            is_top5 = 1 if block["block_id"] in top5_ids else 0
            
            training_examples.append({
                "query_embedding": query_emb,
                "block_embedding": block["embedding"],
                "label": is_top5,
                "question": question,
                "block_id": block["block_id"],
                "attention_score": block["label"]  # Keep original score for reference
            })
    
    print(f"  Created {len(training_examples)} training examples")
    
    # Statistics
    positive = sum(1 for ex in training_examples if ex["label"] == 1)
    negative = len(training_examples) - positive
    print(f"  Class balance: {positive} positive, {negative} negative")
    
    return training_examples


def split_data(examples, train_ratio=0.8):
    """
    Split data by QUESTION (not by individual examples).
    This ensures all blocks from a question are in same split.
    """
    # Get unique questions
    questions = list(set(ex["question"] for ex in examples))
    np.random.shuffle(questions)
    
    split_idx = int(len(questions) * train_ratio)
    train_questions = set(questions[:split_idx])
    val_questions = set(questions[split_idx:])
    
    train_data = [ex for ex in examples if ex["question"] in train_questions]
    val_data = [ex for ex in examples if ex["question"] in val_questions]
    
    return train_data, val_data, train_questions, val_questions


# =============================================================================
# Validation Metric: Recall@5
# =============================================================================

def compute_recall_at_k(model, val_loader, device, k=5):
    """
    Compute Recall@K across all validation questions.
    
    For each question:
    1. Get predicted scores for all blocks
    2. Select top-K predicted blocks
    3. Check how many of the true top-K are in predicted top-K
    
    Returns: Average Recall@K across all questions
    """
    model.eval()
    
    # Collect predictions grouped by question
    question_predictions = defaultdict(list)  # question -> [(block_id, pred_score, true_label)]
    
    with torch.no_grad():
        for embeddings, labels, questions, block_ids in val_loader:
            embeddings = embeddings.to(device)
            
            # Get model predictions
            logits = model(embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Store per question
            for i, (q, bid, prob, label) in enumerate(zip(questions, block_ids, probs, labels)):
                question_predictions[q].append((bid, float(prob), int(label.item())))
    
    # Compute Recall@K per question
    recalls = []
    
    for question, predictions in question_predictions.items():
        # Sort by predicted probability (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-K predicted block IDs
        predicted_top_k = set(p[0] for p in predictions[:k])
        
        # Get true top-K block IDs (those with label=1)
        true_top_k = set(p[0] for p in predictions if p[2] == 1)
        
        # Calculate recall: how many true positives are in predicted top-K
        if len(true_top_k) > 0:
            hits = len(predicted_top_k & true_top_k)
            recall = hits / min(len(true_top_k), k)  # Normalize by min(true_count, k)
            recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


# =============================================================================
# Training Loop
# =============================================================================

def train_model(train_data, val_data, epochs=20, batch_size=32, lr=1e-4):
    """
    Train the Scout model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    
    # Create datasets and loaders
    train_dataset = ScoutDataset(train_data)
    val_dataset = ScoutDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = ScoutModel(input_dim=8192, hidden_dim=1024, dropout=0.3)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Training tracking
    best_recall = 0.0
    best_epoch = 0
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for embeddings, labels, _, _ in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Validation
        val_recall = compute_recall_at_k(model, val_loader, device, k=5)
        
        # Track best
        if val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "scout_v1.pth")
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Recall@5: {val_recall*100:.1f}%")
    
    print("-" * 60)
    print(f"Best Recall@5: {best_recall*100:.1f}% (Epoch {best_epoch})")
    print(f"Model saved to: scout_v1.pth")
    
    return model, best_recall


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Scout-KV Phase 3: Model Training")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load and prepare data
    examples = load_and_prepare_data()
    
    # Split data (by question, not by example)
    train_data, val_data, train_q, val_q = split_data(examples, train_ratio=0.8)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)} examples from {len(train_q)} questions")
    print(f"  Val:   {len(val_data)} examples from {len(val_q)} questions")
    
    # Train model
    model, best_recall = train_model(train_data, val_data)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final Recall@5: {best_recall*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
