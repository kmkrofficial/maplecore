"""
MAPLE Trainer
==============
Logic for training the MapleNet model from oracle data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .core import MapleNet

logger = logging.getLogger(__name__)


class _MapleDataset(Dataset):
    """Internal dataset for MAPLE training."""
    
    def __init__(self, data: List[Dict]) -> None:
        """
        Initialize dataset.
        
        Args:
            data: List of dicts with 'query_emb', 'block_emb', 'label', 'question', 'block_id'
        """
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        item = self.data[idx]
        combined = torch.cat([item["query_emb"], item["block_emb"]], dim=0)
        label = torch.tensor(item["label"], dtype=torch.float32)
        return combined, label, item["question"], item["block_id"]


def _collate_fn(batch):
    """Internal collate function for DataLoader."""
    embeddings = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    questions = [b[2] for b in batch]
    block_ids = [b[3] for b in batch]
    return embeddings, labels, questions, block_ids


class MapleTrainer:
    """Trainer for MAPLE models."""
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        device: str = "cuda"
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            input_dim: Input dimension (query + block)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            device: Training device
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = torch.device(device)
        self.history_ = {"epoch": [], "loss": [], "val_recall": []}
        
        logger.info(f"MapleTrainer initialized (device={device})")
    
    def compute_recall_at_k(
        self,
        model: MapleNet,
        val_loader: DataLoader,
        k: int = 5
    ) -> float:
        """
        Compute Recall@K on validation set.
        
        Args:
            model: MapleNet model
            val_loader: Validation DataLoader
            k: Number of top predictions to consider
            
        Returns:
            Mean Recall@K score
        """
        model.eval()
        question_preds = defaultdict(list)
        
        with torch.no_grad():
            for embs, labels, questions, block_ids in val_loader:
                embs = embs.to(self.device)
                logits = model(embs)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                for q, bid, prob, lbl in zip(questions, block_ids, probs, labels):
                    question_preds[q].append((bid, float(prob), int(lbl.item())))
        
        recalls = []
        for q, preds in question_preds.items():
            preds.sort(key=lambda x: x[1], reverse=True)
            pred_top_k = set(p[0] for p in preds[:k])
            true_top_k = set(p[0] for p in preds if p[2] == 1)
            
            if len(true_top_k) > 0:
                hits = len(pred_top_k & true_top_k)
                recall = hits / min(len(true_top_k), k)
                recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def train(
        self,
        training_data: List[Dict],
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        val_split: float = 0.2,
        save_path: Optional[Union[str, Path]] = None
    ) -> Tuple[MapleNet, float]:
        """
        Train a MAPLE model.
        
        Args:
            training_data: List of training examples
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            val_split: Validation split ratio
            save_path: Path to save best model
            
        Returns:
            Tuple of (trained model, best recall score)
        """
        # Split by question
        questions = list(set(ex["question"] for ex in training_data))
        np.random.shuffle(questions)
        split_idx = int(len(questions) * (1 - val_split))
        train_q = set(questions[:split_idx])
        val_q = set(questions[split_idx:])
        
        train_data = [ex for ex in training_data if ex["question"] in train_q]
        val_data = [ex for ex in training_data if ex["question"] in val_q]
        
        logger.info(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")
        
        train_loader = DataLoader(
            _MapleDataset(train_data), 
            batch_size=batch_size,
            shuffle=True, 
            collate_fn=_collate_fn
        )
        val_loader = DataLoader(
            _MapleDataset(val_data), 
            batch_size=batch_size,
            shuffle=False, 
            collate_fn=_collate_fn
        )
        
        # Initialize model
        model = MapleNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=lr)
        
        best_recall = 0.0
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for embs, labels, _, _ in train_loader:
                embs = embs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                logits = model(embs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            val_recall = self.compute_recall_at_k(model, val_loader, k=5)
            
            if val_recall > best_recall:
                best_recall = val_recall
                best_state = model.state_dict().copy()
            
            logger.info(
                f"Epoch {epoch+1:2d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Recall@5: {val_recall*100:.1f}%"
            )
            
            self.history_["epoch"].append(epoch + 1)
            self.history_["loss"].append(avg_loss)
            self.history_["val_recall"].append(val_recall)
        
        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Save if path provided
        if save_path is not None:
            model.save(save_path)
        
        logger.info(f"Training complete. Best Recall@5: {best_recall*100:.1f}%")
        return model, best_recall
