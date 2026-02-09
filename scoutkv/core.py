"""
Scout-KV Core Module
====================
Contains the ScoutBGE neural network model definition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ScoutBGE(nn.Module):
    """
    Scout model for BGE embeddings.
    
    Architecture: MLP (768 -> 128 -> 1)
    Input: Concatenated [query_embedding, block_embedding]
    Output: Relevance logit (scalar)
    """
    
    def __init__(
        self, 
        input_dim: int = 768, 
        hidden_dim: int = 128, 
        dropout: float = 0.3
    ) -> None:
        """
        Initialize ScoutBGE model.
        
        Args:
            input_dim: Input dimension (query + block embeddings)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        logger.debug(f"ScoutBGE initialized: {input_dim} -> {hidden_dim} -> 1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, input_dim]
            
        Returns:
            Logits tensor of shape [batch]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model weights to disk.
        
        Args:
            path: Path to save weights (.pth file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(
        cls, 
        path: Union[str, Path], 
        device: str = "cuda",
        **kwargs
    ) -> "ScoutBGE":
        """
        Load model weights from disk.
        
        Args:
            path: Path to weights file (.pth)
            device: Device to load model to
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded ScoutBGE model
        """
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
