"""
MAPLE Core Module
==================
Contains the MapleNet neural network model definition.

MapleNet is a lightweight MLP that scores block relevance by learning
from LLM attention patterns (Memory-Aware Predictive Loading).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MapleNet(nn.Module):
    """
    MAPLE relevance scoring network.
    
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
        Initialize MapleNet model.
        
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
        
        logger.debug(f"MapleNet initialized: {input_dim} -> {hidden_dim} -> 1")
    
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
        
    def export_to_onnx(
        self, 
        save_path: Union[str, Path], 
        dummy_input_shape: Optional[tuple] = None
    ) -> None:
        """
        Export the MAPLE model to ONNX format for production inference.
        
        Args:
            save_path: Path to save the .onnx file
            dummy_input_shape: Shape of the input tensor. Defaults to
                               (1, self.input_dim), i.e. concatenated
                               (query + block) embeddings.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if dummy_input_shape is None:
            dummy_input_shape = (1, self.input_dim)
        was_training = self.training
        self.eval()
        
        # We match dtype with model weight dtype
        dummy_input = torch.randn(dummy_input_shape, device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        if was_training:
            self.train()
        logger.info(f"ONNX Model exported successfully to {save_path}")
    
    @classmethod
    def load(
        cls, 
        path: Union[str, Path], 
        device: Optional[str] = None,
        **kwargs
    ) -> "MapleNet":
        """
        Load model weights from disk.
        
        Args:
            path: Path to weights file (.pth)
            device: Device to load model to (None = auto-detect)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded MapleNet model
        """
        from .utils import resolve_device
        device = resolve_device(device)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Infer architecture from the saved weights so callers cannot mismatch it
        if "fc1.weight" in state_dict:
            hidden, inferred_input_dim = state_dict["fc1.weight"].shape
            if kwargs.get("input_dim", inferred_input_dim) != inferred_input_dim:
                logger.warning(
                    f"input_dim={kwargs['input_dim']} requested but checkpoint has "
                    f"input_dim={inferred_input_dim} (encoder mismatch?). Using the checkpoint's."
                )
            kwargs["input_dim"] = int(inferred_input_dim)
            kwargs["hidden_dim"] = int(hidden)

        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
