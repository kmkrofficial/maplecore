"""
MAPLE Utilities
================
Helper functions for device handling, logging, and common operations.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Union

import torch


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for MAPLE.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Normalise a device argument to a string.

    ``None`` (or ``"auto"``) picks CUDA when available, otherwise CPU.
    Requesting ``"cuda"`` on a machine without CUDA falls back to CPU
    with a warning instead of crashing.
    """
    if device is None or str(device) == "auto":
        return str(get_device())
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger(__name__).warning(
            f"Device '{device}' requested but CUDA is unavailable; using CPU."
        )
        return "cpu"
    return device


def _ensure_utf8() -> None:
    """Ensure stdout/stderr use UTF-8 encoding."""
    if hasattr(sys.stdout, 'reconfigure'):
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')


def _compute_entropy(probs: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute entropy of a probability distribution.
    
    Args:
        probs: Probability tensor (normalized)
        eps: Small epsilon to avoid log(0)
        
    Returns:
        Entropy value (float)
    """
    probs = probs.clamp(min=eps)
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()


def _cosine_similarity(
    query: torch.Tensor, 
    blocks: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between query and blocks.
    
    Args:
        query: Query embedding [dim]
        blocks: Block embeddings [N, dim]
        
    Returns:
        Similarity scores [N]
    """
    query_norm = query / query.norm()
    block_norms = blocks / blocks.norm(dim=1, keepdim=True)
    return torch.matmul(block_norms, query_norm)
