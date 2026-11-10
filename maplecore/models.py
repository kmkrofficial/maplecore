"""
MAPLE model registry
====================
Scorer architectures plus a small save/load layer so checkpoints are
self-describing (class + config + weights) instead of bare state dicts.

* ``MapleNet`` (core.py)   - v1 MLP on ``[query ‖ block]``; legacy checkpoints are plain state dicts.
* ``MapleResidualNet``     - cosine similarity plus a learned correction. Untrained, it ranks
  exactly like cosine similarity (the dense-retrieval baseline), so training can only
  move it away from that starting point.

Checkpoint format v2 (loads with ``weights_only=True``)::

    {"format": "maple-v2", "class": "MapleResidualNet", "config": {...}, "state_dict": {...}}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import MapleNet
from .utils import resolve_device

logger = logging.getLogger(__name__)


class MapleResidualNet(nn.Module):
    """
    ``logit = scale * cos(q, b) + mlp(LayerNorm([q, b, q*b, |q-b|]))``

    The input is the usual ``[query ‖ block]`` concatenation (``2 * dim``). The final MLP layer
    is zero-initialised, so the untrained model equals ``scale * cosine``. The LayerNorm keeps
    the MLP's input well-scaled for unit-norm sentence embeddings (raw entries are ~0.05).
    """

    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.1, init_scale: float = 20.0) -> None:
        super().__init__()
        self.dim = dim
        self.input_dim = 2 * dim
        self.hidden_dim, self.dropout_rate, self.init_scale = hidden_dim, dropout, init_scale
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @property
    def config(self) -> dict:
        return {"dim": self.dim, "hidden_dim": self.hidden_dim, "dropout": self.dropout_rate,
                "init_scale": self.init_scale}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, b = x[:, : self.dim], x[:, self.dim:]
        cos = (F.normalize(q, dim=1) * F.normalize(b, dim=1)).sum(dim=1)
        feats = torch.cat([q, b, q * b, (q - b).abs()], dim=1)
        return self.scale * cos + self.mlp(feats).squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


_REGISTRY: Dict[str, Type[nn.Module]] = {"MapleNet": MapleNet, "MapleResidualNet": MapleResidualNet}


def save_model(model: nn.Module, path: Union[str, Path]) -> None:
    """Save ``model`` with its class name and config so ``load_model`` can rebuild it."""
    name = type(model).__name__
    if name not in _REGISTRY:
        raise ValueError(f"{name} is not a registered MAPLE model ({sorted(_REGISTRY)})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "maple-v2", "class": name, "config": model.config,
                "state_dict": model.state_dict()}, path)
    logger.info(f"Model ({name}) saved to {path}")


def load_model(path: Union[str, Path], device: Optional[str] = None,
               expected_input_dim: Optional[int] = None) -> nn.Module:
    """Load a v2 checkpoint or a legacy ``MapleNet`` state dict; returns the model in eval mode."""
    device = resolve_device(device)
    blob = torch.load(Path(path), map_location=device, weights_only=True)
    if isinstance(blob, dict) and blob.get("format") == "maple-v2":
        name = blob["class"]
        if name not in _REGISTRY:
            raise ValueError(f"checkpoint {path} holds unknown model class '{name}' ({sorted(_REGISTRY)})")
        model = _REGISTRY[name](**blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.to(device).eval()
    else:  # legacy: plain MapleNet state dict
        model = MapleNet.load(path, device=device)
    if expected_input_dim is not None and model.input_dim != expected_input_dim:
        raise ValueError(f"checkpoint {path} has input_dim={model.input_dim}, expected {expected_input_dim} "
                         "(was it trained with a different encoder?)")
    return model


def export_to_onnx(model: nn.Module, path: Union[str, Path], input_dim: Optional[int] = None) -> None:
    """Export any registered scorer to ONNX with a dynamic batch axis (input ``input``, output ``output``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    input_dim = input_dim or model.input_dim
    was_training = model.training
    model.eval()
    param = next(model.parameters())
    dummy = torch.randn((1, input_dim), device=param.device, dtype=param.dtype)
    torch.onnx.export(
        model, dummy, path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    if was_training:
        model.train()
    logger.info(f"ONNX model exported to {path}")
