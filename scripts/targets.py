"""
Soft training targets from oracle samples
=========================================
A target is a probability distribution over one question's blocks, aligned with
``sample["blocks"]`` order. Mass sources (teacher attention) are renormalised so
the few percent of attention that lands on the context become a full
distribution; ``temperature`` < 1 sharpens it. Degenerate samples (no mass, no
gold, field missing) return ``None`` and are skipped by the caller.

Sources:
    last        teacher last-layer, head-mean attention per block (``attention_scores``)
    all_layers  teacher attention averaged over all layers (``attention_scores_all_layers``)
    top5        uniform over the oracle's top-5 blocks
    gold        uniform over dataset-evidence blocks (``gold_block_ids``)
    all_layers_calibrated  all-layer attention minus the attention the same prompt gets with a
                content-free question (ICR-style calibration), floored at 0; falls back to
                ``all_layers`` (and counts it in ``FALLBACKS``) when the calibration is missing or
                removes all mass
A later task adds ``heads``.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence, Tuple

import torch

SOURCES = ("last", "all_layers", "top5", "gold", "all_layers_calibrated")
# how often a source had to fall back to another one (reported by the training scripts)
FALLBACKS: Counter = Counter()


def _mass_target(sample: dict, field: str, temperature: float) -> Optional[torch.Tensor]:
    scores = sample.get(field)
    if not scores:
        return None
    mass = torch.tensor([float(scores.get(str(b["id"]), 0.0)) for b in sample["blocks"]], dtype=torch.float32)
    if mass.max() <= 0:
        return None
    mass = mass.clamp(min=0) ** (1.0 / temperature)
    return mass / mass.sum()


def _uniform_target(sample: dict, ids) -> Optional[torch.Tensor]:
    wanted = {int(i) for i in ids}
    mask = torch.tensor([1.0 if int(b["id"]) in wanted else 0.0 for b in sample["blocks"]], dtype=torch.float32)
    if mask.sum() == 0:
        return None
    return mask / mask.sum()


def block_target(
    sample: dict,
    source: str = "all_layers",
    temperature: float = 1.0,
    head_mass: Optional[torch.Tensor] = None,
    heads: Optional[Sequence[Tuple[int, int]]] = None,
    calib_head_mass: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Distribution over ``sample["blocks"]`` for ``source``; ``None`` if the sample is degenerate."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if source == "last":
        return _mass_target(sample, "attention_scores", temperature)
    if source == "all_layers":
        return _mass_target(sample, "attention_scores_all_layers", temperature)
    if source == "all_layers_calibrated":
        scores = sample.get("attention_scores_all_layers")
        calib = sample.get("attention_scores_all_layers_calib")
        if scores and calib:
            mass = torch.tensor([max(float(scores.get(str(b["id"]), 0.0)) - float(calib.get(str(b["id"]), 0.0)), 0.0)
                                 for b in sample["blocks"]], dtype=torch.float32)
            if mass.max() > 0:
                mass = mass ** (1.0 / temperature)
                return mass / mass.sum()
        FALLBACKS["all_layers_calibrated"] += 1
        return _mass_target(sample, "attention_scores_all_layers", temperature)
    if source == "top5":
        return _uniform_target(sample, sample.get("top_5_block_ids", []))
    if source == "gold":
        return _uniform_target(sample, sample.get("gold_block_ids", []))
    raise ValueError(f"unknown target source '{source}'; expected one of {SOURCES}")
