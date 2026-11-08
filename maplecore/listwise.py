"""
Listwise training for MAPLE scorers
===================================
Each training example is one question with *all* its candidate blocks and a target
distribution over them (teacher attention, gold evidence, ...). The model scores the
whole list and is trained to match the target with a KL divergence (the loss that worked
best for attention distillation, Izacard & Grave 2021) instead of independent binary
labels per (question, block) pair.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from .utils import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class QueryGroup:
    """One question and its candidate blocks."""
    group: str
    q: torch.Tensor                       # [d] query embedding
    blocks: torch.Tensor                  # [n, d] block embeddings
    target: torch.Tensor                  # [n] distribution over blocks (sums to 1)
    gold: Optional[torch.Tensor] = None   # [n] bool, dataset-evidence blocks


def listwise_kl_loss(scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Mean over groups of KL(target || softmax(scores / tau)) over unmasked positions.

    scores, target, mask: [B, n]; padded positions have mask False and never influence the loss.
    """
    s = (scores.float() / tau).masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(s, dim=1).masked_fill(~mask, 0.0)  # 0 (not -inf) keeps gradients finite
    t = target.float() * mask
    t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_t = torch.log(t.clamp_min(1e-12))
    return (t * (log_t - logp)).sum(dim=1).mean()


def _pad_batch(groups: Sequence[QueryGroup], device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Pad a list of groups to [B, n_max, ...] tensors."""
    b, n_max, d = len(groups), max(g.blocks.shape[0] for g in groups), groups[0].q.shape[0]
    q = torch.zeros(b, d)
    blocks = torch.zeros(b, n_max, d)
    target = torch.zeros(b, n_max)
    mask = torch.zeros(b, n_max, dtype=torch.bool)
    for i, g in enumerate(groups):
        n = g.blocks.shape[0]
        q[i] = g.q.float()
        blocks[i, :n] = g.blocks.float()
        target[i, :n] = g.target.float()
        mask[i, :n] = True
    return q.to(device), blocks.to(device), target.to(device), mask.to(device)


class ListwiseTrainer:
    """Trains any scorer taking ``[B, 2*dim]`` (query ‖ block) and returning ``[B]`` logits."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.01, tau: float = 1.0,
                 device: Optional[str] = None, seed: int = 42) -> None:
        self.device = torch.device(resolve_device(device))
        self.model = model.to(self.device)
        self.lr, self.weight_decay, self.tau, self.seed = lr, weight_decay, tau, seed

    # -- scoring
    def _scores(self, q: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        b, n, d = blocks.shape
        x = torch.cat([q.unsqueeze(1).expand(b, n, d), blocks], dim=2).reshape(b * n, 2 * d)
        return self.model(x).reshape(b, n)

    @torch.no_grad()
    def evaluate(self, groups: Sequence[QueryGroup], k: int = 5, batch_groups: int = 64) -> Dict[str, float]:
        """Recall@k of the model's ranking against (a) the target's support and (b) gold evidence."""
        self.model.eval()
        target_hits: List[float] = []
        evidence_hits: List[float] = []
        for start in range(0, len(groups), batch_groups):
            chunk = groups[start:start + batch_groups]
            q, blocks, _, mask = _pad_batch(chunk, self.device)
            scores = self._scores(q, blocks).masked_fill(~mask, float("-inf")).cpu()
            for g, s in zip(chunk, scores):
                n = g.blocks.shape[0]
                kk = min(k, n)
                pred = set(torch.topk(s[:n], kk).indices.tolist())
                pos = (g.target > 0).nonzero().flatten().tolist()
                support = set(pos) if len(pos) <= kk else set(torch.topk(g.target, kk).indices.tolist())
                if support:
                    target_hits.append(len(pred & support) / len(support))
                if g.gold is not None and bool(g.gold.any()):
                    gold = set(g.gold.nonzero().flatten().tolist())
                    evidence_hits.append(len(pred & gold) / len(gold))
        out = {f"val_target_recall@{k}": float(np.mean(target_hits)) if target_hits else float("nan")}
        if evidence_hits:
            out[f"val_evidence_recall@{k}"] = float(np.mean(evidence_hits))
        return out

    # -- training
    def fit(self, train: Sequence[QueryGroup], val: Sequence[QueryGroup], epochs: int = 20,
            batch_groups: int = 32, patience: int = 5) -> dict:
        """Train, keep the best epoch's weights (deep copy), stop after ``patience`` epochs without gain."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        metric = ("val_evidence_recall@5"
                  if val and all(g.gold is not None and bool(g.gold.any()) for g in val)
                  else "val_target_recall@5")
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val, best_epoch, best_state, stale = float("-inf"), 0, None, 0
        history: List[dict] = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            order = rng.permutation(len(train))
            losses = []
            for start in range(0, len(order), batch_groups):
                chunk = [train[i] for i in order[start:start + batch_groups]]
                q, blocks, target, mask = _pad_batch(chunk, self.device)
                loss = listwise_kl_loss(self._scores(q, blocks), target, mask, self.tau)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            result = self.evaluate(val)
            value = result[metric]
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), **result})
            logger.info(f"Epoch {epoch:2d}/{epochs} | loss {history[-1]['train_loss']:.4f} | {metric} {value * 100:.1f}%")
            if value > best_val:
                best_val, best_epoch, stale = value, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())  # state_dict() alone holds live references
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return {"best_epoch": best_epoch, "best_val": best_val, "metric": metric, "history": history}
