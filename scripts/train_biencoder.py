#!/usr/bin/env python3
"""
MAPLE-Embed: distil teacher attention into the bi-encoder itself
================================================================
Instead of training a separate scorer on frozen embeddings, fine-tune the sentence encoder so that, for every
question, softmax(cos(question, block) / tau) over its blocks matches the teacher-attention target
(listwise KL, Izacard & Grave 2021). The result is an ordinary sentence-transformers model directory: it can be
used as ``MapleIndexer(model_name=<dir>)``, as the encoder of a ``dense`` ranker, or as the encoder for a
residual MAPLE head.

Usage:
    python scripts/train_biencoder.py --train-oracles data/oracle/hotpotqa_train.json data/oracle/narrativeqa_train.json \
        --output models/maple_embed_b1536_s42 --epochs 2 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore.indexer import query_prefix_for
from maplecore.listwise import listwise_kl_loss
from maplecore.utils import limit_gpu_memory, resolve_device
from scripts.oracle_common import load_oracle_samples
from scripts.targets import block_target

logger = logging.getLogger(__name__)


def listwise_batch(q: torch.Tensor, blocks: torch.Tensor, sizes: Sequence[int], targets: Sequence[torch.Tensor],
                   cross_negatives: bool = False):
    """Score rows for listwise KL. ``q`` (B, d) and ``blocks`` (sum(sizes), d) are unit-norm; question i owns
    ``sizes[i]`` consecutive blocks with soft target ``targets[i]``.

    * ``cross_negatives=False``: row i holds only its own blocks (padded to max(sizes), masked).
    * ``cross_negatives=True``: row i scores every block in the batch; other questions' blocks are negatives with
      target 0, so the encoder must also separate documents, not only blocks inside one document.
    Returns (scores, target, mask)."""
    device = q.device
    if cross_negatives:
        scores = q @ blocks.T
        target = torch.zeros_like(scores)
        mask = torch.ones_like(scores, dtype=torch.bool)
        offset = 0
        for i, n in enumerate(sizes):
            target[i, offset:offset + n] = targets[i].to(device)
            offset += n
        return scores, target, mask
    n_max = max(sizes)
    scores = torch.zeros(len(sizes), n_max, device=device, dtype=q.dtype)
    target = torch.zeros(len(sizes), n_max, device=device)
    mask = torch.zeros(len(sizes), n_max, dtype=torch.bool, device=device)
    offset = 0
    for i, n in enumerate(sizes):
        scores[i, :n] = blocks[offset:offset + n] @ q[i]
        target[i, :n] = targets[i].to(device)
        mask[i, :n] = True
        offset += n
    return scores, target, mask


def freeze_all_but_last_layers(model, n: int) -> int:
    """Freeze every parameter of a sentence-transformers BERT-style model except its top ``n`` transformer layers.
    Returns the number of trainable parameters."""
    for prm in model.parameters():
        prm.requires_grad = False
    layers = model[0].auto_model.encoder.layer
    if not 0 < n <= len(layers):
        raise ValueError(f"train_last_layers must be in 1..{len(layers)}, got {n}")
    for prm in layers[-n:].parameters():
        prm.requires_grad = True
    return sum(prm.numel() for prm in model.parameters() if prm.requires_grad)


def base_kl_loss(scores: torch.Tensor, base_scores: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    """KL(base || student) between per-row softmax(score / tau) distributions over the unmasked entries: keeps the
    fine-tuned encoder's ranking of each question's candidates close to the frozen base encoder's."""
    neg = torch.finfo(scores.dtype).min
    log_p = torch.log_softmax((scores / tau).masked_fill(~mask, neg), dim=-1)
    log_b = torch.log_softmax((base_scores.to(scores.dtype) / tau).masked_fill(~mask, neg), dim=-1)
    kl = (log_b.exp() * (log_b - log_p)).masked_fill(~mask, 0.0).sum(-1)
    return kl.mean()


def distill_biencoder(
    base_model: str,
    train_oracles: Sequence[Path],
    output_dir: Path,
    target_source: str = "all_layers",
    epochs: int = 2,
    lr: float = 2e-5,
    batch_questions: int = 8,
    tau: float = 0.05,
    seed: int = 42,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
    temperature: float = 1.0,
    cache_folder: Optional[str] = None,
    cross_negatives: bool = False,
    max_gpu_mem_frac: Optional[float] = None,
    train_last_layers: Optional[int] = None,
    base_kl: float = 0.0,
) -> dict:
    """Fine-tune ``base_model`` on teacher-attention targets and save it to ``output_dir``."""
    from sentence_transformers import SentenceTransformer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)
    limit_gpu_memory(max_gpu_mem_frac, device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    groups = []  # (query text, [block texts], target tensor)
    for oracle in train_oracles:
        for s in load_oracle_samples(Path(oracle)):
            target = block_target(s, target_source, temperature)
            if target is None:
                continue
            groups.append((s["question"], [b["text"] for b in s["blocks"]], target))
            if max_samples and len(groups) >= max_samples:
                break
    if not groups:
        raise ValueError("no usable training groups (every sample has a degenerate target)")

    model = SentenceTransformer(base_model, device=device, cache_folder=cache_folder)
    prefix = query_prefix_for(base_model)
    n_trainable = (freeze_all_but_last_layers(model, train_last_layers) if train_last_layers
                   else sum(prm.numel() for prm in model.parameters()))
    base = None
    if base_kl > 0:
        base = SentenceTransformer(base_model, device=device, cache_folder=cache_folder).eval()
        for prm in base.parameters():
            prm.requires_grad = False
    opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad], lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)

    def embed(texts: List[str], m=None) -> torch.Tensor:
        m = m or model
        feats = m.tokenize(texts)
        feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}  # some entries are strings
        return m(feats)["sentence_embedding"]

    epoch_loss = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(groups))
        losses = []
        for start in range(0, len(order), batch_questions):
            chunk = [groups[i] for i in order[start:start + batch_questions]]
            q = torch.nn.functional.normalize(embed([prefix + g[0] for g in chunk]), dim=-1)
            all_blocks = torch.nn.functional.normalize(embed([t for g in chunk for t in g[1]]), dim=-1)
            scores, target, mask = listwise_batch(q, all_blocks, [len(g[1]) for g in chunk],
                                                  [g[2] for g in chunk], cross_negatives)
            loss = listwise_kl_loss(scores, target, mask, tau=tau)
            if base is not None:
                with torch.no_grad():
                    bq = torch.nn.functional.normalize(embed([prefix + g[0] for g in chunk], base), dim=-1)
                    bb = torch.nn.functional.normalize(embed([t for g in chunk for t in g[1]], base), dim=-1)
                    base_scores, _, _ = listwise_batch(bq, bb, [len(g[1]) for g in chunk], [g[2] for g in chunk],
                                                       cross_negatives)
                loss = loss + base_kl * base_kl_loss(scores, base_scores, mask, tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        epoch_loss.append(float(np.mean(losses)))
        peak = torch.cuda.max_memory_reserved() / 2**30 if str(device).startswith("cuda") else 0.0
        logger.info(f"Epoch {epoch}/{epochs} | listwise KL {epoch_loss[-1]:.4f} | peak GPU reserved {peak:.2f} GiB")

    model.eval()
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    return {"base_model": base_model, "output_dir": str(output_dir).replace("\\", "/"), "epochs": epochs,
            "loss": epoch_loss, "groups": len(groups), "target_source": target_source, "tau": tau, "lr": lr,
            "seed": seed, "batch_questions": batch_questions, "cross_negatives": cross_negatives,
            "max_gpu_mem_frac": max_gpu_mem_frac, "train_last_layers": train_last_layers, "base_kl": base_kl,
            "trainable_params": n_trainable,
            "peak_gpu_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                      if str(device).startswith("cuda") else 0.0)}


def main(argv: Optional[List[str]] = None) -> dict:
    p = argparse.ArgumentParser(description="Distil teacher attention into a bi-encoder (MAPLE-Embed)")
    p.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--train-oracles", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-source", default="all_layers")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-questions", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--cross-negatives", action="store_true",
                   help="also score the other questions' blocks in each batch as zero-target negatives")
    p.add_argument("--max-gpu-mem-frac", type=float, default=None,
                   help="cap the CUDA allocator (e.g. 0.75) so Windows never pages VRAM into shared memory")
    p.add_argument("--train-last-layers", type=int, default=None,
                   help="fine-tune only the top N transformer layers (others frozen)")
    p.add_argument("--base-kl", type=float, default=0.0,
                   help="weight of a KL term keeping rankings close to the frozen base encoder")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    out = distill_biencoder(a.base_model, a.train_oracles, a.output, a.target_source, a.epochs, a.lr,
                            a.batch_questions, a.tau, a.seed, a.max_samples, a.device, a.temperature,
                            cross_negatives=a.cross_negatives, max_gpu_mem_frac=a.max_gpu_mem_frac,
                            train_last_layers=a.train_last_layers, base_kl=a.base_kl)
    print(f"saved {out['output_dir']} (final loss {out['loss'][-1]:.4f})")
    return out


if __name__ == "__main__":
    from scripts.runlog import start_run

    with start_run("train", "biencoder", {"argv": sys.argv[1:], "encoder": "maple-embed"}) as _run:
        _res = main()
        for _i, _l in enumerate(_res["loss"], 1):
            _run.log_metrics({"train_loss": _l}, step=_i)
