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
from maplecore.utils import resolve_device
from scripts.oracle_common import load_oracle_samples
from scripts.targets import block_target

logger = logging.getLogger(__name__)


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
) -> dict:
    """Fine-tune ``base_model`` on teacher-attention targets and save it to ``output_dir``."""
    from sentence_transformers import SentenceTransformer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)

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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)

    def embed(texts: List[str]) -> torch.Tensor:
        feats = model.tokenize(texts)
        feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}  # some entries are strings
        return model(feats)["sentence_embedding"]

    epoch_loss = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(groups))
        losses = []
        for start in range(0, len(order), batch_questions):
            chunk = [groups[i] for i in order[start:start + batch_questions]]
            q = torch.nn.functional.normalize(embed([prefix + g[0] for g in chunk]), dim=-1)
            all_blocks = torch.nn.functional.normalize(embed([t for g in chunk for t in g[1]]), dim=-1)
            n_max = max(len(g[1]) for g in chunk)
            scores = torch.full((len(chunk), n_max), 0.0, device=device)
            target = torch.zeros(len(chunk), n_max, device=device)
            mask = torch.zeros(len(chunk), n_max, dtype=torch.bool, device=device)
            offset = 0
            for i, g in enumerate(chunk):
                n = len(g[1])
                scores[i, :n] = all_blocks[offset:offset + n] @ q[i]
                target[i, :n] = g[2].to(device)
                mask[i, :n] = True
                offset += n
            loss = listwise_kl_loss(scores, target, mask, tau=tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        epoch_loss.append(float(np.mean(losses)))
        logger.info(f"Epoch {epoch}/{epochs} | listwise KL {epoch_loss[-1]:.4f}")

    model.eval()
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    return {"base_model": base_model, "output_dir": str(output_dir).replace("\\", "/"), "epochs": epochs,
            "loss": epoch_loss, "groups": len(groups), "target_source": target_source, "tau": tau, "lr": lr,
            "seed": seed, "batch_questions": batch_questions}


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
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    out = distill_biencoder(a.base_model, a.train_oracles, a.output, a.target_source, a.epochs, a.lr,
                            a.batch_questions, a.tau, a.seed, a.max_samples, a.device, a.temperature)
    print(f"saved {out['output_dir']} (final loss {out['loss'][-1]:.4f})")
    return out


if __name__ == "__main__":
    from scripts.runlog import start_run

    with start_run("train", "biencoder", {"argv": sys.argv[1:], "encoder": "maple-embed"}) as _run:
        _res = main()
        for _i, _l in enumerate(_res["loss"], 1):
            _run.log_metrics({"train_loss": _l}, step=_i)
