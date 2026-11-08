#!/usr/bin/env python3
"""
MAPLE v2 training: listwise KL on soft targets, from cached embeddings
======================================================================
Each oracle sample becomes one ``QueryGroup`` (its question, all in-context blocks and a
target distribution over them). Targets, model and loss are explained in docs/METRICS.md;
the pieces live in ``scripts/targets.py``, ``maplecore/models.py`` and ``maplecore/listwise.py``.

Usage:
    python scripts/train_v2.py --train-oracles data/oracle/hotpotqa_train.json data/oracle/narrativeqa_train.json \
        --output models/maple_v2.pt --target-source all_layers --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleNet, MapleResidualNet, save_model
from maplecore.listwise import ListwiseTrainer, QueryGroup
from scripts.embedding_cache import DEFAULT_ENCODER, build_cache, cache_path_for, load_cache
from scripts.oracle_common import load_oracle_samples
from scripts.runlog import current_run, write_model_card
from scripts.targets import block_target

logger = logging.getLogger(__name__)

MODEL_KINDS = ("residual", "mlp")


def rows_to_groups(rows: Sequence[dict]) -> List[QueryGroup]:
    """Convert flat (group, query_emb, block_emb, label) rows (synthetic needles) into QueryGroups."""
    by_group: Dict[str, List[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    groups = []
    for name, rs in by_group.items():
        label = torch.tensor([float(r["label"]) for r in rs])
        if label.sum() <= 0:
            continue
        groups.append(QueryGroup(
            group=name,
            q=rs[0]["query_emb"].float().cpu(),
            blocks=torch.stack([r["block_emb"].float().cpu() for r in rs]),
            target=label / label.sum(),
            gold=label > 0,
        ))
    return groups


def add_distractors(groups: Sequence[QueryGroup], n: int, seed: int = 0) -> List[QueryGroup]:
    """Append ``n`` random blocks from OTHER groups to every group, with target mass 0.

    Real candidate pools are mostly unrelated text; training only on a question's own ~11 related blocks
    (v1) produced a scorer that collapsed on pools of 100+ blocks. If fewer than ``n`` foreign blocks exist,
    all of them are used. Deterministic given ``seed``.
    """
    if n <= 0:
        return list(groups)
    rng = np.random.default_rng(seed)
    all_blocks = torch.cat([g.blocks for g in groups])
    owner = torch.cat([torch.full((g.blocks.shape[0],), i, dtype=torch.long) for i, g in enumerate(groups)])
    out: List[QueryGroup] = []
    for i, g in enumerate(groups):
        cand = (owner != i).nonzero().flatten()
        take = min(n, cand.numel())
        idx = cand[torch.from_numpy(rng.choice(cand.numel(), size=take, replace=False))] if take else cand[:0]
        extra = all_blocks[idx]
        gold = None if g.gold is None else torch.cat([g.gold, torch.zeros(take, dtype=torch.bool)])
        out.append(QueryGroup(g.group, g.q, torch.cat([g.blocks, extra]),
                              torch.cat([g.target, torch.zeros(take)]), gold))
    return out


def _oracle_groups(oracle: Path, cache_file: Path, encoder: str, target_source: str, temperature: float,
                   skipped: Counter) -> List[QueryGroup]:
    samples = load_oracle_samples(oracle)
    cache = load_cache(cache_file, expected_encoder=encoder)
    ids = [str(s["id"]) for s in samples]
    block_ids = [[int(b["id"]) for b in s["blocks"]] for s in samples]
    if cache["ids"] != ids or cache["block_ids"] != block_ids:
        raise ValueError(f"cache {cache_file} does not match {oracle} (different samples or blocks); rebuild it")
    groups = []
    for s, q, blocks in zip(samples, cache["q"], cache["blocks"]):
        target = block_target(s, target_source, temperature)
        if target is None:
            skipped["degenerate_target"] += 1
            continue
        gold_ids = {int(i) for i in s.get("gold_block_ids", [])}
        gold = torch.tensor([int(b["id"]) in gold_ids for b in s["blocks"]], dtype=torch.bool)
        groups.append(QueryGroup(f"{oracle.stem}::{s['id']}", q.float(), blocks.float(), target,
                                 gold if gold.any() else None))
    return groups


def run(
    train_oracles: Sequence[Path],
    output: Path,
    target_source: str = "all_layers",
    temperature: float = 1.0,
    model_kind: str = "residual",
    encoder: str = DEFAULT_ENCODER,
    epochs: int = 20,
    seed: int = 42,
    val_fraction: float = 0.1,
    needle_samples: int = 0,
    cache_paths: Optional[Sequence[Path]] = None,
    lr: float = 1e-3,
    tau: float = 1.0,
    batch_groups: int = 32,
    patience: int = 5,
    device: Optional[str] = None,
    distractors: int = 0,
) -> dict:
    """Train a scorer on oracle files and save it (plus ``<output stem>.train.json``). Returns the summary."""
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}")
    train_oracles = [Path(p) for p in train_oracles]
    output = Path(output)
    cache_files = [Path(p) for p in cache_paths] if cache_paths else [cache_path_for(p, encoder) for p in train_oracles]
    if len(cache_files) != len(train_oracles):
        raise ValueError("cache_paths must align one-to-one with train_oracles")

    for oracle, cache_file in zip(train_oracles, cache_files):
        if not cache_file.exists():
            logger.info(f"building missing cache {cache_file}")
            build_cache(oracle, cache_file, encoder=encoder, device=device)

    skipped: Counter = Counter()
    groups: List[QueryGroup] = []
    for oracle, cache_file in zip(train_oracles, cache_files):
        groups.extend(_oracle_groups(oracle, cache_file, encoder, target_source, temperature, skipped))
    if needle_samples > 0:
        from maplecore import MapleIndexer
        from scripts.train_generalist import build_needle_rows
        groups.extend(rows_to_groups(build_needle_rows(MapleIndexer(model_name=encoder, device=device),
                                                       needle_samples, seed=seed)))
    groups = add_distractors(groups, distractors, seed=seed)
    if len(groups) < 10:
        raise ValueError(f"only {len(groups)} usable training groups; check the oracle files / target source")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_val = max(1, round(val_fraction * len(groups)))
    val = [groups[i] for i in order[:n_val]]
    train = [groups[i] for i in order[n_val:]]

    dim = groups[0].q.shape[0]
    torch.manual_seed(seed)
    model = MapleResidualNet(dim=dim) if model_kind == "residual" else MapleNet(input_dim=2 * dim, hidden_dim=128)
    trainer = ListwiseTrainer(model, lr=lr, tau=tau, device=device, seed=seed)
    fit = trainer.fit(train, val, epochs=epochs, batch_groups=batch_groups, patience=patience)

    save_model(model, output)
    summary = {
        "train_oracles": [str(p).replace("\\", "/") for p in train_oracles],
        "output": str(output).replace("\\", "/"),
        "target_source": target_source, "temperature": temperature, "model_kind": model_kind,
        "encoder": encoder, "epochs": epochs, "seed": seed, "val_fraction": val_fraction,
        "needle_samples": needle_samples, "lr": lr, "tau": tau, "batch_groups": batch_groups,
        "patience": patience, "dim": dim, "distractors": distractors,
        "groups": {"train": len(train), "val": len(val)},
        "skipped": dict(skipped),
        "parameters": sum(p.numel() for p in model.parameters()),
        **fit,
    }
    output.with_suffix(".train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cur = current_run()
    if cur is not None:  # per-epoch metrics, checkpoint and model card go into the run registry
        cur.config["data_files"] = summary["train_oracles"]
        cur.config["embedding_model"] = encoder
        cur.config["seed"] = seed
        for h in fit["history"]:
            cur.log_metrics({k: v for k, v in h.items() if k != "epoch"}, step=h["epoch"])
        cur.log_metrics({"best_val": fit["best_val"], "best_epoch": fit["best_epoch"], "metric": fit["metric"]})
        cur.add_artifact(output, "checkpoint")
        write_model_card(output, cur, {fit["metric"]: fit["best_val"], "best_epoch": fit["best_epoch"]})
    logger.info(f"best {fit['metric']} = {fit['best_val'] * 100:.1f}% at epoch {fit['best_epoch']} -> {output}")
    return summary


def main(argv: Optional[List[str]] = None) -> dict:
    p = argparse.ArgumentParser(description="Train a MAPLE v2 scorer (listwise KL)")
    p.add_argument("--train-oracles", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-source", default="all_layers")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--model-kind", choices=MODEL_KINDS, default="residual")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--needle-samples", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--batch-groups", type=int, default=32)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--distractors", type=int, default=0,
                   help="Random blocks from other questions appended to every group (target mass 0)")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    out = run(a.train_oracles, a.output, a.target_source, a.temperature, a.model_kind, a.encoder, a.epochs,
              a.seed, a.val_fraction, a.needle_samples, None, a.lr, a.tau, a.batch_groups, a.patience, a.device, a.distractors)
    print(f"best {out['metric']}: {out['best_val'] * 100:.1f}%  ->  {a.output}")
    return out


if __name__ == "__main__":
    from scripts.runlog import start_run

    _argv = sys.argv[1:]
    _tag = "v2"
    for _flag in ("--model-kind", "--target-source", "--seed"):
        if _flag in _argv:
            _tag += f"_{_argv[_argv.index(_flag) + 1]}"
    with start_run("train", _tag, {"argv": _argv}):
        main()
