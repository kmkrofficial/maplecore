#!/usr/bin/env python3
"""
Query-focused head selection (QRHead-style)
===========================================
Averaging attention over all heads dilutes the signal: only a few heads actually look at the
evidence (Retrieval Head, Wu et al. 2025; QRHead, Zhang et al. 2025). For every (layer, head) we
measure which share of its attention mass over the context blocks lands on the gold-evidence blocks,
averaged over oracle samples that have gold evidence, and keep the top-k heads.

Selection must use a TRAIN split only (held-out files are never used to choose anything).

Usage:
    python scripts/select_heads.py data/oracle/v2/hotpotqa_train.json --k 16 --out data/oracle/heads_llama3-8b.json
Needs ``<oracle stem>.heads.pt`` written by ``generate_oracle*.py --save-heads``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def score_heads(samples: Sequence[dict], heads_file: dict) -> torch.Tensor:
    """Mean over samples-with-gold of mass(gold blocks) / mass(all blocks), per head: ``[layers, heads]``."""
    total, n = None, 0
    for s in samples:
        sid = str(s["id"])
        gold = {int(i) for i in s.get("gold_block_ids", [])}
        if not gold or sid not in heads_file["mass"]:
            continue
        cols = [j for j, b in enumerate(heads_file["block_order"][sid]) if int(b) in gold]
        if not cols:
            continue
        mass = heads_file["mass"][sid].float()                     # [layers, heads, n_blocks]
        ratio = mass[..., cols].sum(-1) / mass.sum(-1).clamp_min(1e-12)
        total = ratio if total is None else total + ratio
        n += 1
    if n == 0:
        raise ValueError("no sample has both gold evidence and saved head mass; cannot score heads")
    return total / n


def select_top_heads(scores: torch.Tensor, k: int = 16) -> List[Tuple[int, int]]:
    """The ``k`` best (layer, head) pairs by score, best first (``k`` is clipped to the number of heads)."""
    n_heads = scores.shape[1]
    order = torch.argsort(scores.flatten(), descending=True)[: min(k, scores.numel())]
    return [(int(i) // n_heads, int(i) % n_heads) for i in order]


def main(argv: Optional[List[str]] = None) -> dict:
    p = argparse.ArgumentParser(description="Select query-focused attention heads on a train oracle split")
    p.add_argument("oracle", type=Path)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    raw = json.loads(a.oracle.read_text(encoding="utf-8"))
    split = raw.get("metadata", {}).get("split")
    if split != "train":
        raise ValueError(f"head selection must use a train split, got split={split!r} ({a.oracle})")
    heads_path = a.oracle.with_suffix(".heads.pt")
    if not heads_path.exists():
        raise FileNotFoundError(f"{heads_path} missing; generate the oracle with --save-heads")
    heads_file = torch.load(heads_path, map_location="cpu", weights_only=True)

    scores = score_heads(raw["samples"], heads_file)
    chosen = select_top_heads(scores, a.k)
    result = {
        "teacher": raw.get("metadata", {}).get("teacher_model"),
        "k": a.k,
        "heads": [[l, h] for l, h in chosen],
        "scores_top": [round(float(scores[l, h]), 4) for l, h in chosen],
        "score_mean_all_heads": round(float(scores.mean()), 4),
        "source": str(a.oracle).replace("\\", "/"),
        "samples_used": sum(1 for s in raw["samples"] if s.get("gold_block_ids")),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"selected {len(chosen)} heads (top score {result['scores_top'][0]:.3f} vs mean {result['score_mean_all_heads']:.3f}) -> {a.out}")
    return result


if __name__ == "__main__":
    from scripts.runlog import start_run

    with start_run("oracle", "select_heads", {"argv": sys.argv[1:]}):
        main()
