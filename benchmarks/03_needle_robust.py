#!/usr/bin/env python3
"""
Benchmark 03: Needle-in-a-Haystack robustness
=============================================

Inserts a single out-of-place fact ("needle") into real book text
(Alice in Wonderland, Gutenberg #11, tiled if needed) at a given depth and
checks whether the block(s) covering the needle are retrieved in the top-k.

Changes vs v0.2.0:
  * Needle templates are HELD OUT: disjoint from the synthetic training
    templates in scripts/train_generalist.py.
  * No padding of the text to chapter boundaries (the search bug it hid is fixed).
  * N trials per (length, depth) cell with fresh values/templates; success rate
    reported with a 95% bootstrap CI.
  * Success = any top-k block overlaps the needle's character span, so a
    needle split across two fixed-size blocks is still counted.
  * BM25 and dense baselines alongside MAPLE.

Output: benchmarks/results/needle_robustness.json (+ heatmap per method)

Usage:
    python benchmarks/03_needle_robust.py
    python benchmarks/03_needle_robust.py --lengths 10000 100000 1000000 --trials 20
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as M
from benchmarks.config import MODEL_PATH, RESULTS_DIR, BGE_MODEL_NAME
from benchmarks.data_loader import get_gutenberg_books
from maplecore import MapleIndexer, MapleScanner, load_model
from maplecore.indexer import Index
from maplecore.utils import _cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONTEXT_LENGTHS = [10_000, 50_000, 100_000, 500_000]
DEPTHS = [0, 25, 50, 75, 100]
CHUNK = 500
TOP_K = 5

# HELD-OUT templates: must not overlap scripts/train_generalist.py TRAIN_NEEDLE_TEMPLATES.
HELDOUT_TEMPLATES: List[Tuple[str, str]] = [
    ("The Project MAPLE secret meeting is held in {city} every {day} at {hour} PM.",
     "Where and when is the secret meeting?"),
    ("Dr. Whitlock's emergency contact number is {num}.",
     "What is Dr. Whitlock's emergency contact number?"),
    ("The rare orchid was last seen growing beside the {city} railway station.",
     "Where was the rare orchid last seen?"),
    ("Inventory record {code} lists exactly {num} brass telescopes.",
     "How many brass telescopes are in the inventory record?"),
]
_CITIES = ["Lisbon", "Nairobi", "Osaka", "Winnipeg", "Valparaiso", "Tbilisi"]
_DAYS = ["Monday", "Wednesday", "Thursday", "Saturday"]


def _needle(rng: random.Random) -> Tuple[str, str]:
    template, question = rng.choice(HELDOUT_TEMPLATES)
    text = template.format(city=rng.choice(_CITIES), day=rng.choice(_DAYS), hour=rng.randint(1, 12),
                           num=rng.randint(1000, 999999), code=str(uuid.UUID(int=rng.getrandbits(128)))[:8])
    return text, question


def _haystack(length: int) -> str:
    alice = get_gutenberg_books([11]).get(11)
    if not alice:
        raise RuntimeError("Gutenberg #11 unavailable; run scripts/download_data.py --only gutenberg")
    start = alice.find("CHAPTER I")
    alice = alice[start if start > 0 else 0:]
    return (alice * (length // len(alice) + 1))[:length]


def _insert(haystack: str, needle: str, depth: int) -> Tuple[str, Tuple[int, int]]:
    idx = int(len(haystack) * depth / 100)
    # move to the next sentence end so the needle does not split a word
    while 0 < idx < len(haystack) and haystack[idx - 1] not in ".!?\n":
        idx += 1
    text = haystack[:idx] + " " + needle + " " + haystack[idx:]
    return text, (idx + 1, idx + 1 + len(needle))


def run(lengths: Sequence[int] = CONTEXT_LENGTHS, depths: Sequence[int] = DEPTHS, trials: int = 10,
        model_path: Optional[Path] = None, embedding_model: str = BGE_MODEL_NAME, seed: int = 0,
        device: Optional[str] = None) -> dict:
    from rank_bm25 import BM25Okapi

    rng = random.Random(seed)
    indexer = MapleIndexer(model_name=embedding_model, device=device)
    model_path = Path(model_path or MODEL_PATH)
    scanner = None
    if model_path.exists():
        scanner = MapleScanner(load_model(model_path, device=indexer.device), device=indexer.device)
    else:
        logger.warning(f"No MAPLE model at {model_path}; baselines only.")

    cells = defaultdict(lambda: defaultdict(list))  # (len, depth) -> method -> [0/1]
    for length in lengths:
        base = _haystack(length)
        base_blocks = indexer.chunk_text(base, CHUNK)
        logger.info(f"length={length:,} chars ({len(base_blocks)} blocks): encoding haystack...")
        base_emb = indexer.encode_blocks(base_blocks, show_progress=False)
        for depth in depths:
            for _ in range(trials):
                needle, question = _needle(rng)
                text, (ns, ne) = _insert(base, needle, depth)
                blocks = indexer.chunk_text(text, CHUNK)
                hit_ids = {b.id for b in blocks if b.start_char < ne and ns < b.end_char}
                # Only blocks at/after the insertion point change; reuse the rest
                # (one block of slack: the leading space may land in the previous block).
                first = max(0, min(hit_ids) - 1)
                emb = torch.cat([base_emb[:first], indexer.encode_blocks(blocks[first:], show_progress=False)])
                index = Index(blocks=blocks, embeddings=emb, chunk_size=CHUNK)
                q = indexer.encode_query(question)

                rankings: Dict[str, List[int]] = {}
                sims = _cosine_similarity(q, emb)
                rankings["dense"] = torch.topk(sims, TOP_K).indices.tolist()
                bm = BM25Okapi([b.text.lower().split() for b in blocks]).get_scores(question.lower().split())
                rankings["bm25"] = np.argsort(-bm)[:TOP_K].tolist()
                if scanner:
                    for strat in ("linear", "adaptive"):
                        rankings[f"maple_{strat}"] = scanner.search(q, index, k=TOP_K, strategy=strat).block_ids[:TOP_K]
                for method, r in rankings.items():
                    cells[(length, depth)][method].append(float(bool(set(r) & hit_ids)))
            summary = {m: f"{np.mean(v) * 100:.0f}%" for m, v in cells[(length, depth)].items()}
            logger.info(f"  depth={depth:>3}%: {summary}")

    methods = sorted({m for c in cells.values() for m in c})
    out = {
        "config": {"lengths": list(lengths), "depths": list(depths), "trials": trials, "chunk_size": CHUNK,
                   "top_k": TOP_K, "templates": "held-out", "model_path": str(model_path) if scanner else None,
                   "seed": seed},
        "cells": [{"length": l, "depth": d, **{m: M.summarize(v) for m, v in c.items()}}
                  for (l, d), c in sorted(cells.items())],
        "overall": {m: M.summarize([x for c in cells.values() for x in c.get(m, [])]) for m in methods},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "needle_robustness.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _plot(out, methods, RESULTS_DIR)
    print("\nOverall success@5:")
    for m, s in out["overall"].items():
        print(f"  {m:<16} {s['mean'] * 100:5.1f}%  [{s['ci_low'] * 100:.1f}, {s['ci_high'] * 100:.1f}]  n={s['n']}")
    print(f"Saved -> {path}")
    return out


def _plot(out: dict, methods: List[str], results_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    lengths, depths = out["config"]["lengths"], out["config"]["depths"]
    for m in methods:
        grid = np.full((len(lengths), len(depths)), np.nan)
        for c in out["cells"]:
            if m in c:
                grid[lengths.index(c["length"]), depths.index(c["depth"])] = c[m]["mean"]
        plt.figure(figsize=(8, 5))
        sns.heatmap(grid, xticklabels=depths, yticklabels=[f"{l // 1000}k" for l in lengths],
                    cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f")
        plt.xlabel("Needle depth (%)")
        plt.ylabel("Haystack length (chars)")
        plt.title(f"Needle success@{TOP_K} — {m} ({out['config']['trials']} trials/cell)")
        plt.tight_layout()
        plt.savefig(results_dir / f"needle_heatmap_{m}.png", dpi=130)
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Needle-in-a-Haystack robustness")
    p.add_argument("--lengths", type=int, nargs="+", default=CONTEXT_LENGTHS)
    p.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    from scripts.runlog import flatten_metrics, record_outputs, start_run
    with start_run("bench", "03_needle", vars(a)) as _run:
        _run.log_metrics(flatten_metrics(run(a.lengths, a.depths, a.trials, a.model, a.embedding_model, a.seed, a.device)))
        record_outputs(_run, RESULTS_DIR / 'needle_robustness.json')
