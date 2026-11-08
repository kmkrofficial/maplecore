#!/usr/bin/env python3
"""
Compare two oracle files sample-by-sample (e.g. fast vs eager attention).

Reports, over samples present in both files:
  * exact top-5 set agreement and mean top-5 Jaccard
  * max / mean absolute difference of per-block attention scores
  * Spearman correlation of the per-block last-layer scores

Usage:
    python scripts/compare_oracles.py data/oracle/hotpotqa_validation.json data/oracle/_fastcheck.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.oracle_common import load_oracle_samples


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(ref_path: Path, new_path: Path) -> dict:
    ref = {s["id"]: s for s in load_oracle_samples(ref_path)}
    new = {s["id"]: s for s in load_oracle_samples(new_path)}
    common = [i for i in new if i in ref]
    if not common:
        raise SystemExit("No overlapping sample ids.")

    exact, jacc, max_diff, mean_diff, rho, max_diff_all = [], [], [], [], [], []
    for i in common:
        a, b = ref[i], new[i]
        ta, tb = set(a["top_5_block_ids"]), set(b["top_5_block_ids"])
        exact.append(ta == tb)
        jacc.append(len(ta & tb) / len(ta | tb))
        keys = sorted(a["attention_scores"], key=int)
        va = np.array([a["attention_scores"][k] for k in keys])
        vb = np.array([b["attention_scores"].get(k, np.nan) for k in keys])
        d = np.abs(va - vb)
        max_diff.append(np.nanmax(d))
        mean_diff.append(np.nanmean(d))
        rho.append(_spearman(va, vb))
        if a.get("attention_scores_all_layers") and b.get("attention_scores_all_layers"):
            wa = np.array([a["attention_scores_all_layers"][k] for k in keys])
            wb = np.array([b["attention_scores_all_layers"][k] for k in keys])
            max_diff_all.append(np.abs(wa - wb).max())

    out = {
        "samples_compared": len(common),
        "top5_exact_match": float(np.mean(exact)),
        "top5_jaccard_mean": float(np.mean(jacc)),
        "last_layer_abs_diff_max": float(np.max(max_diff)),
        "last_layer_abs_diff_mean": float(np.mean(mean_diff)),
        "all_layers_abs_diff_max": float(np.max(max_diff_all)) if max_diff_all else None,
        "spearman_mean": float(np.nanmean(rho)),
    }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare two oracle files")
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.reference, a.candidate), indent=2))
