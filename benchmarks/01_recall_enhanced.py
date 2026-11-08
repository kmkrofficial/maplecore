#!/usr/bin/env python3
"""
Benchmark 01: Retrieval quality on held-out oracle sets
=======================================================

Ranks the in-context blocks of every held-out oracle sample with several
methods and scores the rankings against two kinds of ground truth:

  * oracle  — the teacher's top-5 attention blocks (what MAPLE is trained to
              imitate)                                 -> Oracle-R@k, nDCG@10
  * gold    — dataset evidence: HotpotQA supporting sentences, NarrativeQA
              answer-containing blocks (heuristic)      -> Evidence-R@k,
              AllEvidence@k, MRR@10, AnswerIn@k

Methods: random, lead (document order), BM25, dense cosine (same encoder),
MAPLE (linear scan), and the teacher's own attention (oracle_last /
oracle_all_layers) as a reference point for the evidence metrics.

Held-out inputs (generate with scripts/generate_oracle*.py):
    data/oracle/narrativeqa_test.json
    data/oracle/hotpotqa_validation.json

Output: benchmarks/results/retrieval_quality.json (+ grouped bar chart)

Usage:
    python benchmarks/01_recall_enhanced.py
    python benchmarks/01_recall_enhanced.py --model models/maple_generalist.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as M
from benchmarks.config import MODEL_PATH, RESULTS_DIR, BGE_MODEL_NAME
from maplecore import MapleIndexer, MapleNet
from maplecore.utils import _cosine_similarity
from scripts.oracle_common import load_oracle_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_VALUES = [1, 3, 5, 10]
HELDOUT = {
    "NarrativeQA": Path("data/oracle/narrativeqa_test.json"),
    "HotpotQA": Path("data/oracle/hotpotqa_validation.json"),
}
METHODS = ["random", "lead", "bm25", "dense", "maple", "oracle_last", "oracle_all_layers"]


def _bm25_rank(texts: List[str], query: str) -> List[int]:
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    return list(np.argsort(-bm25.get_scores(query.lower().split()), kind="stable"))


def rank_sample(sample: dict, indexer: MapleIndexer, model: Optional[MapleNet], rng: random.Random) -> Dict[str, List[int]]:
    """Return {method: ranking of block ids}."""
    ids = [b["id"] for b in sample["blocks"]]
    texts = [b["text"] for b in sample["blocks"]]

    q = indexer.encode_query(sample["question"])
    embs = indexer.encode_texts(texts)
    dense_order = torch.argsort(_cosine_similarity(q, embs), descending=True).tolist()

    rankings = {
        "random": rng.sample(ids, len(ids)),
        "lead": ids,
        "bm25": [ids[i] for i in _bm25_rank(texts, sample["question"])],
        "dense": [ids[i] for i in dense_order],
    }
    if model is not None:
        with torch.no_grad():
            x = torch.cat([q.unsqueeze(0).expand(len(ids), -1), embs], dim=1).to(next(model.parameters()).device)
            scores = model(x.float()).cpu()
        rankings["maple"] = [ids[i] for i in torch.argsort(scores, descending=True).tolist()]

    for key, field in [("oracle_last", "attention_scores"), ("oracle_all_layers", "attention_scores_all_layers")]:
        att = sample.get(field)
        if att:
            rankings[key] = sorted(ids, key=lambda b: -att.get(str(b), 0.0))
    return rankings


def score_sample(sample: dict, ranking: List[int]) -> Dict[str, Optional[float]]:
    oracle = set(sample["top_5_block_ids"])
    gold = set(sample.get("gold_block_ids") or [])
    att = {int(k): v for k, v in (sample.get("attention_scores") or {}).items()}
    text_of = {b["id"]: b["text"] for b in sample["blocks"]}
    answers = sample.get("answers") or []

    out: Dict[str, Optional[float]] = {}
    for k in K_VALUES:
        out[f"oracle_recall@{k}"] = M.recall_at_k(ranking, oracle, k, capped=True)
        out[f"evidence_recall@{k}"] = M.recall_at_k(ranking, gold, k)
        out[f"all_evidence@{k}"] = M.hit_all_at_k(ranking, gold, k)
        ctx = " ".join(text_of[b] for b in ranking[:k])
        out[f"answer_in@{k}"] = M.answer_in_context(ctx, answers, min_len=3) if answers else None
    out["oracle_ndcg@10"] = M.ndcg_at_k(ranking, att, 10)
    out["evidence_mrr@10"] = M.mrr_at_k(ranking, gold, 10)
    return out


def evaluate(samples: List[dict], indexer, model, seed: int = 0) -> dict:
    rng = random.Random(seed)
    per_method: Dict[str, Dict[str, list]] = {}
    for n, s in enumerate(samples):
        for method, ranking in rank_sample(s, indexer, model, rng).items():
            scores = score_sample(s, ranking)
            bucket = per_method.setdefault(method, {})
            for metric, v in scores.items():
                bucket.setdefault(metric, []).append(v)
        if (n + 1) % 50 == 0:
            logger.info(f"  {n + 1}/{len(samples)}")

    summary = {m: {metric: M.summarize(v) for metric, v in d.items()} for m, d in per_method.items()}
    # Significance of MAPLE vs the strongest non-oracle baselines on key metrics
    sig = {}
    if "maple" in per_method:
        for base in ["dense", "bm25"]:
            for metric in ["oracle_recall@5", "evidence_recall@5", "answer_in@5"]:
                a, b = per_method["maple"].get(metric), per_method[base].get(metric)
                if a and b:
                    sig[f"maple_vs_{base}/{metric}"] = M.paired_bootstrap_pvalue(a, b)
    return {"summary": summary, "paired_pvalues": sig, "per_item": per_method}


def _print_table(name: str, summary: dict, metrics: List[str]) -> None:
    print(f"\n--- {name} ---")
    print(f"  {'method':<18}" + "".join(f"{m:>20}" for m in metrics))
    for method in METHODS:
        if method not in summary:
            continue
        row = ""
        for m in metrics:
            s = summary[method].get(m)
            if not s or s["n"] == 0:
                row += f"{'n/a':>20}"
            else:
                half = (s["ci_high"] - s["ci_low"]) * 50
                row += f"{s['mean'] * 100:.1f} ± {half:.1f}".rjust(20)
        print(f"  {method:<18}{row}")


def run(model_path: Optional[Path] = None, embedding_model: str = BGE_MODEL_NAME,
        max_samples: Optional[int] = None, device: Optional[str] = None,
        datasets: Optional[List[str]] = None) -> dict:
    model_path = Path(model_path or MODEL_PATH)
    indexer = MapleIndexer(model_name=embedding_model, device=device)
    model = None
    if model_path.exists():
        model = MapleNet.load(model_path, device=indexer.device)
    else:
        logger.warning(f"No MAPLE model at {model_path}; evaluating baselines only.")

    results = {"config": {"model_path": str(model_path), "embedding_model": embedding_model,
                          "k_values": K_VALUES, "max_samples": max_samples}, "datasets": {}}
    for name, path in HELDOUT.items():
        if datasets and name not in datasets:
            continue
        if not path.exists():
            logger.warning(f"{path} missing — skipping {name}")
            continue
        samples = load_oracle_samples(path)[:max_samples]
        logger.info(f"{name}: {len(samples)} held-out samples")
        res = evaluate(samples, indexer, model)
        res["num_samples"] = len(samples)
        res["mean_candidates"] = float(np.mean([len(s["blocks"]) for s in samples]))
        res["frac_with_gold"] = float(np.mean([bool(s.get("gold_block_ids")) for s in samples]))
        results["datasets"][name] = res
        _print_table(name, res["summary"], ["oracle_recall@5", "evidence_recall@5", "all_evidence@5", "answer_in@5", "evidence_mrr@10"])
        print(f"  candidates/sample: {res['mean_candidates']:.1f}   samples with gold evidence: {res['frac_with_gold'] * 100:.0f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "retrieval_quality.json"
    slim = {**results, "datasets": {k: {kk: vv for kk, vv in v.items() if kk != "per_item"}
                                     for k, v in results["datasets"].items()}}
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (RESULTS_DIR / "retrieval_quality_per_item.json").write_text(json.dumps(results, default=str), encoding="utf-8")
    print(f"\nSaved -> {out}")
    if results["datasets"]:
        _plot(results, RESULTS_DIR / "recall_grouped_bar.png")
    return slim


def _plot(results: dict, path: Path, metric: str = "evidence_recall@5") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results["datasets"])
    methods = [m for m in METHODS if any(m in results["datasets"][n]["summary"] for n in names)]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        means, errs = [], []
        for n in names:
            s = results["datasets"][n]["summary"].get(m, {}).get(metric)
            ok = s and s["n"] > 0
            means.append(s["mean"] * 100 if ok else 0)
            errs.append([(s["mean"] - s["ci_low"]) * 100 if ok else 0, (s["ci_high"] - s["mean"]) * 100 if ok else 0])
        ax.bar(np.arange(len(names)) + i * width, means, width, yerr=np.array(errs).T, label=m, capsize=2)
    ax.set_xticks(np.arange(len(names)) + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, fontsize=9)
    ax.set_title(f"{metric} on held-out oracle sets (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Retrieval quality on held-out oracle sets")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    run(a.model, a.embedding_model, a.max_samples, a.device)
