#!/usr/bin/env python3
"""
Benchmark 05: Chunk-size ablation
=================================

How does block size affect evidence retrieval at a FIXED context budget?

For each chunk size, every HotpotQA (distractor, validation) context is
re-chunked; gold = blocks overlapping a supporting sentence (span-based, so it
is valid for any chunk size). Each method returns as many top blocks as fit in
``--budget-chars`` (default 2,500 chars ≈ 5 × 500-char blocks), so small and
large chunks are compared at equal LLM-context cost.

Note: the MAPLE checkpoint is trained on 500-char blocks; other sizes are
out-of-distribution for it (the baselines have no such dependency).

Metrics: evidence_recall@budget, all_evidence@budget, answer_in@budget,
         blocks per document, embedding memory per document.

Output: benchmarks/results/ablation_chunking.json (+ tradeoff plot)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as M
from benchmarks.config import BGE_MODEL_NAME, MODEL_PATH, RESULTS_DIR
from maplecore import MapleIndexer, MapleNet
from maplecore.utils import _cosine_similarity
from scripts.generate_oracle_hotpot import flatten_hotpot
from scripts.oracle_common import overlapping_block_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK_SIZES = [200, 300, 500, 800, 1200]


def run(chunk_sizes: Sequence[int] = CHUNK_SIZES, budget_chars: int = 2500, max_questions: int = 300,
        model_path: Optional[Path] = None, embedding_model: str = BGE_MODEL_NAME, seed: int = 0,
        device: Optional[str] = None) -> dict:
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ids = list(range(len(ds)))
    random.Random(seed).shuffle(ids)
    items = [ds[i] for i in ids[:max_questions]]

    indexer = MapleIndexer(model_name=embedding_model, device=device)
    model_path = Path(model_path or MODEL_PATH)
    model = MapleNet.load(model_path, device=indexer.device) if model_path.exists() else None

    results = {}
    for size in chunk_sizes:
        k = max(1, budget_chars // size)
        per = defaultdict(lambda: defaultdict(list))
        n_blocks = []
        for item in items:
            text, sent_spans, _ = flatten_hotpot(item)
            blocks = indexer.chunk_text(text, size)
            texts = [b.text for b in blocks]
            gold = set(overlapping_block_ids(blocks, sent_spans))
            n_blocks.append(len(blocks))
            q = indexer.encode_query(item["question"])
            emb = indexer.encode_texts(texts)

            rankings = {
                "dense": torch.argsort(_cosine_similarity(q, emb), descending=True).tolist(),
                "bm25": np.argsort(-BM25Okapi([t.lower().split() for t in texts])
                                   .get_scores(item["question"].lower().split()), kind="stable").tolist(),
            }
            if model is not None:
                with torch.no_grad():
                    x = torch.cat([q.unsqueeze(0).expand(len(texts), -1), emb], dim=1).float()
                    rankings["maple"] = torch.argsort(model(x), descending=True).tolist()
            for m, r in rankings.items():
                per[m]["evidence_recall"].append(M.recall_at_k(r, gold, k))
                per[m]["all_evidence"].append(M.hit_all_at_k(r, gold, k))
                per[m]["answer_in"].append(M.answer_in_context(" ".join(texts[i] for i in r[:k]), [item["answer"]]))

        dim = indexer.get_embedding_dimension()
        results[str(size)] = {
            "k_at_budget": k,
            "mean_blocks_per_doc": float(np.mean(n_blocks)),
            "embedding_kb_per_doc": float(np.mean(n_blocks) * dim * 4 / 1024),
            "methods": {m: {metric: M.summarize(v) for metric, v in d.items()} for m, d in per.items()},
        }
        line = "  ".join(f"{m}: R={d['evidence_recall']['mean'] * 100:.1f}%" for m, d in results[str(size)]["methods"].items())
        logger.info(f"chunk={size:>5} (k={k:>2}, {np.mean(n_blocks):.1f} blocks/doc)  {line}")

    out = {"config": {"chunk_sizes": list(chunk_sizes), "budget_chars": budget_chars, "max_questions": len(items),
                      "embedding_model": embedding_model, "model_path": str(model_path) if model else None,
                      "seed": seed}, "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "ablation_chunking.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _plot(out, RESULTS_DIR / "ablation_tradeoff.png")
    print(f"Saved -> {path}")
    return out


def _plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = out["config"]["chunk_sizes"]
    methods = list(out["results"][str(sizes[0])]["methods"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in methods:
        ys = [out["results"][str(s)]["methods"][m]["evidence_recall"]["mean"] * 100 for s in sizes]
        ax.plot(sizes, ys, marker="o", label=m)
    ax.set_xlabel("Chunk size (chars)")
    ax.set_ylabel(f"Evidence recall @ {out['config']['budget_chars']} chars (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Chunk-size ablation at a fixed context budget (HotpotQA)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chunk-size ablation")
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=CHUNK_SIZES)
    p.add_argument("--budget-chars", type=int, default=2500)
    p.add_argument("--max-questions", type=int, default=300)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    run(a.chunk_sizes, a.budget_chars, a.max_questions, a.model, a.embedding_model, a.seed, a.device)
