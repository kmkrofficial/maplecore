#!/usr/bin/env python3
"""
Benchmark 01 (Enhanced): Multi-Dataset Recall
===============================================

Compares three retrieval methods across two datasets:
  - BM25  (keyword search baseline)
  - RAG   (cosine similarity with BGE embeddings)
  - MAPLE (learned MLP scorer)

Metrics: Recall@1, @5, @10
Datasets: NarrativeQA, HotpotQA (distractor)

Output:
  - results/recall_enhanced.json
  - results/recall_grouped_bar.png
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleNet, MapleIndexer
from maplecore.utils import _cosine_similarity as cosine_similarity
from benchmarks.config import DATA_DIR, RESULTS_DIR, BGE_MODEL_NAME, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_VALUES = [1, 5, 10]
CHUNK_SIZE = 500   # characters per block

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================================================================
# BM25 Baseline
# ===================================================================

class BM25Retriever:
    """Thin wrapper around rank_bm25 for block retrieval."""

    def __init__(self, block_texts: List[str]):
        from rank_bm25 import BM25Okapi
        tokenized = [text.lower().split() for text in block_texts]
        self.bm25 = BM25Okapi(tokenized)

    def rank(self, query: str) -> List[int]:
        """Return block indices sorted by BM25 score (descending)."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return list(np.argsort(scores)[::-1])


# ===================================================================
# Dataset Adapters
# ===================================================================

def _load_narrativeqa_samples(max_samples: int = 200) -> List[Dict]:
    """
    Load NarrativeQA samples with oracle ground truth.

    Uses oracle_data.json to get ground-truth block IDs.
    """
    from benchmarks.data_loader import get_narrative_qa

    oracle_path = DATA_DIR / ".." / "data" / "oracle_data.json"
    if not oracle_path.resolve().exists():
        # Try alternate path
        oracle_path = Path("data/oracle_data.json")
    if not oracle_path.exists():
        logger.warning("oracle_data.json not found, skipping NarrativeQA")
        return []

    with open(oracle_path, "r", encoding="utf-8") as f:
        oracle = json.load(f)

    samples = []
    for s in oracle["samples"][:max_samples]:
        block_texts = list(s["all_block_texts"].values())
        block_ids = list(s["all_block_texts"].keys())
        if len(block_texts) < 5:
            continue
        samples.append({
            "question": s["question"],
            "block_texts": block_texts,
            "block_ids": block_ids,
            "ground_truth_ids": set(str(tid) for tid in s["top_5_block_ids"][:5]),
            "dataset": "NarrativeQA",
        })

    logger.info(f"NarrativeQA: {len(samples)} samples from oracle data")
    return samples


def _load_hotpotqa_samples(max_samples: int = 200) -> List[Dict]:
    """
    Load HotpotQA distractor samples.

    Ground truth = paragraphs from the supporting facts' titles.
    """
    from benchmarks.data_loader import get_hotpot_qa

    dataset = get_hotpot_qa("validation", max_samples=max_samples)

    samples = []
    for item in dataset:
        question = item["question"]
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        sup_titles = set(item["supporting_facts"]["title"])

        # Each title's sentences form a "block"
        block_texts = []
        ground_truth_indices = set()
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            block_texts.append(f"{title}: " + " ".join(sents))
            if title in sup_titles:
                ground_truth_indices.add(str(i))

        if len(block_texts) < 3 or not ground_truth_indices:
            continue

        samples.append({
            "question": question,
            "block_texts": block_texts,
            "block_ids": [str(i) for i in range(len(block_texts))],
            "ground_truth_ids": ground_truth_indices,
            "dataset": "HotpotQA",
        })

    logger.info(f"HotpotQA: {len(samples)} samples")
    return samples


# ===================================================================
# Evaluation
# ===================================================================

def evaluate(
    samples: List[Dict],
    indexer: MapleIndexer,
    maple_model: Optional[MapleNet],
) -> Dict:
    """
    Evaluate BM25, RAG, and MAPLE on a set of samples.

    Returns dict: {method: {k: [recall_values]}}
    """
    results = {
        method: {k: [] for k in K_VALUES}
        for method in ["BM25", "RAG", "MAPLE"]
    }

    for idx, sample in enumerate(samples):
        question = sample["question"]
        block_texts = sample["block_texts"]
        block_ids = sample["block_ids"]
        gt = sample["ground_truth_ids"]

        # -- BM25 --
        bm25 = BM25Retriever(block_texts)
        bm25_ranking = bm25.rank(question)

        # -- RAG (cosine similarity) --
        query_emb = indexer.encode_query(question)
        block_embs = indexer.model.encode(
            block_texts, convert_to_tensor=True, device=DEVICE, show_progress_bar=False,
        )
        sims = cosine_similarity(query_emb, block_embs)
        rag_ranking = torch.argsort(sims, descending=True).tolist()

        # -- MAPLE --
        maple_ranking = rag_ranking  # fallback if no model
        if maple_model is not None:
            with torch.no_grad():
                query_exp = query_emb.unsqueeze(0).expand(len(block_texts), -1)
                combined = torch.cat([query_exp, block_embs], dim=1)
                scores = torch.sigmoid(maple_model(combined))
            maple_ranking = torch.argsort(scores, descending=True).tolist()

        # -- Compute Recall@K --
        for k in K_VALUES:
            denom = min(len(gt), k)

            bm25_pred = set(str(block_ids[i]) for i in bm25_ranking[:k])
            rag_pred = set(str(block_ids[i]) for i in rag_ranking[:k])
            maple_pred = set(str(block_ids[i]) for i in maple_ranking[:k])

            results["BM25"][k].append(len(bm25_pred & gt) / denom)
            results["RAG"][k].append(len(rag_pred & gt) / denom)
            results["MAPLE"][k].append(len(maple_pred & gt) / denom)

        if (idx + 1) % 50 == 0:
            logger.info(f"  Evaluated {idx+1}/{len(samples)}")

    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_grouped_bar(all_results: Dict, output_path: Path):
    """
    Generate grouped bar chart: X = Dataset, Y = Recall@5, Groups = Methods.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = list(all_results.keys())
    methods = ["BM25", "RAG", "MAPLE"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.22

    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = []
        for ds in datasets:
            r = all_results[ds]
            recall_5 = np.mean(r[method][5]) * 100 if r[method][5] else 0
            vals.append(recall_5)

        bars = ax.bar(x + i * width, vals, width, label=method, color=color, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Dataset", fontsize=13)
    ax.set_ylabel("Recall@5 (%)", fontsize=13)
    ax.set_title("Retrieval Recall@5 Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grouped bar chart saved -> {output_path}")


# ===================================================================
# Main
# ===================================================================

def run(max_samples: int = 200):
    """Run the enhanced multi-dataset recall benchmark."""
    print("=" * 70)
    print("BENCHMARK 01 (ENHANCED): Multi-Dataset Recall")
    print(f"  Methods:  BM25, RAG, MAPLE")
    print(f"  Metrics:  Recall@{K_VALUES}")
    print(f"  Datasets: NarrativeQA, HotpotQA")
    print("=" * 70)

    # ---- Load components ----
    indexer = MapleIndexer(device=DEVICE)

    maple_model = None
    if MODEL_PATH.exists():
        maple_model = MapleNet.load(str(MODEL_PATH), device=DEVICE)
        logger.info(f"Loaded MAPLE model: {maple_model.num_parameters:,} params")
    else:
        logger.warning(f"No MAPLE model at {MODEL_PATH}, MAPLE will fallback to RAG")

    # ---- Load datasets ----
    all_results = {}

    # NarrativeQA
    nqa_samples = _load_narrativeqa_samples(max_samples)
    if nqa_samples:
        logger.info("Evaluating on NarrativeQA...")
        all_results["NarrativeQA"] = evaluate(nqa_samples, indexer, maple_model)

    # HotpotQA
    hqa_samples = _load_hotpotqa_samples(max_samples)
    if hqa_samples:
        logger.info("Evaluating on HotpotQA...")
        all_results["HotpotQA"] = evaluate(hqa_samples, indexer, maple_model)

    if not all_results:
        logger.error("No data to evaluate. Ensure oracle_data.json exists or HotpotQA is accessible.")
        return

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    output_data = {}
    for ds_name, res in all_results.items():
        print(f"\n--- {ds_name} ---")
        print(f"  {'K':<6} | {'BM25':<12} | {'RAG':<12} | {'MAPLE':<12}")
        print(f"  {'-'*48}")

        ds_output = {}
        for k in K_VALUES:
            bm25_r = np.mean(res["BM25"][k]) * 100
            rag_r = np.mean(res["RAG"][k]) * 100
            maple_r = np.mean(res["MAPLE"][k]) * 100
            print(f"  {k:<6} | {bm25_r:>8.1f}%   | {rag_r:>8.1f}%   | {maple_r:>8.1f}%")
            ds_output[f"recall@{k}"] = {
                "BM25": round(bm25_r, 2),
                "RAG": round(rag_r, 2),
                "MAPLE": round(maple_r, 2),
            }

        output_data[ds_name] = {
            "k_values": K_VALUES,
            "metrics": ds_output,
            "num_samples": len(nqa_samples if ds_name == "NarrativeQA" else hqa_samples),
        }

    # ---- Save ----
    json_path = RESULTS_DIR / "recall_enhanced.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ---- Plot ----
    plot_path = RESULTS_DIR / "recall_grouped_bar.png"
    plot_grouped_bar(all_results, plot_path)

    print("\n" + "=" * 70)
    print(f"Saved -> {json_path}")
    print(f"Chart -> {plot_path}")
    print("=" * 70)

    return output_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Recall Benchmark")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()
    run(max_samples=args.max_samples)
