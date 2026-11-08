#!/usr/bin/env python3
"""
Train a MAPLE model (canonical training script)
===============================================

Trains MapleNet on oracle attention labels from one or more domains:

  * NarrativeQA   (``data/oracle/narrativeqa_train.json``)
  * HotpotQA      (``data/oracle/hotpotqa_train.json``)
  * optional synthetic "needle" samples (``--needle-samples``)

For every oracle sample, *every* in-context block becomes one training row
(label 1 for the oracle top-5, 0 otherwise) with its real block id, so the
validation Recall@5 ranks the full candidate set of each question.

The synthetic needle data uses TRAIN_NEEDLE_TEMPLATES only; the needle
benchmark (benchmarks/03_needle_robust.py) uses disjoint held-out templates.

Usage:
    python scripts/train_generalist.py
    python scripts/train_generalist.py --datasets narrativeqa --needle-samples 0
    python scripts/train_generalist.py --embedding-model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleIndexer, MapleTrainer
from scripts.oracle_common import load_oracle_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ORACLE_DIR = Path("data/oracle")
DATASET_FILES = {
    "narrativeqa": ORACLE_DIR / "narrativeqa_train.json",
    "hotpotqa": ORACLE_DIR / "hotpotqa_train.json",
}
DEFAULT_OUTPUT = Path("models/maple_generalist.pth")
HAYSTACK_FILE = Path("data/gutenberg/84.txt")  # Frankenstein (benchmark uses Alice, #11)

# Training-only needle templates: (needle sentence, question). The needle
# benchmark uses a disjoint set (see benchmarks/03_needle_robust.py).
TRAIN_NEEDLE_TEMPLATES = [
    ("The secret passkey is {v}.", "What is the secret passkey?"),
    ("The vault combination is {v}.", "What is the vault combination?"),
    ("Remember: the access token for the server is {v}.", "What is the access token for the server?"),
    ("The shipment tracking number is {v}.", "What is the shipment tracking number?"),
    ("Note that the laboratory door code is {v}.", "What is the laboratory door code?"),
]


def build_oracle_rows(samples: List[dict], indexer: MapleIndexer) -> List[Dict]:
    """One row per (question, in-context block), with real block ids."""
    rows: List[Dict] = []
    for n, s in enumerate(samples):
        blocks = s["blocks"]
        if not blocks:
            continue
        positives = set(s["top_5_block_ids"])
        q_emb = indexer.encode_query(s["question"]).cpu()
        b_embs = indexer.encode_texts([b["text"] for b in blocks]).cpu()
        group = f"{s['domain']}::{s['id']}"
        for b, emb in zip(blocks, b_embs):
            rows.append({
                "group": group,
                "question": s["question"],
                "query_emb": q_emb,
                "block_emb": emb,
                "label": 1.0 if b["id"] in positives else 0.0,
                "block_id": int(b["id"]),
                "domain": s["domain"],
            })
        if (n + 1) % 100 == 0:
            logger.info(f"  encoded {n + 1}/{len(samples)} samples")
    return rows


def build_needle_rows(indexer: MapleIndexer, num_samples: int, seed: int = 0,
                      negatives_per_sample: int = 9) -> List[Dict]:
    """Synthetic needle rows: one positive + hard/easy negatives per sample."""
    if num_samples <= 0:
        return []
    from benchmarks.data_loader import get_gutenberg_books

    rng = random.Random(seed)
    text = get_gutenberg_books([84]).get(84) or ("It was a dark and stormy night. " * 4000)
    blocks = indexer.chunk_text(text, chunk_size=500)[:2000]
    block_embs = indexer.encode_blocks(blocks, show_progress=False).cpu()

    rows: List[Dict] = []
    for i in range(num_samples):
        template, question = rng.choice(TRAIN_NEEDLE_TEMPLATES)
        value = str(uuid.UUID(int=rng.getrandbits(128)))
        needle = template.format(v=value)
        target = rng.randrange(len(blocks))
        base = blocks[target].text
        style = rng.choice(["append", "prepend", "middle", "alone"])
        if style == "append":
            injected = f"{base} {needle}"
        elif style == "prepend":
            injected = f"{needle} {base}"
        elif style == "middle":
            cut = len(base) // 2
            injected = f"{base[:cut]} {needle} {base[cut:]}"
        else:
            injected = needle

        wrong = f"{base} " + rng.choice(TRAIN_NEEDLE_TEMPLATES)[0].format(v=str(uuid.uuid4()))
        q_emb = indexer.encode_query(question).cpu()
        pos_emb, wrong_emb = indexer.encode_texts([injected, wrong]).cpu()

        group = f"needle::{i}"
        cand = [(pos_emb, 1.0, 0), (wrong_emb, 0.0, 1), (block_embs[target], 0.0, 2)]
        neg_ids = rng.sample(range(len(blocks)), k=min(negatives_per_sample - 2, len(blocks)))
        cand += [(block_embs[j], 0.0, 3 + n) for n, j in enumerate(neg_ids) if j != target]
        for emb, label, bid in cand:
            rows.append({"group": group, "question": question, "query_emb": q_emb,
                         "block_emb": emb, "label": label, "block_id": bid,
                         "domain": "synthetic_needle"})
    logger.info(f"Built {len(rows)} needle rows from {num_samples} samples")
    return rows


def run(
    datasets: Sequence[str] = ("narrativeqa", "hotpotqa"),
    needle_samples: int = 1000,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    output_model_path: Path = DEFAULT_OUTPUT,
    device: Optional[str] = None,
    seed: int = 42,
    max_samples_per_dataset: Optional[int] = None,
) -> dict:
    """Train and save a MAPLE model. Returns a summary dict."""
    random.seed(seed)
    torch.manual_seed(seed)
    indexer = MapleIndexer(model_name=embedding_model, device=device)

    rows: List[Dict] = []
    used = {}
    for name in datasets:
        path = DATASET_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Generate it first (see docs/DATA.md).")
        samples = load_oracle_samples(path)
        if max_samples_per_dataset:
            samples = samples[:max_samples_per_dataset]
        logger.info(f"{name}: {len(samples)} oracle samples")
        rows.extend(build_oracle_rows(samples, indexer))
        used[name] = len(samples)

    rows.extend(build_needle_rows(indexer, needle_samples, seed=seed))
    used["synthetic_needle"] = needle_samples
    pos = sum(r["label"] for r in rows)
    logger.info(f"Total rows: {len(rows)} ({int(pos)} positive)")

    trainer = MapleTrainer(
        input_dim=indexer.get_embedding_dimension() * 2,
        hidden_dim=128,
        device=device,
        seed=seed,
    )
    model, best_recall = trainer.train(
        training_data=rows, epochs=epochs, batch_size=batch_size, lr=lr,
        val_split=0.2, save_path=str(output_model_path),
    )

    summary = {
        "output_model": str(output_model_path),
        "embedding_model": embedding_model,
        "datasets": used,
        "rows": len(rows),
        "positives": int(pos),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "best_val_recall@5": best_recall,
        "history": trainer.history_,
    }
    summary_path = Path(output_model_path).with_suffix(".train.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary -> {summary_path}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Train MAPLE (canonical)")
    p.add_argument("--datasets", nargs="+", default=["narrativeqa", "hotpotqa"], choices=list(DATASET_FILES))
    p.add_argument("--needle-samples", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples-per-dataset", type=int, default=None)
    a = p.parse_args()

    from benchmarks.profiler import HardwareMonitor, wrap_result
    with HardwareMonitor(interval=0.5) as mon:
        summary = run(a.datasets, a.needle_samples, a.epochs, a.batch_size, a.lr,
                      a.embedding_model, a.output, a.device, a.seed, a.max_samples_per_dataset)
    out = Path(a.output).with_suffix(".hardware.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(wrap_result({"best_val_recall@5": summary["best_val_recall@5"]}, mon), f, indent=2)
    print(f"Best val Recall@5: {summary['best_val_recall@5'] * 100:.1f}%  ->  {a.output}")


if __name__ == "__main__":
    main()
