#!/usr/bin/env python3
"""
MAPLE single-domain trainer (compatibility wrapper)
===================================================

Kept for ``benchmarks/run_matrix.py`` and older notes. Trains on NarrativeQA
oracle labels only, without synthetic needle data, by delegating to the
canonical ``scripts/train_generalist.py``.

Usage:
    python scripts/train_maple.py --epochs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_generalist import run as _run_generalist

MODEL_PATH = Path("models/maple_v1.pth")


def run(
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,  # kept for signature compatibility (fixed at 0.2)
    device: Optional[str] = None,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    output_model_path: Optional[Path] = None,
    datasets=("narrativeqa",),
    needle_samples: int = 0,
):
    return _run_generalist(
        datasets=datasets,
        needle_samples=needle_samples,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        embedding_model=embedding_model,
        output_model_path=Path(output_model_path or MODEL_PATH),
        device=device,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MAPLE single-domain trainer")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--output", type=Path, default=MODEL_PATH)
    a = p.parse_args()
    s = run(a.epochs, a.batch_size, a.lr, embedding_model=a.embedding_model, output_model_path=a.output)
    print(f"Best val Recall@5: {s['best_val_recall@5'] * 100:.1f}%  ->  {s['output_model']}")
