#!/usr/bin/env python3
"""
Embedding cache for oracle files
================================
Encoding every question and block of an oracle file with the sentence encoder
takes minutes; training variants (targets, models, seeds) all need the same
vectors. Encode once per (oracle file, encoder) and reload in well under a
second.

Cache file (torch, tensors / lists / strings only, so it loads with
``weights_only=True``):

    {"encoder": str, "oracle": str, "ids": [sample id],
     "q": float16 [N, d], "blocks": [float16 [n_i, d]], "block_ids": [[block id]]}

Usage:
    python scripts/embedding_cache.py data/oracle/*.json data/oracle/t4096/*.json
    python scripts/embedding_cache.py data/oracle/hotpotqa_train.json --encoder sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.oracle_common import load_oracle_samples

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"
CACHE_DIR = Path("data/cache")


def cache_path_for(oracle_path: Path, encoder: str) -> Path:
    """data/cache/<oracle stem>__<encoder>.pt ; oracle files in sub-folders of data/oracle
    (e.g. t4096/) get the folder name as a prefix so budgets never share a cache."""
    oracle_path = Path(oracle_path)
    folder = oracle_path.parent.name
    prefix = "" if folder in ("oracle", "") else f"{folder}__"
    return CACHE_DIR / f"{prefix}{oracle_path.stem}__{encoder.replace('/', '--')}.pt"


def build_cache(oracle_path: Path, out_path: Path, indexer=None, encoder: str = DEFAULT_ENCODER,
                device: Optional[str] = None) -> Path:
    """Encode every sample's question and blocks and save them to ``out_path``."""
    oracle_path, out_path = Path(oracle_path), Path(out_path)
    if indexer is None:
        from maplecore import MapleIndexer
        indexer = MapleIndexer(model_name=encoder, device=device)
    samples = load_oracle_samples(oracle_path)

    ids, q, blocks, block_ids = [], [], [], []
    for n, s in enumerate(samples):
        ids.append(str(s["id"]))
        q.append(indexer.encode_query(s["question"]).detach().float().cpu())
        texts = [b["text"] for b in s["blocks"]]
        blocks.append(indexer.encode_texts(texts).detach().float().cpu().half())
        block_ids.append([int(b["id"]) for b in s["blocks"]])
        if (n + 1) % 250 == 0:
            logger.info(f"  {oracle_path.name}: encoded {n + 1}/{len(samples)}")

    payload = {
        "encoder": indexer.model_name,
        "oracle": str(oracle_path).replace("\\", "/"),
        "ids": ids,
        "q": torch.stack(q).half(),
        "blocks": blocks,
        "block_ids": block_ids,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    logger.info(f"cache -> {out_path} ({len(ids)} samples)")
    return out_path


def load_cache(path: Path, expected_encoder: Optional[str] = None) -> dict:
    """Load a cache; refuse one built with a different encoder than ``expected_encoder``."""
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    if expected_encoder is not None and cache["encoder"] != expected_encoder:
        raise ValueError(
            f"cache {path} was built with encoder '{cache['encoder']}', not '{expected_encoder}'; "
            f"rebuild it with --encoder {expected_encoder}"
        )
    return cache


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Build embedding caches for oracle files")
    p.add_argument("oracles", nargs="+", type=Path)
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="Rebuild caches that already exist")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    from maplecore import MapleIndexer
    indexer = MapleIndexer(model_name=a.encoder, device=a.device)
    for oracle in a.oracles:
        out = cache_path_for(oracle, a.encoder)
        if out.exists() and not a.force:
            print(f"exists: {out}")
            continue
        build_cache(oracle, out, indexer=indexer)
        print(f"built:  {out}")


if __name__ == "__main__":
    from scripts.runlog import start_run

    with start_run("train", "embedding_cache", {"argv": sys.argv[1:]}):
        main()
