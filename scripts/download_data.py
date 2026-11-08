#!/usr/bin/env python3
"""
Download every public dataset / model MAPLE experiments need
============================================================

Datasets (Hugging Face, cached under HF_DATASETS_CACHE):
    * deepmind/narrativeqa        train / validation / test  (~10 GB, full stories)
    * hotpotqa/hotpot_qa          distractor: train / validation
    * rajpurkar/squad             validation
Texts (Project Gutenberg, saved to data/gutenberg/):
    * 1661 Sherlock Holmes, 1342 Pride & Prejudice, 84 Frankenstein, 11 Alice
Encoders (sentence-transformers):
    * BAAI/bge-small-en-v1.5, sentence-transformers/all-MiniLM-L6-v2,
      nomic-ai/nomic-embed-text-v1.5
Teacher (gated — needs `hf auth login` + accepted licence):
    * meta-llama/Meta-Llama-3-8B-Instruct   (~16 GB)

Writes ``data/MANIFEST.json`` with sizes / row counts so runs are traceable.

Usage:
    python scripts/download_data.py                 # everything public
    python scripts/download_data.py --teacher       # also the gated teacher
    python scripts/download_data.py --only hotpotqa squad
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MANIFEST = Path("data/MANIFEST.json")

HF_DATASETS = {
    "narrativeqa": ("deepmind/narrativeqa", None, ["train", "validation", "test"]),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", ["train", "validation"]),
    "squad": ("rajpurkar/squad", None, ["validation"]),
}
GUTENBERG_IDS = [1661, 1342, 84, 11]
ENCODERS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "nomic-ai/nomic-embed-text-v1.5",
]
TEACHER = "meta-llama/Meta-Llama-3-8B-Instruct"


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"datasets": {}, "gutenberg": {}, "encoders": {}, "teacher": {}}


def _save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    m["updated"] = datetime.now().isoformat()
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")


def download_dataset(name: str, manifest: dict) -> None:
    from datasets import load_dataset

    repo, config, splits = HF_DATASETS[name]
    entry = {"repo": repo, "config": config, "splits": {}}
    for split in splits:
        t = time.time()
        logger.info(f"[{name}] downloading {repo} {config or ''} split={split} ...")
        ds = load_dataset(repo, config, split=split)
        entry["splits"][split] = {"rows": len(ds), "fingerprint": ds._fingerprint}
        logger.info(f"[{name}] {split}: {len(ds):,} rows ({time.time() - t:.0f}s)")
    manifest["datasets"][name] = entry
    _save_manifest(manifest)


def download_gutenberg(manifest: dict) -> None:
    from benchmarks.data_loader import get_gutenberg_books

    books = get_gutenberg_books(GUTENBERG_IDS)
    manifest["gutenberg"] = {str(k): len(v) for k, v in books.items()}
    missing = set(GUTENBERG_IDS) - set(books)
    if missing:
        logger.warning(f"Gutenberg books not downloaded: {sorted(missing)}")
    _save_manifest(manifest)


def download_encoders(manifest: dict) -> None:
    from sentence_transformers import SentenceTransformer

    for name in ENCODERS:
        logger.info(f"[encoder] {name}")
        m = SentenceTransformer(name, device="cpu", trust_remote_code=True)
        manifest["encoders"][name] = {"dim": m.get_sentence_embedding_dimension()}
        del m
    _save_manifest(manifest)


def download_teacher(manifest: dict) -> None:
    from huggingface_hub import snapshot_download

    logger.info(f"[teacher] {TEACHER} (gated)")
    path = snapshot_download(TEACHER, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    shards = list(Path(path).glob("*.safetensors"))
    if not shards:
        # Seen on Windows: snapshot_download returned after fetching only
        # config.json. Fail loudly instead of recording a broken snapshot.
        raise RuntimeError(
            f"No .safetensors shards in {path}. Retry with: hf download {TEACHER} "
            '--include "*.json" --include "*.safetensors"'
        )
    size_gb = sum(p.stat().st_size for p in shards) / 1e9
    manifest["teacher"] = {"repo": TEACHER, "path": path, "shards": len(shards), "size_gb": round(size_gb, 2)}
    _save_manifest(manifest)


def main():
    p = argparse.ArgumentParser(description="Download MAPLE data")
    p.add_argument("--only", nargs="+", default=None,
                   choices=list(HF_DATASETS) + ["gutenberg", "encoders", "teacher"])
    p.add_argument("--teacher", action="store_true", help="Also download the gated teacher LLM")
    a = p.parse_args()

    targets = a.only or (["gutenberg", "encoders", "squad", "hotpotqa", "narrativeqa"]
                         + (["teacher"] if a.teacher else []))
    manifest = _load_manifest()
    failures = {}
    for t in targets:
        try:
            if t in HF_DATASETS:
                download_dataset(t, manifest)
            elif t == "gutenberg":
                download_gutenberg(manifest)
            elif t == "encoders":
                download_encoders(manifest)
            elif t == "teacher":
                download_teacher(manifest)
        except Exception as e:  # keep going; report at the end
            logger.error(f"{t} failed: {e}")
            failures[t] = str(e)

    print("\n" + "=" * 60)
    print(f"Manifest: {MANIFEST}")
    if failures:
        print("FAILED:")
        for k, v in failures.items():
            print(f"  {k}: {v[:200]}")
        sys.exit(1)
    print("All requested downloads complete.")


if __name__ == "__main__":
    from scripts.runlog import record_outputs, start_run
    with start_run("download", "data", {"argv": sys.argv[1:]}) as _run:
        main()
        record_outputs(_run, MANIFEST, role='manifest')
