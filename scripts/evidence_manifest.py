#!/usr/bin/env python3
"""
Evidence manifest: SHA-256, size and modification time of every result, model and data file
==========================================================================================
A number quoted in a report can be traced to the exact file it came from, and a later change to that file is
detectable. Default roots: benchmarks/results, models, data/oracle, runs/index.jsonl.

Usage:
    python scripts/evidence_manifest.py                       # -> docs/private/evidence_manifest.json
    python scripts/evidence_manifest.py --roots benchmarks/results --out manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

DEFAULT_ROOTS = ("benchmarks/results", "models", "data/oracle", "runs/index.jsonl")
DEFAULT_OUT = Path("docs/private/evidence_manifest.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(roots: Sequence[Path], base: Path = Path(".")) -> List[dict]:
    """One row per file under ``roots`` (files or folders; missing roots are skipped), sorted by path."""
    base = Path(base).resolve()
    files = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            files.add(root.resolve())
        elif root.is_dir():
            files.update(p.resolve() for p in root.rglob("*") if p.is_file())
    rows = []
    for p in sorted(files, key=lambda x: x.relative_to(base).as_posix()):
        st = p.stat()
        rows.append({"path": p.relative_to(base).as_posix(), "bytes": st.st_size, "sha256": _sha256(p),
                     "modified": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
    return rows


def _git_head(base: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=base, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def write_manifest(roots: Sequence[Path], out: Path, base: Path = Path(".")) -> dict:
    rows = build_manifest(roots, base)
    data = {"created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "git_head": _git_head(base),
            "roots": [str(r) for r in roots], "n_files": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
            "files": rows}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def main(argv: Optional[List[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description="SHA-256 manifest of results, models and data")
    ap.add_argument("--roots", nargs="+", type=Path, default=[Path(r) for r in DEFAULT_ROOTS])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    data = write_manifest(a.roots, a.out)
    print(f"wrote {a.out}: {data['n_files']} files, {data['total_bytes'] / 1e9:.2f} GB, git {data['git_head']}")
    return data


if __name__ == "__main__":
    main()
