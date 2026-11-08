#!/usr/bin/env python3
"""
Run registry: every training / label / benchmark / comparison run leaves a
self-contained directory so results can be referenced and reused later.

    runs/<YYYYMMDD-HHMMSS>_<kind>_<tag>/
        config.json      arguments + argv + git state + environment + data manifest
        run.log          everything logged / printed during the run
        metrics.jsonl    one line per Run.log_metrics call
        items/*.jsonl    per-question outputs (rankings, scores, predictions)
        artifacts.json   path, role, bytes, sha256 of every recorded output
        artifacts/       copies of small (< 50 MB) artifacts
        hardware.json    CPU / RAM / VRAM usage
        summary.json     status (completed | failed), traceback, duration, final metrics
    runs/index.jsonl     one line per finished run (append-only)

Usage:
    with start_run("bench", "retrieval", vars(args)) as run:
        run.log_metrics({"recall@5": 0.8})
        run.save_items("per_question", rows)
        run.add_artifact(Path("benchmarks/results/x.json"), role="result")

CLI:
    python scripts/runlog.py list [--kind K]
    python scripts/runlog.py show RUN_ID
    python scripts/runlog.py import-legacy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_ROOT = Path("runs")
COPY_LIMIT_BYTES = 50 * 1024 * 1024
KINDS = {"oracle", "train", "bench", "compare", "download", "aggregate", "legacy"}
_ACTIVE: List["Run"] = []


def current_run() -> Optional["Run"]:
    """The innermost active Run, or None outside a start_run context."""
    return _ACTIVE[-1] if _ACTIVE else None


# ---------------------------------------------------------------- helpers

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def git_state(cwd: Optional[Path] = None) -> dict:
    diff = _git("diff", "HEAD", cwd=cwd)
    return {
        "commit": _git("rev-parse", "HEAD", cwd=cwd) or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd) or None,
        "dirty": bool(_git("status", "--porcelain", cwd=cwd)),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }


def environment() -> dict:
    env = {"platform": "wsl" if "microsoft" in platform.release().lower() else platform.system().lower(),
           "os": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        env.update(torch=None, cuda=None, gpu=None)
    try:  # importlib.metadata works in venvs without pip (uv)
        from importlib import metadata
        env["packages"] = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                                  if d.metadata["Name"]}, key=str.lower)
    except Exception:
        env["packages"] = []
    return env


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------- the tee

class _Tee:
    """File-like that writes to the original stream and to the run log."""

    def __init__(self, original, sink):
        self._original, self._sink = original, sink

    def write(self, s):
        try:
            self._original.write(s)
        except Exception:
            pass
        try:
            self._sink.write(s)
            self._sink.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        for f in (self._original, self._sink):
            try:
                f.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._original, name)


# -------------------------------------------------------------------- Run

class Run:
    def __init__(self, kind: str, tag: str, config: dict, root: Path, wandb: bool = False):
        self.kind, self.tag, self.root = kind, tag, Path(root)
        self.config = dict(config)
        base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{kind}_{tag}"
        self.id, n = base, 1
        while (self.root / self.id).exists():  # two runs in the same second must not share a directory
            n += 1
            self.id = f"{base}-{n}"
        self.dir = self.root / self.id
        self.final_metrics: Dict[str, object] = {}
        self._artifacts: List[dict] = []
        self._monitor = None
        self._t0 = 0.0
        self._wandb_run = None
        self._want_wandb = wandb
        self._saved = {}

    # -- context manager
    def __enter__(self) -> "Run":
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        manifest = Path("data/MANIFEST.json")
        cfg = {**self.config, "run_id": self.id, "kind": self.kind, "tag": self.tag, "argv": sys.argv,
               "git": git_state(), "env": environment(),
               "data_manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None}
        _write_json(self.dir / "config.json", cfg)
        self._cfg_written = cfg

        self._log_file = open(self.dir / "run.log", "a", encoding="utf-8", buffering=1)
        self._saved = {"stdout": sys.stdout, "stderr": sys.stderr}
        sys.stdout = _Tee(sys.stdout, self._log_file)
        sys.stderr = _Tee(sys.stderr, self._log_file)
        self._handler = logging.StreamHandler(self._log_file)
        self._handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        self._prev_level = root_logger.level
        if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
            root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self._handler)

        try:
            from benchmarks.profiler import HardwareMonitor
            self._monitor = HardwareMonitor(interval=1.0)
            self._monitor.start()
        except Exception:
            self._monitor = None

        if self._want_wandb:
            self._start_wandb()
        _ACTIVE.append(self)
        print(f"[runlog] run {self.id} -> {self.dir}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status, tb_text = "completed", None
        if exc_type is not None:
            status = "failed"
            tb_text = "".join(traceback.format_exception(exc_type, exc, tb))
            if exc_type is KeyboardInterrupt and not tb_text.startswith("KeyboardInterrupt"):
                tb_text = "KeyboardInterrupt\n" + tb_text
        try:
            hardware = {}
            if self._monitor is not None:
                self._monitor.stop()
                hardware = self._monitor.get_stats()
            _write_json(self.dir / "hardware.json", hardware)
            _write_json(self.dir / "artifacts.json", self._artifacts)
            summary = {"run_id": self.id, "kind": self.kind, "tag": self.tag, "status": status,
                       "traceback": tb_text, "duration_s": round(time.time() - self._t0, 2),
                       "metrics": self.final_metrics, "finished": datetime.now().isoformat()}
            _write_json(self.dir / "summary.json", summary)
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.root / "index.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": self.id, "kind": self.kind, "tag": self.tag, "status": status,
                                    "duration_s": summary["duration_s"], "dir": str(self.dir),
                                    "git_commit": self._cfg_written["git"]["commit"],
                                    "finished": summary["finished"]}) + "\n")
            if self._wandb_run is not None:
                try:
                    self._wandb_run.summary.update({"status": status, **{k: v for k, v in self.final_metrics.items()
                                                                          if isinstance(v, (int, float))}})
                    self._wandb_run.finish()
                except Exception:
                    pass
        finally:
            if self in _ACTIVE:
                _ACTIVE.remove(self)
            sys.stdout, sys.stderr = self._saved["stdout"], self._saved["stderr"]
            logging.getLogger().removeHandler(self._handler)
            logging.getLogger().setLevel(self._prev_level)
            self._handler.close()
            self._log_file.close()
        return False  # never swallow the exception

    # -- API
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        row = {"step": step, "time": datetime.now().isoformat(), **metrics}
        with open(self.dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self.final_metrics.update({k: v for k, v in metrics.items()})
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception:
                pass

    def save_items(self, name: str, rows: Iterable[dict]) -> Path:
        path = self.dir / "items" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        return path

    def add_artifact(self, path: Path, role: str) -> None:
        path = Path(path)
        size = path.stat().st_size
        entry = {"path": str(path).replace("\\", "/"), "role": role, "bytes": size, "sha256": _sha256(path)}
        if size < COPY_LIMIT_BYTES:
            dest = self.dir / "artifacts" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            entry["copied"] = True
        self._artifacts.append(entry)
        _write_json(self.dir / "artifacts.json", self._artifacts)

    def _start_wandb(self) -> None:
        import os
        if not os.environ.get("WANDB_API_KEY"):
            return
        try:
            import wandb
            self._wandb_run = wandb.init(project="maplecore", name=self.id, config=self.config, reinit=True)
        except Exception as e:  # mirroring is optional; the local dir is the source of truth
            print(f"[runlog] wandb disabled: {e}")


def start_run(kind: str, tag: str, config: dict, root: Path = DEFAULT_ROOT, wandb: bool = False) -> Run:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    return Run(kind, tag, config, Path(root), wandb)


def flatten_metrics(tree, leaf: str = "mean", _prefix: str = "") -> Dict[str, float]:
    """Flatten a benchmark result tree to {"a/b/c": value}, keeping only `leaf` entries
    (the `mean` of each {mean, ci_low, ci_high, n} summary) so metrics.jsonl stays readable."""
    out: Dict[str, float] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == leaf and isinstance(v, (int, float)) and not isinstance(v, bool):
                out[_prefix.rstrip("/")] = float(v)
            elif isinstance(v, dict):
                out.update(flatten_metrics(v, leaf, f"{_prefix}{k}/"))
    return out


def record_outputs(run: Run, *paths, role: str = "result") -> None:
    """Add every existing path as an artifact of the run (missing paths are skipped)."""
    for p in paths:
        p = Path(p)
        if p.is_file():
            run.add_artifact(p, role)


# ------------------------------------------------------------- model card

def write_model_card(model_path: Path, run: Run, metrics: dict, extra: Optional[dict] = None) -> Path:
    """Write <model>.card.json linking a checkpoint to the run that produced it."""
    model_path = Path(model_path)
    data_files = []
    for p in run.config.get("data_files", []) or []:
        p = Path(p)
        if p.exists():
            data_files.append({"path": str(p).replace("\\", "/"), "sha256": _sha256(p)})
    card = {
        "run_id": run.id,
        "model": str(model_path).replace("\\", "/"),
        "sha256": _sha256(model_path),
        "bytes": model_path.stat().st_size,
        "config": run.config,
        "encoder": run.config.get("embedding_model") or run.config.get("encoder"),
        "seed": run.config.get("seed"),
        "data_files": data_files,
        "metrics": metrics,
        "git": git_state(),
        "created": datetime.now().isoformat(),
        **(extra or {}),
    }
    out = model_path.with_suffix(model_path.suffix + ".card.json")
    _write_json(out, card)
    return out


# ------------------------------------------------------------ legacy import

def import_legacy(root: Path = DEFAULT_ROOT, repo: Path = Path(".")) -> Path:
    """Copy the pre-registry (v1) logs/results into runs/legacy_v1 with checksums."""
    repo, root = Path(repo), Path(root)
    out = root / "legacy_v1"
    out.mkdir(parents=True, exist_ok=True)

    def copy_tree(rel: str, pattern: str = "*") -> None:
        for p in (repo / rel).rglob(pattern) if (repo / rel).exists() else []:
            if p.is_file():
                dest = out / p.relative_to(repo)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)

    copy_tree("logs", "*.log")
    copy_tree("benchmarks/results")
    for pat in ("*.train.json", "*.hardware.json", "*.card.json"):
        for p in (repo / "models").glob(pat) if (repo / "models").exists() else []:
            dest = out / "models" / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    manifest = repo / "data" / "MANIFEST.json"
    if manifest.exists():
        (out / "data").mkdir(exist_ok=True)
        shutil.copy2(manifest, out / "data" / "MANIFEST.json")

    arts = []
    for p in sorted((repo / "models").glob("*.pth")) if (repo / "models").exists() else []:
        arts.append({"path": p.relative_to(repo).as_posix(), "role": "checkpoint", "bytes": p.stat().st_size,
                     "sha256": _sha256(p)})
    oracle = repo / "data" / "oracle"
    for p in sorted(oracle.rglob("*.json")) if oracle.exists() else []:
        arts.append({"path": p.relative_to(repo).as_posix(), "role": "oracle", "bytes": p.stat().st_size,
                     "sha256": _sha256(p)})
    _write_json(out / "artifacts.json", arts)

    root.mkdir(parents=True, exist_ok=True)
    with open(root / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "legacy_v1", "kind": "legacy", "tag": "v1", "status": "completed",
                            "dir": str(out), "finished": datetime.now().isoformat(),
                            "git_commit": git_state(repo)["commit"]}) + "\n")
    return out


# --------------------------------------------------------------------- CLI

def _read_index(root: Path) -> List[dict]:
    p = Path(root) / "index.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run registry")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--kind", default=None)
    sh = sub.add_parser("show")
    sh.add_argument("run_id")
    sub.add_parser("import-legacy")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        rows = [r for r in _read_index(a.root) if not a.kind or r["kind"] == a.kind]
        print(f"{'id':<48} {'kind':<9} {'status':<10} {'duration_s':>10}")
        for r in rows:
            print(f"{r['id']:<48} {r['kind']:<9} {r['status']:<10} {r.get('duration_s', ''):>10}")
        print(f"{len(rows)} run(s)")
    elif a.cmd == "show":
        d = a.root / a.run_id
        for name in ("summary.json", "config.json", "artifacts.json"):
            p = d / name
            if p.exists():
                print(f"--- {name}\n{p.read_text(encoding='utf-8')[:4000]}")
    elif a.cmd == "import-legacy":
        out = import_legacy(a.root)
        arts = json.loads((out / "artifacts.json").read_text(encoding="utf-8"))
        n_files = sum(1 for p in out.rglob("*") if p.is_file())
        print(f"legacy import -> {out}: {n_files} files, {len(arts)} checksummed artifacts")


if __name__ == "__main__":
    main()
