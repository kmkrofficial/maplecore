#!/usr/bin/env python3
"""
Side-by-side comparison with published numbers
==============================================
A number of ours is placed next to a paper's only when both measure the same thing: the *protocol* (dataset split,
candidate pool, metric) must match and our reproduction of that protocol must be sane (``protocol_ok``, e.g. our BM25
within +/-2 nDCG of the published BM25; the benchmarks compute that flag). Otherwise the row says
``protocol mismatch`` with no delta; protocols we have not run say ``not run``.

Protocol ids (published rows live in docs/private/published_numbers.json, transcribed from the papers):
    icr_multihop              ICR (ICLR 2025) Tables 2-3, bench 09
    beir_bm25top100           ICR Table 1, bench 08 (first_stage_k = 100)
    beir_bm25top200           QRHead (EMNLP 2025) Table 2, BM25 top-200 (never comparable to bench 08 at k=100)
    longbench_v1_llama3_8b    InfLLM (NeurIPS 2024) Table 5, bench 10
    narrativeqa_fullstory_e2e Izacard & Grave 2021 Table 4 / RAPTOR 2024 Table 1 - different readers: context only

Usage:
    python benchmarks/compare_published.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import RESULTS_DIR

PUBLISHED_FILE = Path("docs/private/published_numbers.json")
OUT_FILE = Path("docs/private/results/published_comparison.md")


@dataclass
class PublishedResult:
    paper: str
    venue: str
    protocol: str
    dataset: str
    metric: str
    method: str
    value: float
    note: str = ""


@dataclass
class ComparisonRow:
    protocol: str
    dataset: str
    metric: str
    ours_method: Optional[str]
    ours_value: Optional[float]
    paper: str
    paper_method: str
    paper_value: float
    delta: Optional[float]
    status: str  # comparable | protocol mismatch | not run


def load_published(path: Path) -> List[PublishedResult]:
    return [PublishedResult(**r) for r in json.loads(Path(path).read_text(encoding="utf-8"))]


def _latest(results_dir: Path, name: str) -> Optional[dict]:
    files = sorted(Path(results_dir).rglob(name), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def collect_ours(results_dir: Path) -> Dict[str, dict]:
    """{protocol: {"ok": bool, "rows": {(dataset, metric, method): value x100}}} from the benchmark result files."""
    out: Dict[str, dict] = {}

    mh = _latest(results_dir, "multihop_rerank.json")
    if mh:
        rows = {}
        for ds, res in mh["datasets"].items():
            for method, metrics in res.items():
                if method.startswith("_"):
                    continue
                for metric, summ in metrics.items():
                    rows[(ds, metric, method)] = 100 * summ["mean"]
        out["icr_multihop"] = {"ok": bool(mh.get("protocol_ok")), "rows": rows}

    beir = _latest(results_dir, "beir_rerank.json")
    if beir:
        rows = {}
        for ds, res in beir["datasets"].items():
            rows[(ds, "ndcg@10", "bm25")] = 100 * res["_first_stage"]["bm25"]["ndcg@10"]["mean"]
            for method, metrics in res.items():
                if not method.startswith("_"):
                    rows[(ds, "ndcg@10", method)] = 100 * metrics["ndcg@10"]["mean"]
        ok = not beir.get("protocol_mismatch")
        out["beir_bm25top100"] = {"ok": ok and beir["config"]["first_stage_k"] == 100, "rows": rows}
        out["beir_bm25top200"] = {"ok": ok and beir["config"]["first_stage_k"] == 200, "rows": rows}

    e2e = _latest(results_dir, "e2e_reader_fixed.json")
    if e2e:
        lb, nqa = {}, {}
        for task, methods in e2e["tasks"].items():
            for method, res in methods.items():
                m = res["metrics"]
                if task.startswith("longbench_"):
                    lb[(task.replace("longbench_", ""), "f1", method)] = 100 * m["f1"]["mean"]
                else:
                    for k, v in m.items():
                        if v is not None:
                            nqa[("narrativeqa", k, method)] = float(v)
        if lb:
            out["longbench_v1_llama3_8b"] = {"ok": bool(e2e.get("protocol_ok")), "rows": lb}
        if nqa:
            out["narrativeqa_fullstory_e2e"] = {"ok": False, "rows": nqa}  # different reader: context, not a comparison
    return out


def join(published: Sequence[PublishedResult], ours: Dict[str, dict]) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for p in published:
        run = ours.get(p.protocol)
        cells = [(k[2], v) for k, v in run["rows"].items() if k[0] == p.dataset and k[1] == p.metric] if run else []
        if not cells:
            rows.append(ComparisonRow(p.protocol, p.dataset, p.metric, None, None, p.paper, p.method, p.value, None,
                                      "not run"))
            continue
        for method, value in cells:
            ok = run["ok"]
            rows.append(ComparisonRow(p.protocol, p.dataset, p.metric, method, value, p.paper, p.method, p.value,
                                      value - p.value if ok else None, "comparable" if ok else "protocol mismatch"))
    return rows


def _fmt_delta(d: Optional[float]) -> str:
    return "n/a" if d is None else f"{d:+.1f}".replace("-", "−")


def render_markdown(rows: Sequence[ComparisonRow]) -> str:
    lines = ["# Ours vs published results", "",
             "`comparable` = same protocol and our reproduction check passed; `protocol mismatch` = numbers shown "
             "for context only (no delta); `not run` = we have no result for that cell.", ""]
    for protocol in sorted({r.protocol for r in rows}):
        lines += [f"## {protocol}", "", "| dataset | metric | paper (method) | paper | ours (method) | ours | Δ | status |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in (x for x in rows if x.protocol == protocol):
            ours = "n/a" if r.ours_value is None else f"{r.ours_value:.1f}"
            lines.append(f"| {r.dataset} | {r.metric} | {r.paper} ({r.paper_method}) | {r.paper_value:.1f} | "
                         f"{r.ours_method or 'n/a'} | {ours} | {_fmt_delta(r.delta)} | {r.status} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> str:
    ap = argparse.ArgumentParser(description="Compare our results with published numbers")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--published", type=Path, default=PUBLISHED_FILE)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    a = ap.parse_args(argv)
    md = render_markdown(join(load_published(a.published), collect_ours(a.results)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    print(f"wrote {a.out} ({md.count(chr(10))} lines)")
    return md


if __name__ == "__main__":
    from scripts.runlog import record_outputs, start_run

    with start_run("compare", "published", {"argv": sys.argv[1:]}) as _run:
        main()
        record_outputs(_run, OUT_FILE)
