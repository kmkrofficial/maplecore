#!/usr/bin/env python3
"""
Aggregate benchmark outputs from scripts/run_experiment.sh into one summary.

Reads benchmarks/results/b<budget>_s<seed>/*.json and benchmarks/results/latency/,
writes benchmarks/results/SUMMARY.md and SUMMARY.json.

* Retrieval quality (01): per (budget, dataset, method, metric) the mean over
  seeds of the per-seed mean, ± std over seeds (baselines do not depend on the
  MAPLE seed, so their std is 0). The per-seed 95% bootstrap CI half-width is
  shown in brackets for the first seed.
* Long context (07), needle (03), chunk ablation (05): first seed per budget.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.config import RESULTS_DIR

METHODS = ["random", "lead", "bm25", "dense", "maple", "oracle_last", "oracle_all_layers"]
Q_METRICS = ["oracle_recall@5", "oracle_ndcg@10", "evidence_recall@5", "all_evidence@5", "answer_in@5", "evidence_mrr@10"]


def _runs():
    out = defaultdict(dict)
    for d in sorted(RESULTS_DIR.glob("b*_s*")):
        m = re.fullmatch(r"b(\d+)_s(\d+)", d.name)
        if m:
            out[int(m.group(1))][int(m.group(2))] = d
    return out


def _pct(x):
    return f"{x * 100:.1f}"


def retrieval_tables(runs) -> tuple[list[str], dict]:
    lines, data = [], {}
    for budget, seeds in sorted(runs.items()):
        per = defaultdict(lambda: defaultdict(list))  # (ds, method) -> metric -> [seed means]
        ci = {}
        meta = {}
        for seed, d in sorted(seeds.items()):
            f = d / "retrieval_quality.json"
            if not f.exists():
                continue
            res = json.loads(f.read_text(encoding="utf-8"))
            for ds, r in res["datasets"].items():
                meta[ds] = (r["num_samples"], r["mean_candidates"], r["frac_with_gold"])
                for method, mets in r["summary"].items():
                    for metric in Q_METRICS:
                        s = mets.get(metric)
                        if s and s["n"]:
                            per[(ds, method)][metric].append(s["mean"])
                            ci.setdefault((ds, method, metric), (s["ci_high"] - s["ci_low"]) / 2)
        if not per:
            continue
        n_seeds = max(len(v) for d in per.values() for v in d.values())
        lines.append(f"\n### Teacher context budget {budget} tokens — held-out retrieval (benchmark 01, {n_seeds} seed(s))\n")
        for ds in sorted({k[0] for k in per}):
            n, cand, gold = meta[ds]
            lines.append(f"\n**{ds}** — {n} questions, {cand:.1f} candidate blocks/question, gold evidence for {gold * 100:.0f}%\n")
            lines.append("| method | " + " | ".join(Q_METRICS) + " |")
            lines.append("|---|" + "---|" * len(Q_METRICS))
            for method in METHODS:
                if (ds, method) not in per:
                    continue
                cells = []
                for metric in Q_METRICS:
                    v = per[(ds, method)].get(metric)
                    if not v:
                        cells.append("n/a")
                        continue
                    mean, std = float(np.mean(v)), float(np.std(v))
                    half = ci.get((ds, method, metric), 0.0)
                    cell = f"{_pct(mean)}" + (f" ± {_pct(std)}" if len(v) > 1 and std > 0 else "") + f" [±{_pct(half)}]"
                    cells.append(cell)
                    data.setdefault(str(budget), {}).setdefault(ds, {}).setdefault(method, {})[metric] = {
                        "mean": mean, "std_over_seeds": std, "seeds": len(v), "ci95_half_first_seed": half}
                lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.insert(0, "Cells: mean over seeds (%) ± std over seeds [± 95% bootstrap CI half-width of one run].")
    return lines, data


def long_context_tables(runs) -> list[str]:
    lines = []
    for budget, seeds in sorted(runs.items()):
        seed = min(seeds)
        f = seeds[seed] / "long_context.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        lines.append(f"\n### Long context (benchmark 07) — model b{budget}_s{seed}\n")
        if "hotpotqa" in res:
            h = res["hotpotqa"]
            sizes = h["pool_sizes"]
            lines.append(f"\n**HotpotQA + random distractors**, {h['num_questions']} questions — evidence_recall@5 (%) by pool size (blocks); p50 search ms at the largest pool\n")
            lines.append("| method | " + " | ".join(str(s) for s in sizes) + " | p50 ms | blocks scored |")
            lines.append("|---|" + "---|" * (len(sizes) + 2))
            methods = list(h["results"][str(sizes[0])])
            for m in methods:
                vals = [_pct(h["results"][str(s)][m]["evidence_recall@5"]["mean"]) for s in sizes]
                big = h["results"][str(sizes[-1])][m]
                lines.append(f"| {m} | " + " | ".join(vals) + f" | {big['latency_ms']['p50']:.2f} | {big['blocks_scored']['mean']:.0f} |")
        if "narrativeqa" in res:
            n = res["narrativeqa"]
            lines.append(f"\n**NarrativeQA full stories**, {n['num_questions']} questions (answer-span gold for {n['gold_coverage'] * 100:.0f}% of examined)\n")
            cols = ["evidence_recall@5", "evidence_recall@20", "all_evidence@20", "answer_in@5", "evidence_mrr@10"]
            lines.append("| method | " + " | ".join(cols) + " | p50 ms | blocks scored |")
            lines.append("|---|" + "---|" * (len(cols) + 2))
            for m, d in n["results"].items():
                vals = [_pct(d[c]["mean"]) if d.get(c) and d[c]["n"] else "n/a" for c in cols]
                lines.append(f"| {m} | " + " | ".join(vals) + f" | {d['latency_ms']['p50']:.2f} | {d['blocks_scored']['mean']:.0f} |")
    return lines


def needle_tables(runs) -> list[str]:
    lines = []
    for budget, seeds in sorted(runs.items()):
        seed = min(seeds)
        f = seeds[seed] / "needle_robustness.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        cfg = res["config"]
        lines.append(f"\n### Needle-in-a-haystack (benchmark 03) — model b{budget}_s{seed}, held-out templates, "
                     f"{cfg['trials']} trials × {len(cfg['lengths'])} lengths × {len(cfg['depths'])} depths\n")
        lengths = cfg["lengths"]
        methods = list(res["overall"])
        lines.append("| method | overall success@5 [95% CI] | " + " | ".join(f"{l // 1000}k chars" for l in lengths) + " |")
        lines.append("|---|---|" + "---|" * len(lengths))
        for m in methods:
            o = res["overall"][m]
            by_len = []
            for l in lengths:
                cells = [c[m]["mean"] for c in res["cells"] if c["length"] == l and m in c]
                by_len.append(_pct(float(np.mean(cells))) if cells else "n/a")
            lines.append(f"| {m} | {_pct(o['mean'])} [{_pct(o['ci_low'])}, {_pct(o['ci_high'])}] | " + " | ".join(by_len) + " |")
    return lines


def chunk_tables(runs) -> list[str]:
    lines = []
    for budget, seeds in sorted(runs.items()):
        seed = min(seeds)
        f = seeds[seed] / "ablation_chunking.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        sizes = res["config"]["chunk_sizes"]
        lines.append(f"\n### Chunk size at a {res['config']['budget_chars']}-char budget (benchmark 05, HotpotQA) — model b{budget}_s{seed}\n")
        lines.append("| method | " + " | ".join(f"{s} chars (k={res['results'][str(s)]['k_at_budget']})" for s in sizes) + " |")
        lines.append("|---|" + "---|" * len(sizes))
        for m in res["results"][str(sizes[0])]["methods"]:
            vals = [_pct(res["results"][str(s)]["methods"][m]["evidence_recall"]["mean"]) for s in sizes]
            lines.append(f"| {m} | " + " | ".join(vals) + " |")
    return lines


def latency_table() -> list[str]:
    f = RESULTS_DIR / "latency" / "latency_results.json"
    if not f.exists():
        return []
    res = json.loads(f.read_text(encoding="utf-8"))
    rows = res.get("metrics", res)
    lines = ["\n### Search latency vs index size (benchmark 02, real BGE embeddings, p50 / p95 ms)\n",
             "| blocks | linear | hierarchical | adaptive | linear ÷ adaptive |", "|---|---|---|---|---|"]
    for r in rows:
        t = r.get("latency_ms", {})
        cell = lambda s: f"{t[s]['p50']:.2f} / {t[s]['p95']:.2f}" if s in t else "n/a"
        lines.append(f"| {r['n_blocks']:,} | {cell('linear')} | {cell('hierarchical')} | {cell('adaptive')} | {r['speedup']:.1f}× |")
    return lines


def main():
    runs = _runs()
    ret_lines, ret_data = retrieval_tables(runs)
    md = ["# MAPLE — benchmark summary", "",
          "Generated by `benchmarks/aggregate_results.py`. Metric definitions: docs/METRICS.md.", ""]
    md += ["## Retrieval quality on held-out oracle sets", ""] + ret_lines
    md += ["", "## Long-context evidence retrieval"] + long_context_tables(runs)
    md += ["", "## Robustness"] + needle_tables(runs)
    md += ["", "## Chunk-size ablation"] + chunk_tables(runs)
    md += ["", "## Efficiency"] + latency_table()
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS_DIR / "SUMMARY.json").write_text(json.dumps({"retrieval_quality": ret_data}, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    from scripts.runlog import record_outputs, start_run
    with start_run("aggregate", "summary", {"argv": sys.argv[1:]}) as _run:
        main()
        record_outputs(_run, RESULTS_DIR / 'SUMMARY.md', RESULTS_DIR / 'SUMMARY.json')
