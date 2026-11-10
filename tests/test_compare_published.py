import json
from pathlib import Path

import pytest

from benchmarks.compare_published import (ComparisonRow, PublishedResult, collect_ours, join, load_published,
                                          render_markdown)


def _pub(protocol, method="ICR (Llama-3.1-8B)", value=86.2, dataset="hotpotqa", metric="recall@5"):
    return PublishedResult("ICR", "ICLR25", protocol, dataset, metric, method, value)


def _row(delta):
    return ComparisonRow("icr_multihop", "hotpotqa", "recall@5", "maple:v2", 87.0, "ICR", "ICR (Llama-3.1-8B)", 86.2,
                         delta, "comparable")


def test_join_comparable_row_has_delta():
    pub = [_pub("icr_multihop")]
    ours = {"icr_multihop": {"ok": True, "rows": {("hotpotqa", "recall@5", "maple:v2"): 87.0}}}
    (row,) = join(pub, ours)
    assert row.status == "comparable" and row.delta == pytest.approx(0.8) and row.ours_method == "maple:v2"


def test_one_row_per_ours_method():
    pub = [_pub("icr_multihop")]
    ours = {"icr_multihop": {"ok": True, "rows": {("hotpotqa", "recall@5", "maple:v2"): 87.0,
                                                  ("hotpotqa", "recall@5", "dense"): 80.0}}}
    rows = join(pub, ours)
    assert {r.ours_method for r in rows} == {"maple:v2", "dense"}


def test_mismatched_protocol_excluded():
    ours = {"icr_multihop": {"ok": False, "rows": {("hotpotqa", "recall@5", "maple:v2"): 87.0}}}
    (row,) = join([_pub("icr_multihop")], ours)
    assert row.status == "protocol mismatch" and row.delta is None


def test_not_run_protocol():
    (row,) = join([_pub("longbench_v1_llama3_8b")], {})
    assert row.status == "not run" and row.ours_value is None


def test_missing_cell_inside_run_protocol_is_not_run():
    ours = {"icr_multihop": {"ok": True, "rows": {("2wiki", "recall@5", "maple:v2"): 70.0}}}
    (row,) = join([_pub("icr_multihop")], ours)
    assert row.status == "not run"


def test_render_marks_sign_and_status():
    md = render_markdown([_row(0.8), _row(-1.2)])
    assert "+0.8" in md and "−1.2" in md
    assert "protocol mismatch" in render_markdown(
        join([_pub("icr_multihop")], {"icr_multihop": {"ok": False, "rows": {("hotpotqa", "recall@5", "m"): 1.0}}}))


def test_load_published_roundtrip(tmp_path):
    p = tmp_path / "pub.json"
    p.write_text(json.dumps([{"paper": "ICR", "venue": "ICLR25", "protocol": "icr_multihop", "dataset": "hotpotqa",
                              "metric": "recall@5", "method": "ICR (Llama-3.1-8B)", "value": 86.2}]), encoding="utf-8")
    (r,) = load_published(p)
    assert r.value == 86.2 and r.note == ""


def _summ(v):
    return {"mean": v, "ci_low": v - .01, "ci_high": v + .01, "n": 10}


def _write(d: Path, name: str, obj):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


def test_collect_ours_reads_multihop_beir_and_e2e_results(tmp_path):
    _write(tmp_path, "multihop_rerank.json", {"protocol_ok": False, "datasets": {"hotpotqa": {
        "_size": {"questions": 10}, "first_stage": {"recall@5": _summ(0.79)},
        "maple:main": {"recall@5": _summ(0.87), "recall@2": _summ(0.7), "all_recall@5": _summ(0.7)}}}})
    _write(tmp_path / "sub", "beir_rerank.json", {
        "config": {"first_stage_k": 100}, "protocol_mismatch": False,
        "datasets": {"scifact": {"_first_stage": {"bm25": {"ndcg@10": _summ(0.679)}}, "dense": {"ndcg@10": _summ(0.7)}}}})
    _write(tmp_path, "e2e_reader_fixed.json", {"protocol_ok": True, "tasks": {
        "longbench_hotpotqa": {"full": {"metrics": {"f1": _summ(0.47)}}},
        "narrativeqa_fullstory": {"dense": {"metrics": {"rouge_l": 31.0, "meteor": None}}}}})
    ours = collect_ours(tmp_path)
    mh = ours["icr_multihop"]
    assert mh["ok"] is False and mh["rows"][("hotpotqa", "recall@5", "maple:main")] == pytest.approx(87.0)
    assert ("hotpotqa", "recall@5", "first_stage") in mh["rows"] and not any(k[2] == "_size" for k in mh["rows"])
    b = ours["beir_bm25top100"]
    assert b["ok"] is True and b["rows"][("scifact", "ndcg@10", "bm25")] == pytest.approx(67.9)
    assert ours["beir_bm25top200"]["ok"] is False                    # run used k=100, not 200
    assert ours["longbench_v1_llama3_8b"]["rows"][("hotpotqa", "f1", "full")] == pytest.approx(47.0)
    nqa = ours["narrativeqa_fullstory_e2e"]
    assert nqa["ok"] is False and nqa["rows"] == {("narrativeqa", "rouge_l", "dense"): 31.0}


def test_collect_ours_empty_directory(tmp_path):
    assert collect_ours(tmp_path) == {}
