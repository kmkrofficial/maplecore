import json
from pathlib import Path

import pytest

from benchmarks.aggregate_results import long_context_variant_tables, variant_tables


def _summ(mean, n=100):
    return {"mean": mean, "ci_low": mean - 0.02, "ci_high": mean + 0.02, "n": n}


def _write_run(root: Path, name: str, maple_ev5: float, dense_ev5: float = 0.70, items_maple=None):
    d = root / name
    d.mkdir(parents=True)
    metrics = ["oracle_recall@5", "evidence_recall@5", "all_evidence@5", "answer_in@5"]
    summary = {
        "bm25": {m: _summ(0.5) for m in metrics},
        "dense": {m: _summ(dense_ev5 if m == "evidence_recall@5" else 0.5) for m in metrics},
        "maple": {m: _summ(maple_ev5 if m == "evidence_recall@5" else 0.6) for m in metrics},
        "oracle_all_layers": {m: _summ(0.8) for m in metrics},
    }
    res = {"config": {}, "datasets": {"HotpotQA": {"summary": summary, "num_samples": 100,
                                                    "mean_candidates": 11.0, "frac_with_gold": 0.98}}}
    (d / "retrieval_quality.json").write_text(json.dumps(res), encoding="utf-8")
    n = 40
    per_item = {"HotpotQA": {"per_item": {
        "maple": {"evidence_recall@5": items_maple if items_maple is not None else [maple_ev5] * n},
        "dense": {"evidence_recall@5": [dense_ev5] * n}}}}
    (d / "retrieval_quality_per_item.json").write_text(
        json.dumps({"datasets": {"HotpotQA": per_item["HotpotQA"]}}), encoding="utf-8")


def test_variant_table_mean_std_over_seeds(tmp_path):
    _write_run(tmp_path, "A1_b1536_s42", 0.80)
    _write_run(tmp_path, "A1_b1536_s43", 0.82)
    _write_run(tmp_path, "A2_b1536_s42", 0.75)
    lines, data = variant_tables(tmp_path)
    text = "\n".join(lines)
    assert "81.0 ± 1.0" in text                      # A1: mean 81.0, std 1.0 over two seeds
    assert "| A2 |" in text and "| dense |" in text and "| bm25 |" in text and "oracle_all_layers" in text
    assert data["1536"]["HotpotQA"]["A1"]["evidence_recall@5"]["mean"] == pytest.approx(0.81)
    assert data["1536"]["HotpotQA"]["A1"]["evidence_recall@5"]["seeds"] == 2
    assert data["1536"]["HotpotQA"]["dense"]["evidence_recall@5"]["mean"] == pytest.approx(0.70)


def test_variant_table_paired_pvalue_vs_dense(tmp_path):
    better = [1.0] * 40
    worse = [0.0] * 40
    _write_run(tmp_path, "A1_b1536_s42", 1.0, dense_ev5=0.0, items_maple=better)   # dense per-item all 0.0
    _write_run(tmp_path, "A2_b1536_s42", 0.0, dense_ev5=0.0, items_maple=worse)
    lines, data = variant_tables(tmp_path)
    assert data["1536"]["HotpotQA"]["A1"]["p_vs_dense"] < 0.01
    assert data["1536"]["HotpotQA"]["A2"]["p_vs_dense"] != data["1536"]["HotpotQA"]["A2"]["p_vs_dense"] \
        or data["1536"]["HotpotQA"]["A2"]["p_vs_dense"] > 0.5   # identical -> no evidence of a difference (nan or high)


def test_budgets_are_reported_separately(tmp_path):
    _write_run(tmp_path, "A1_b1536_s42", 0.8)
    _write_run(tmp_path, "A1_b4096_s42", 0.6)
    lines, data = variant_tables(tmp_path)
    assert set(data) == {"1536", "4096"}
    assert data["4096"]["HotpotQA"]["A1"]["evidence_recall@5"]["mean"] == pytest.approx(0.6)


def test_ignores_folders_without_results(tmp_path):
    (tmp_path / "A1_b1536_s42").mkdir()
    (tmp_path / "notes").mkdir()
    lines, data = variant_tables(tmp_path)
    assert data == {}


def _write_lc(root: Path, name: str, maple_by_pool: dict, nqa: float = 0.05):
    results = {str(p): {"dense": {"evidence_recall@5": _summ(0.86)}, "bm25": {"evidence_recall@5": _summ(0.65)},
                        "maple_linear": {"evidence_recall@5": _summ(v)}} for p, v in maple_by_pool.items()}
    lc = {"hotpotqa": {"pool_sizes": list(maple_by_pool), "num_questions": 300, "results": results},
          "narrativeqa": {"num_questions": 300, "gold_coverage": 0.35, "results": {
              "dense": {"evidence_recall@5": _summ(0.10)}, "bm25": {"evidence_recall@5": _summ(0.06)},
              "maple_linear": {"evidence_recall@5": _summ(nqa)}}}}
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "long_context.json").write_text(json.dumps(lc), encoding="utf-8")


def test_long_context_variant_table(tmp_path):
    _write_lc(tmp_path, "A1_b1536_s42", {10: 0.70, 1000: 0.80}, nqa=0.12)
    _write_lc(tmp_path, "A2_b1536_s42", {10: 0.70, 1000: 0.10})
    lines, data = long_context_variant_tables(tmp_path)
    assert data["A1"]["hotpotqa"]["1000"] == pytest.approx(0.80) and data["A2"]["hotpotqa"]["1000"] == pytest.approx(0.10)
    assert data["dense"]["hotpotqa"]["1000"] == pytest.approx(0.86) and data["A1"]["narrativeqa"] == pytest.approx(0.12)
    text = "\n".join(lines)
    assert "| A1 |" in text and "| A2 |" in text and "| dense |" in text and "| bm25 |" in text


def test_long_context_variant_table_empty_without_results(tmp_path):
    (tmp_path / "A1_b1536_s42").mkdir()
    assert long_context_variant_tables(tmp_path) == ([], {})
