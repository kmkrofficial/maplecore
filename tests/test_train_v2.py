import json
from pathlib import Path

import pytest
import torch

from maplecore import load_model
from scripts.embedding_cache import build_cache
from scripts.train_v2 import rows_to_groups, run

from test_embedding_cache import FakeIndexer


def _write_oracle(tmp_path: Path, n_samples: int, blocks_per_sample, with_gold: bool = True,
                  degenerate=()) -> Path:
    samples = []
    for i in range(n_samples):
        n = blocks_per_sample[i]
        mass = {str(b): (0.0 if i in degenerate else 0.01 * (b + 1)) for b in range(n)}
        samples.append({
            "id": f"s{i}", "domain": "hotpot_qa", "question": f"question {i}?",
            "blocks": [{"id": b, "text": f"sample {i} block {b}"} for b in range(n)],
            "top_5_block_ids": [n - 1], "gold_block_ids": [0] if with_gold else [],
            "attention_scores": mass, "attention_scores_all_layers": mass,
        })
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({"metadata": {"schema_version": 2}, "samples": samples}), encoding="utf-8")
    return path


def test_train_v2_end_to_end_from_cache(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=40, blocks_per_sample=[6] * 40)
    cache = build_cache(oracle, tmp_path / "c.pt", indexer=FakeIndexer(8, "fake-enc"))
    out = run([oracle], tmp_path / "m.pt", encoder="fake-enc", epochs=2, cache_paths=[cache], device="cpu")
    assert load_model(tmp_path / "m.pt", device="cpu").input_dim == 16
    summary = json.loads((tmp_path / "m.train.json").read_text(encoding="utf-8"))
    assert summary["target_source"] == "all_layers" and summary["groups"]["train"] + summary["groups"]["val"] == 40
    assert out["metric"] == "val_evidence_recall@5"
    assert summary["seed"] == 42 and summary["model_kind"] == "residual"


def test_degenerate_samples_are_skipped_and_counted(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=20, blocks_per_sample=[5] * 20, degenerate=(3, 7))
    cache = build_cache(oracle, tmp_path / "c.pt", indexer=FakeIndexer(8, "fake-enc"))
    run([oracle], tmp_path / "m.pt", encoder="fake-enc", epochs=1, cache_paths=[cache], device="cpu")
    summary = json.loads((tmp_path / "m.train.json").read_text(encoding="utf-8"))
    assert summary["skipped"] == {"degenerate_target": 2}
    assert summary["groups"]["train"] + summary["groups"]["val"] == 18


def test_mlp_model_kind_trains(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=20, blocks_per_sample=[5] * 20)
    cache = build_cache(oracle, tmp_path / "c.pt", indexer=FakeIndexer(8, "fake-enc"))
    run([oracle], tmp_path / "m.pt", model_kind="mlp", encoder="fake-enc", epochs=1, cache_paths=[cache], device="cpu")
    assert type(load_model(tmp_path / "m.pt", device="cpu")).__name__ == "MapleNet"


def test_cache_encoder_mismatch_raises(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=5, blocks_per_sample=[4] * 5)
    cache = build_cache(oracle, tmp_path / "c.pt", indexer=FakeIndexer(8, "enc-a"))
    with pytest.raises(ValueError, match="enc-a"):
        run([oracle], tmp_path / "m.pt", encoder="enc-b", epochs=1, cache_paths=[cache], device="cpu")


def test_cache_not_matching_oracle_raises(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=5, blocks_per_sample=[4] * 5)
    other = tmp_path / "other"
    other.mkdir()
    other_oracle = _write_oracle(other, n_samples=4, blocks_per_sample=[4] * 4)
    cache = build_cache(other_oracle, tmp_path / "c.pt", indexer=FakeIndexer(8, "fake-enc"))
    with pytest.raises(ValueError, match="does not match"):
        run([oracle], tmp_path / "m.pt", encoder="fake-enc", epochs=1, cache_paths=[cache], device="cpu")


def test_unknown_target_source_raises(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=5, blocks_per_sample=[4] * 5)
    cache = build_cache(oracle, tmp_path / "c.pt", indexer=FakeIndexer(8, "fake-enc"))
    with pytest.raises(ValueError, match="nope"):
        run([oracle], tmp_path / "m.pt", target_source="nope", encoder="fake-enc", epochs=1,
            cache_paths=[cache], device="cpu")


def test_rows_to_groups_converts_flat_rows():
    q = torch.randn(8)
    rows = [{"group": "needle::0", "query_emb": q, "block_emb": torch.randn(8), "label": 1.0},
            {"group": "needle::0", "query_emb": q, "block_emb": torch.randn(8), "label": 0.0},
            {"group": "needle::0", "query_emb": q, "block_emb": torch.randn(8), "label": 0.0},
            {"group": "needle::1", "query_emb": q, "block_emb": torch.randn(8), "label": 1.0}]
    groups = {g.group: g for g in rows_to_groups(rows)}
    assert groups["needle::0"].blocks.shape == (3, 8) and groups["needle::0"].target.tolist() == [1.0, 0.0, 0.0]
    assert groups["needle::0"].gold.tolist() == [True, False, False]
    assert groups["needle::1"].blocks.shape == (1, 8)
