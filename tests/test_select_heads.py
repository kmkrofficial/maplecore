import json
from pathlib import Path

import pytest
import torch

from scripts.select_heads import main, score_heads, select_top_heads


def _synthetic(layers: int = 4, heads: int = 8, focused=(3, 5), n_samples: int = 12, n_blocks: int = 6, seed: int = 0):
    """Head `focused` puts ~90% of its mass on the gold blocks; every other head is diffuse noise."""
    g = torch.Generator().manual_seed(seed)
    samples, mass, order = [], {}, {}
    for i in range(n_samples):
        gold = [i % n_blocks]
        sample = {"id": f"s{i}", "gold_block_ids": gold,
                  "blocks": [{"id": b, "text": f"t{b}"} for b in range(n_blocks)]}
        m = torch.rand(layers, heads, n_blocks, generator=g)
        m = m / m.sum(-1, keepdim=True)                       # diffuse heads: ~uniform mass
        focus = torch.full((n_blocks,), 0.02)
        focus[gold[0]] = 0.9
        m[focused[0], focused[1]] = focus / focus.sum()
        samples.append(sample)
        mass[sample["id"]] = m.half()
        order[sample["id"]] = list(range(n_blocks))
    return samples, {"block_order": order, "mass": mass}


def test_selects_head_focused_on_gold():
    samples, heads = _synthetic(layers=4, heads=8, focused=(3, 5))
    scores = score_heads(samples, heads)
    assert scores.shape == (4, 8)
    assert select_top_heads(scores, k=3)[0] == (3, 5)


def test_select_top_heads_orders_by_score_and_clips_k():
    scores = torch.tensor([[0.1, 0.9], [0.5, 0.2]])
    assert select_top_heads(scores, k=2) == [(0, 1), (1, 0)]
    assert len(select_top_heads(scores, k=10)) == 4


def test_samples_without_gold_are_ignored():
    samples, heads = _synthetic(focused=(1, 1))
    samples.append({"id": "nogold", "gold_block_ids": [], "blocks": [{"id": 0, "text": "x"}]})
    heads["mass"]["nogold"] = torch.rand(4, 8, 1).half()
    heads["block_order"]["nogold"] = [0]
    assert select_top_heads(score_heads(samples, heads), k=1) == [(1, 1)]


def _write_oracle(tmp_path: Path, split: str, samples, heads) -> Path:
    path = tmp_path / f"{split}.json"
    path.write_text(json.dumps({"metadata": {"split": split, "teacher_model": "tiny"}, "samples": samples}), encoding="utf-8")
    torch.save(heads, path.with_suffix(".heads.pt"))
    return path


def test_select_heads_cli_writes_head_list(tmp_path):
    samples, heads = _synthetic(focused=(2, 4))
    oracle = _write_oracle(tmp_path, "train", samples, heads)
    out = tmp_path / "heads.json"
    main([str(oracle), "--k", "4", "--out", str(out)])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["k"] == 4 and res["heads"][0] == [2, 4] and res["teacher"] == "tiny"
    assert res["source"].endswith("train.json")


def test_select_heads_rejects_heldout(tmp_path):
    samples, heads = _synthetic()
    oracle = _write_oracle(tmp_path, "validation", samples, heads)
    with pytest.raises(ValueError, match="train"):
        main([str(oracle), "--k", "4", "--out", str(tmp_path / "h.json")])
