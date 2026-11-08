import math

import numpy as np
import pytest
import torch

from maplecore import MapleNet, MapleScanner
from maplecore.indexer import compute_chapter_embeddings

from conftest import DotModel, make_index


def _plant(index, block_id, dim=16):
    q = torch.zeros(dim)
    q[0] = 1.0
    index.embeddings[block_id] = q * 3
    return q


def test_chapter_embeddings_include_partial_tail():
    emb = torch.arange(25 * 2, dtype=torch.float32).reshape(25, 2)
    ch = compute_chapter_embeddings(emb, 10)
    assert ch.shape == (3, 2)
    assert torch.allclose(ch[2], emb[20:].mean(dim=0))


@pytest.mark.parametrize("strategy", ["linear", "hierarchical", "adaptive"])
def test_tail_block_is_reachable(dot_model, strategy):
    """Regression for B1: blocks after the last full chapter were unsearchable."""
    index = make_index(1049)  # 10 full chapters + 49-block tail
    q = _plant(index, 1040)
    scanner = MapleScanner(dot_model, device="cpu", chapter_size=100, fast_chapters=2)
    res = scanner.search(q, index, k=5, strategy=strategy)
    assert res.block_ids[0] == 1040


def test_adaptive_returns_at_least_k_and_respects_cap(dot_model):
    index = make_index(1000)
    q = _plant(index, 10)
    scanner = MapleScanner(dot_model, device="cpu", max_blocks=7, chapter_size=100)
    res = scanner.search(q, index, k=3)
    assert 3 <= res.num_results <= 7
    assert len(set(res.block_ids)) == res.num_results
    assert res.block_ids[0] == 10


def test_adaptive_small_index_scans_everything(dot_model):
    index = make_index(50)
    q = _plant(index, 49)
    res = MapleScanner(dot_model, device="cpu").search(q, index, k=5)
    assert res.metadata["blocks_scored"] == 50
    assert res.block_ids[0] == 49


def test_entropy_threshold_triggers_fallback(dot_model):
    index = make_index(2000)
    q = torch.zeros(16)  # flat scores -> maximal entropy
    scanner = MapleScanner(dot_model, device="cpu", confidence_threshold=0.0,
                           entropy_threshold=0.5, fast_chapters=2, fallback_chapters=10)
    res = scanner.search(q, index, k=5)
    assert res.metadata["action"].startswith("Fallback")
    assert res.metadata["blocks_scored"] == 1000
    assert 0.99 < res.metadata["normalized_entropy"] <= 1.0 + 1e-6


class _Batch1OnnxStub:
    """Mimics MapleONNXRunner returning a 0-d array for a single row (B3)."""

    def __init__(self, dim):
        self.dim = dim

    def predict(self, x):
        q, b = x[:, : self.dim], x[:, self.dim:]
        s = 1 / (1 + np.exp(-(q * b).sum(axis=1)))
        return s.squeeze()


def test_onnx_style_model_path_handles_single_row():
    """Regression for B2/B3: ONNX path referenced undefined `logits`."""
    index = make_index(1)
    scanner = MapleScanner(_Batch1OnnxStub(16), device="cpu")
    res = scanner.search(index.embeddings[0], index, k=5, strategy="linear")
    assert res.block_ids == [0]


def test_mean_pooled_chapter_scores_are_cached(dot_model):
    index = make_index(1000)
    MapleScanner(dot_model, device="cpu").search(torch.ones(16), index, k=5)
    assert 100 in index._chapter_cache


def test_unknown_strategy_raises(dot_model):
    with pytest.raises(ValueError):
        MapleScanner(dot_model, device="cpu").search(torch.ones(16), make_index(5), strategy="nope")


def test_maplenet_roundtrip(tmp_path):
    m = MapleNet(input_dim=32, hidden_dim=8)
    p = tmp_path / "m.pth"
    m.save(p)
    loaded = MapleNet.load(p, device="cpu")
    assert loaded.input_dim == 32
    x = torch.randn(4, 32)
    m.eval()
    assert torch.allclose(m(x), loaded(x))


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from maplecore.onnx_runner import MapleONNXRunner

    m = MapleNet(input_dim=32, hidden_dim=8).eval()
    path = tmp_path / "m.onnx"
    m.export_to_onnx(path)
    runner = MapleONNXRunner(path)

    x = torch.randn(5, 32)
    expected = torch.sigmoid(m(x)).detach().numpy()
    np.testing.assert_allclose(runner.predict(x.numpy()), expected, rtol=1e-4, atol=1e-5)
    assert runner.predict(x[:1].numpy()).shape == (1,)

    index = make_index(300, dim=16)
    res_t = MapleScanner(m, device="cpu").search(torch.randn(16), index, k=5, strategy="linear")
    res_o = MapleScanner(runner, device="cpu").search(torch.randn(16), index, k=5, strategy="linear")
    assert res_o.num_results == res_t.num_results == 5
