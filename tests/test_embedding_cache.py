import json
import zlib
from pathlib import Path

import pytest
import torch

from scripts.embedding_cache import build_cache, cache_path_for, load_cache


class FakeIndexer:
    """Deterministic stand-in for MapleIndexer (no model download)."""

    def __init__(self, dim: int, name: str):
        self.dim, self.model_name = dim, name

    def _vec(self, text: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(zlib.crc32(text.encode("utf-8")) % 1000)
        return torch.randn(self.dim, generator=g)

    def encode_query(self, q: str) -> torch.Tensor:
        return self._vec("q:" + q)

    def encode_texts(self, texts, show_progress: bool = False) -> torch.Tensor:
        return torch.stack([self._vec(t) for t in texts]) if texts else torch.empty(0, self.dim)


def _write_oracle(tmp_path: Path, n_samples: int, blocks_per_sample) -> Path:
    samples = []
    for i in range(n_samples):
        n = blocks_per_sample[i]
        samples.append({
            "id": f"s{i}", "domain": "hotpot_qa", "question": f"question {i}?",
            "blocks": [{"id": b, "text": f"sample {i} block {b}"} for b in range(n)],
            "top_5_block_ids": [0], "gold_block_ids": [],
        })
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({"metadata": {"schema_version": 2}, "samples": samples}), encoding="utf-8")
    return path


def test_build_and_load_cache_roundtrip(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=3, blocks_per_sample=[4, 2, 5])
    out = build_cache(oracle, tmp_path / "c.pt", indexer=FakeIndexer(dim=8, name="fake-enc"))
    c = load_cache(out, expected_encoder="fake-enc")
    assert c["ids"] == ["s0", "s1", "s2"]
    assert c["q"].shape == (3, 8) and c["q"].dtype == torch.float16
    assert [b.shape[0] for b in c["blocks"]] == [4, 2, 5]
    assert c["block_ids"][2] == [0, 1, 2, 3, 4]
    assert c["encoder"] == "fake-enc" and c["oracle"] == str(oracle).replace("\\", "/")


def test_cached_vectors_match_encoder_output(tmp_path):
    oracle = _write_oracle(tmp_path, n_samples=1, blocks_per_sample=[3])
    enc = FakeIndexer(8, "fake-enc")
    c = load_cache(build_cache(oracle, tmp_path / "c.pt", indexer=enc))
    expected = enc.encode_texts([f"sample 0 block {b}" for b in range(3)]).half()
    assert torch.equal(c["blocks"][0], expected)
    assert torch.equal(c["q"][0], enc.encode_query("question 0?").half())


def test_cache_encoder_mismatch_raises(tmp_path):
    out = build_cache(_write_oracle(tmp_path, 1, [2]), tmp_path / "c.pt", indexer=FakeIndexer(8, "enc-a"))
    with pytest.raises(ValueError, match="enc-a"):
        load_cache(out, expected_encoder="enc-b")


def test_cache_path_for():
    p = cache_path_for(Path("data/oracle/hotpotqa_train.json"), "BAAI/bge-small-en-v1.5")
    assert p == Path("data/cache/hotpotqa_train__BAAI--bge-small-en-v1.5.pt")


def test_cache_distinguishes_budget_directories():
    # data/oracle/t4096/hotpotqa_train.json must not collide with data/oracle/hotpotqa_train.json
    a = cache_path_for(Path("data/oracle/hotpotqa_train.json"), "enc")
    b = cache_path_for(Path("data/oracle/t4096/hotpotqa_train.json"), "enc")
    assert a != b
