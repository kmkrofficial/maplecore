import copy
import json

import pytest
import torch

from maplecore import MapleIndexer, MapleTrainer
from maplecore.context_manager import ContextSynthesizer
from maplecore.indexer import Index, query_prefix_for
from maplecore.orchestrator import RecursiveOrchestrator
from maplecore.utils import resolve_device

from conftest import make_index


# ---------------------------------------------------------------- utils

def test_resolve_device_auto_and_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == "cpu"
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_ignores_cuda_that_reports_no_devices(monkeypatch):
    """torch can report is_available() while device_count() is 0 (e.g. CUDA_VISIBLE_DEVICES='' on Windows)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert resolve_device(None) == "cpu"
    assert resolve_device("cuda") == "cpu"


# -------------------------------------------------------------- indexer

def test_chunk_text_no_overlap_matches_legacy_behaviour():
    idx = MapleIndexer(device="cpu")
    text = "abcdefghij" * 5  # 50 chars
    blocks = idx.chunk_text(text, chunk_size=20)
    assert [b.text for b in blocks] == [text[0:20], text[20:40], text[40:50]]
    assert [(b.start_char, b.end_char) for b in blocks] == [(0, 20), (20, 40), (40, 50)]


def test_chunk_text_overlap():
    idx = MapleIndexer(device="cpu")
    blocks = idx.chunk_text("x" * 50, chunk_size=20, overlap=5)
    assert [b.start_char for b in blocks] == [0, 15, 30]
    with pytest.raises(ValueError):
        idx.chunk_text("x", chunk_size=10, overlap=10)


def test_query_prefix_only_for_bge():
    assert query_prefix_for("BAAI/bge-small-en-v1.5").startswith("Represent")
    assert query_prefix_for("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_index_save_load_roundtrip(tmp_path, caplog):
    index = make_index(7)
    MapleIndexer(model_name="test-encoder", device="cpu").save_index(index, tmp_path / "doc")
    loaded = MapleIndexer(model_name="test-encoder", device="cpu").load_index(tmp_path / "doc")
    assert loaded.num_blocks == 7
    assert loaded.embedding_model == "test-encoder"
    assert torch.allclose(loaded.embeddings, index.embeddings)
    assert loaded.blocks[3].text == "block 3"

    with caplog.at_level("WARNING"):
        MapleIndexer(model_name="other-encoder", device="cpu").load_index(tmp_path / "doc")
    assert "Query embeddings will not match" in caplog.text


# ------------------------------------------------------ context manager

def test_context_synthesizer_dedup_and_window():
    cs = ContextSynthesizer(max_tokens=10, chars_per_token=1.0)  # 10 chars
    assert cs.add_chunks(["aaaa", "bbbb"]) == 2
    assert cs.add_chunks(["aaaa"]) == 0
    assert cs.add_chunks(["cccc"]) == 1  # evicts "aaaa"
    assert cs.get_context_string() == "bbbb\n---\ncccc"
    # Evicted chunks are not re-added (prevents search loops)
    assert cs.add_chunks(["aaaa"]) == 0


def test_context_synthesizer_keeps_single_oversized_chunk():
    cs = ContextSynthesizer(max_tokens=2, chars_per_token=1.0)
    cs.add_chunks(["a long chunk"])
    assert cs.get_context_string() == "a long chunk"


# --------------------------------------------------------- orchestrator

class _FakeIndexer:
    def encode_query(self, q):
        return q


class _FakeScanner:
    """Returns a fresh block for every new query string."""

    def __init__(self):
        self.calls = 0

    def search(self, q, index, strategy="adaptive"):
        self.calls += 1

        class R:
            block_ids = [self.calls - 1]
        return R()


@pytest.mark.parametrize("max_hops", [0, 1, 3])
def test_orchestrator_respects_max_hops(max_hops):
    index = make_index(20)
    llm_calls = []

    def llm(prompt):
        llm_calls.append(prompt)
        if "MAX SEARCHES REACHED" in prompt:
            return "final"
        return f'[SEARCH: "q{len(llm_calls)}"]'

    scanner = _FakeScanner()
    orch = RecursiveOrchestrator(_FakeIndexer(), scanner, max_hops=max_hops)
    answer, hops = orch.generate_answer("question?", index, llm, top_k=1)
    assert answer == "final"
    assert hops == max_hops
    assert scanner.calls == max_hops + 1          # initial + one per hop
    assert len(llm_calls) == max_hops + 2         # one per round + final fallback


def test_orchestrator_returns_direct_answer():
    orch = RecursiveOrchestrator(_FakeIndexer(), _FakeScanner(), max_hops=3)
    answer, hops = orch.generate_answer("q", make_index(5), lambda p: " Paris ", top_k=1)
    assert (answer, hops) == ("Paris", 0)


# --------------------------------------------------------------- trainer

def _toy_rows(n_groups=20, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for q in range(n_groups):
        qe = torch.randn(dim, generator=g)
        for b in range(6):
            pos = b < 2
            be = qe + 0.1 * torch.randn(dim, generator=g) if pos else torch.randn(dim, generator=g)
            rows.append({"group": f"g{q}", "question": "same question", "query_emb": qe,
                         "block_emb": be, "label": float(pos), "block_id": b})
    return rows


def test_trainer_returns_best_epoch_weights(monkeypatch):
    """Regression for B4: shallow state_dict().copy() returned the last epoch."""
    trainer = MapleTrainer(input_dim=16, hidden_dim=8, device="cpu", seed=0)
    snapshots = []
    recalls = iter([0.9, 0.1, 0.1, 0.1])

    def fake_recall(model, loader, k=5):
        snapshots.append(copy.deepcopy(model.state_dict()))
        return next(recalls)

    monkeypatch.setattr(trainer, "compute_recall_at_k", fake_recall)
    model, best = trainer.train(_toy_rows(), epochs=4, batch_size=8, lr=1e-2)
    assert best == 0.9
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), snapshots[0][k].cpu()), k
        assert not torch.equal(v.cpu(), snapshots[-1][k].cpu()) or "running" in k


def test_trainer_splits_by_group_and_is_deterministic():
    rows = _toy_rows()
    t1 = MapleTrainer(input_dim=16, hidden_dim=8, device="cpu", seed=1)
    _, r1 = t1.train(rows, epochs=3, batch_size=8, lr=1e-2)
    t2 = MapleTrainer(input_dim=16, hidden_dim=8, device="cpu", seed=1)
    _, r2 = t2.train(rows, epochs=3, batch_size=8, lr=1e-2)
    assert r1 == r2
    assert r1 > 0.5  # learnable toy task; all rows share one question string


# ------------------------------------------------------------- oracle io

def test_load_oracle_samples_normalises_v1_and_v2(tmp_path):
    from scripts.oracle_common import load_oracle_samples

    v1_nqa = {"metadata": {}, "samples": [{"question": "q", "top_5_block_ids": ["1", "2"],
                                           "all_block_texts": {"1": "a", "2": "b", "3": "c"}}]}
    v1_hot = [{"id": "h1", "question": "q", "labels": [0], "domain": "hotpot_qa",
               "blocks": [{"id": 0, "text": "x"}]}]
    (tmp_path / "a.json").write_text(json.dumps(v1_nqa))
    (tmp_path / "b.json").write_text(json.dumps(v1_hot))

    a = load_oracle_samples(tmp_path / "a.json")[0]
    assert a["top_5_block_ids"] == [1, 2] and len(a["blocks"]) == 3 and a["gold_block_ids"] == []
    b = load_oracle_samples(tmp_path / "b.json")[0]
    assert b["top_5_block_ids"] == [0] and b["id"] == "h1"


def test_hotpot_flatten_spans_point_at_supporting_sentences():
    from scripts.generate_oracle_hotpot import flatten_hotpot

    item = {
        "context": {"title": ["A", "B"], "sentences": [["A one.", "A two."], ["B one.", "B two."]]},
        "supporting_facts": {"title": ["A", "B"], "sent_id": [1, 0]},
    }
    text, sents, paras = flatten_hotpot(item)
    assert [text[s:e] for s, e in sents] == ["A two.", "B one."]
    assert len(paras) == 2 and text[paras[1][0]:paras[1][1]].startswith("Title: B")


def test_answer_block_ids_normalises():
    from maplecore.indexer import Block
    from scripts.oracle_common import answer_block_ids

    blocks = [Block(0, "He went to The Tower of London."), Block(1, "Nothing here.")]
    assert answer_block_ids(blocks, ["the tower of london"]) == [0]
    assert answer_block_ids(blocks, ["a"]) == []  # too short after normalisation
