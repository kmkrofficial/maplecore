import copy
import math

import pytest
import torch

from maplecore import MapleResidualNet
from maplecore.listwise import ListwiseTrainer, QueryGroup, listwise_kl_loss


def _toy_groups(n: int, seed: int, dim: int = 8, blocks: int = 10, with_gold: bool = False):
    """Target favours blocks along a fixed direction w, independent of the query -
    something plain cosine similarity cannot express."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, generator=torch.Generator().manual_seed(123))
    w = w / w.norm()
    out = []
    for i in range(n):
        q = torch.randn(dim, generator=g)
        b = torch.randn(blocks, dim, generator=g)
        target = torch.softmax(4.0 * (b @ w), dim=0)
        gold = None
        if with_gold:
            gold = torch.zeros(blocks, dtype=torch.bool)
            gold[target.argmax()] = True
        out.append(QueryGroup(f"g{i}", q, b, target, gold))
    return out


def test_kl_zero_at_target():
    t = torch.tensor([[0.7, 0.2, 0.1]])
    assert listwise_kl_loss(t.log(), t, torch.ones_like(t, dtype=torch.bool)).item() < 1e-6


def test_kl_positive_when_wrong():
    t = torch.tensor([[0.9, 0.05, 0.05]])
    wrong = torch.tensor([[0.0, 5.0, 0.0]])
    assert listwise_kl_loss(wrong, t, torch.ones_like(t, dtype=torch.bool)).item() > 1.0


def test_padding_ignored():
    t = torch.tensor([[0.5, 0.5, 0.0]])
    m = torch.tensor([[True, True, False]])
    a = listwise_kl_loss(torch.tensor([[1.0, 2.0, 0.0]]), t, m)
    b = listwise_kl_loss(torch.tensor([[1.0, 2.0, 99.0]]), t, m)
    assert torch.isfinite(a) and torch.allclose(a, b)


def test_loss_finite_with_skipped_groups():
    g = _toy_groups(n=10, seed=0)
    g.append(QueryGroup("single", g[0].q, g[0].blocks[:1], torch.tensor([1.0])))  # one-block group
    torch.manual_seed(0)
    out = ListwiseTrainer(MapleResidualNet(dim=8), device="cpu").fit(g, g[:3], epochs=2)
    assert all(math.isfinite(h["train_loss"]) for h in out["history"])


def test_fit_beats_cosine_init_on_learnable_task():
    train, val = _toy_groups(200, seed=1), _toy_groups(50, seed=2)
    torch.manual_seed(0)
    tr = ListwiseTrainer(MapleResidualNet(dim=8), device="cpu", seed=0)
    before = tr.evaluate(val)["val_target_recall@5"]
    out = tr.fit(train, val, epochs=15)
    assert out["metric"] == "val_target_recall@5" and out["best_val"] > before + 0.1


def test_selection_metric_is_evidence_when_all_groups_have_gold():
    train, val = _toy_groups(40, seed=1, with_gold=True), _toy_groups(10, seed=2, with_gold=True)
    torch.manual_seed(0)
    out = ListwiseTrainer(MapleResidualNet(dim=8), device="cpu").fit(train, val, epochs=2)
    assert out["metric"] == "val_evidence_recall@5"


def test_restores_best_epoch(monkeypatch):
    train, val = _toy_groups(40, seed=1), _toy_groups(10, seed=2)
    torch.manual_seed(0)
    tr = ListwiseTrainer(MapleResidualNet(dim=8), device="cpu", seed=0)
    snaps, scores = [], iter([0.9, 0.1, 0.1])

    def fake_evaluate(groups, k=5):
        snaps.append(copy.deepcopy(tr.model.state_dict()))
        v = next(scores)
        return {"val_target_recall@5": v, "val_evidence_recall@5": v}

    monkeypatch.setattr(tr, "evaluate", fake_evaluate)
    out = tr.fit(train, val, epochs=3)
    assert out["best_epoch"] == 1 and out["best_val"] == 0.9
    for k, v in tr.model.state_dict().items():
        assert torch.equal(v, snaps[0][k]), k
    assert any(not torch.equal(snaps[0][k], snaps[-1][k]) for k in snaps[0])  # later epochs really differed


def test_early_stopping_respects_patience(monkeypatch):
    train, val = _toy_groups(20, seed=1), _toy_groups(5, seed=2)
    torch.manual_seed(0)
    tr = ListwiseTrainer(MapleResidualNet(dim=8), device="cpu")
    monkeypatch.setattr(tr, "evaluate", lambda groups, k=5: {"val_target_recall@5": 0.5, "val_evidence_recall@5": 0.5})
    out = tr.fit(train, val, epochs=20, patience=2)
    assert len(out["history"]) == 3  # best at epoch 1, then two epochs without improvement


def test_deterministic_given_seed():
    train, val = _toy_groups(60, seed=1), _toy_groups(20, seed=2)

    def once():
        torch.manual_seed(3)
        return ListwiseTrainer(MapleResidualNet(dim=8), device="cpu", seed=3).fit(train, val, epochs=3)["best_val"]

    assert once() == once()
