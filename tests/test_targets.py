import pytest
import torch

from scripts.targets import block_target


def _sample(blocks, all_layers=None, last=None, top5=None, gold=None):
    s = {"id": "s", "question": "q", "blocks": [{"id": b, "text": f"b{b}"} for b in blocks],
         "top_5_block_ids": list(top5 if top5 is not None else blocks[:1]),
         "gold_block_ids": list(gold or [])}
    if all_layers is not None:
        s["attention_scores_all_layers"] = all_layers
    if last is not None:
        s["attention_scores"] = last
    return s


def test_mass_target_normalised_and_ordered():
    s = _sample(blocks=[7, 3, 9], all_layers={"7": 0.2, "3": 0.6, "9": 0.2})
    t = block_target(s, "all_layers")
    assert torch.allclose(t, torch.tensor([0.2, 0.6, 0.2]))
    assert t.dtype == torch.float32


def test_mass_target_renormalises_partial_mass():
    s = _sample(blocks=[0, 1], all_layers={"0": 0.02, "1": 0.06})  # teacher mass on context is only a few percent
    assert torch.allclose(block_target(s, "all_layers"), torch.tensor([0.25, 0.75]))


def test_last_layer_source_uses_attention_scores():
    s = _sample(blocks=[0, 1], last={"0": 0.9, "1": 0.1}, all_layers={"0": 0.1, "1": 0.9})
    assert block_target(s, "last").argmax() == 0 and block_target(s, "all_layers").argmax() == 1


def test_temperature_sharpens():
    s = _sample(blocks=[0, 1], all_layers={"0": 0.4, "1": 0.6})
    assert block_target(s, "all_layers", temperature=0.5)[1] > 0.6


def test_top5_and_gold_uniform():
    s = _sample(blocks=list(range(8)), top5=[1, 2, 3, 4, 5], gold=[2, 6])
    assert torch.allclose(block_target(s, "top5")[[1, 2, 3, 4, 5]], torch.full((5,), 0.2))
    assert block_target(s, "gold")[6] == 0.5


def test_degenerate_targets_return_none():
    assert block_target(_sample(blocks=[0, 1], gold=[]), "gold") is None
    assert block_target(_sample(blocks=[0, 1], all_layers={"0": 0.0, "1": 0.0}), "all_layers") is None


def test_missing_score_field_returns_none():
    assert block_target(_sample(blocks=[0, 1]), "all_layers") is None  # e.g. a v1 file without all-layer scores


def test_ids_outside_blocks_are_ignored():
    s = _sample(blocks=[0, 1], top5=[1, 99], gold=[99])
    assert torch.allclose(block_target(s, "top5"), torch.tensor([0.0, 1.0]))
    assert block_target(s, "gold") is None


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        block_target(_sample(blocks=[0]), "nope")
