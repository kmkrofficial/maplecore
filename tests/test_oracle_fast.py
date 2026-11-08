"""The fast (SDPA + last-row) oracle path must match the eager reference."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from scripts.oracle_common import (
    LastRowAttentionRecorder,
    LastRowCapture,
    block_attention,
    score_batch,
)


def _tiny_llama(attn: str, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=101, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      pad_token_id=0)
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).eval()


def _items(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    items = []
    for L in lengths:
        ids = torch.randint(1, 101, (1, L), generator=g)
        ranges = [(b, s, min(s + 7, L - 3)) for b, s in enumerate(range(2, L - 10, 7))]
        items.append((ids, ranges))
    return items


def _eager_scores(items):
    eager = _tiny_llama("eager")
    rec = LastRowAttentionRecorder(eager)
    out = [block_attention(eager, rec, ids, ranges) for ids, ranges in items]
    rec.remove()
    return out


def _assert_close(a, b, tol=1e-5):
    assert a.keys() == b.keys()
    for k in a:
        assert abs(a[k] - b[k]) < tol, (k, a[k], b[k])


def test_fast_matches_eager_unbatched():
    items = _items([40, 63])
    ref = _eager_scores(items)
    fast = _tiny_llama("sdpa")
    for item, (r_last, r_all) in zip(items, ref):
        (f_last, f_all), = score_batch(fast, [item], pad_token_id=0)
        _assert_close(f_last, r_last)
        _assert_close(f_all, r_all)


def test_fast_batched_with_padding_matches_eager():
    items = _items([30, 64, 47])  # different lengths -> left padding inside the batch
    ref = _eager_scores(items)
    got = score_batch(_tiny_llama("sdpa"), items, pad_token_id=0)
    for (f_last, f_all), (r_last, r_all) in zip(got, ref):
        _assert_close(f_last, r_last)
        _assert_close(f_all, r_all)


def test_capture_restores_sdpa_function():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    before = ALL_ATTENTION_FUNCTIONS["sdpa"]
    with LastRowCapture():
        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is not before
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is before


# ------------------------------------------------------------------ per-head capture (Task 7)

def test_head_mass_reduces_to_existing_scores():
    items = _items([40, 63])
    last, all_, heads = score_batch(_tiny_llama("sdpa"), items, 0, keep_heads=True)[0]
    assert heads.shape == (3, 4, len(items[0][1])) and heads.dtype == torch.float16  # layers, heads, blocks
    for i, (bid, _, _) in enumerate(items[0][1]):
        assert abs(float(heads[-1, :, i].float().mean()) - last[bid]) < 1e-3
        assert abs(float(heads[:, :, i].float().mean()) - all_[bid]) < 1e-3


def test_head_mass_matches_eager_per_head():
    ids, ranges = _items([40])[0]
    eager = _tiny_llama("eager")
    with torch.no_grad():
        att = eager(ids, output_attentions=True, use_cache=False).attentions  # layers x [1, H, L, L]
    ref = torch.stack([torch.stack([a[0, :, -1, s:e].sum(-1) for _, s, e in ranges], dim=1) for a in att])
    got = score_batch(_tiny_llama("sdpa"), [(ids, ranges)], 0, keep_heads=True)[0][2]
    assert torch.allclose(got.float(), ref, atol=1e-3)


def test_head_mass_batched_with_padding_matches_unbatched():
    items = _items([30, 64, 47])
    model = _tiny_llama("sdpa")
    batched = score_batch(model, items, 0, keep_heads=True)
    for item, (_, _, h_b) in zip(items, batched):
        _, _, h_1 = score_batch(model, [item], 0, keep_heads=True)[0]
        assert torch.allclose(h_b.float(), h_1.float(), atol=1e-3)


def test_default_score_batch_unchanged_without_heads():
    assert len(score_batch(_tiny_llama("sdpa"), _items([40]), 0)[0]) == 2


class _FakeTok:
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, add_special_tokens=True):
        return [(ord(c) % 100) + 1 for c in text]


def _fake_hotpot(n):
    rows = []
    for i in range(n):
        rows.append({
            "id": f"h{i}", "type": "bridge", "level": "easy", "question": f"who is {i}?", "answer": "x",
            "context": {"title": [f"A{i}", f"B{i}"],
                        "sentences": [[f"Alpha {i} is a city in the far north of the country, beside a wide river."],
                                      [f"Beta {i} is a long river that runs near the old city walls."]]},
            "supporting_facts": {"title": [f"A{i}"], "sent_id": [0]},
        })
    return rows


def test_heads_file_resume_alignment(tmp_path, monkeypatch):
    """--save-heads + --resume must keep <stem>.heads.pt aligned with the JSON samples (ids and block order)."""
    import datasets
    from scripts import generate_oracle_hotpot as gen
    from scripts.oracle_common import load_oracle_samples

    model = _tiny_llama("sdpa")
    monkeypatch.setattr(gen, "load_teacher", lambda *a, **k: (model, _FakeTok()))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _fake_hotpot(8))
    monkeypatch.setattr(gen, "CHUNK_SIZE", 20)
    out = tmp_path / "o.json"

    gen.run(split="validation", max_samples=3, max_tokens=400, output=out, batch_size=2, save_heads=True)
    gen.run(split="validation", max_samples=6, max_tokens=400, output=out, batch_size=2, save_heads=True, resume=True)

    samples = load_oracle_samples(out)
    heads = torch.load(out.with_suffix(".heads.pt"), weights_only=True)
    assert [s["id"] for s in samples] == [f"h{i}" for i in range(6)]
    assert list(heads["mass"]) == [s["id"] for s in samples]
    for s in samples:
        assert heads["block_order"][s["id"]] == [b["id"] for b in s["blocks"]]
        assert heads["mass"][s["id"]].shape == (3, 4, len(s["blocks"]))


def test_save_heads_requires_fast_path(tmp_path, monkeypatch):
    import datasets
    from scripts import generate_oracle_hotpot as gen

    monkeypatch.setattr(gen, "load_teacher", lambda *a, **k: (_tiny_llama("eager"), _FakeTok()))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _fake_hotpot(2))
    with pytest.raises(ValueError, match="fast"):
        gen.run(split="validation", max_samples=1, output=tmp_path / "o.json", attn_impl="eager", save_heads=True)
