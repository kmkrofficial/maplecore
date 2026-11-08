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
