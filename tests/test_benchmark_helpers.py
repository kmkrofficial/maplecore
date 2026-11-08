import importlib
import random

import pytest

pytest.importorskip("rank_bm25")


def test_needle_insert_span_and_heldout_templates():
    needle_mod = importlib.import_module("benchmarks.03_needle_robust")
    from scripts.train_generalist import TRAIN_NEEDLE_TEMPLATES

    hay = "One sentence here. Another sentence there. " * 50
    for depth in (0, 37, 100):
        text, (s, e) = needle_mod._insert(hay, "NEEDLE TEXT.", depth)
        assert text[s:e] == "NEEDLE TEXT."

    train_qs = {q for _, q in TRAIN_NEEDLE_TEMPLATES}
    train_ns = {t.split("{")[0] for t, _ in TRAIN_NEEDLE_TEMPLATES}
    for t, q in needle_mod.HELDOUT_TEMPLATES:
        assert q not in train_qs
        assert t.split("{")[0] not in train_ns
    needle, question = needle_mod._needle(random.Random(0))
    assert "{" not in needle and question.endswith("?")


def test_long_context_paragraph_blocks_mark_supporting_sentences():
    lc = importlib.import_module("benchmarks.07_long_context")
    item = {
        "context": {"title": ["Alpha", "Beta"],
                    "sentences": [["Alpha is a city.", " It has a river."], ["Beta is a dog."]]},
        "supporting_facts": {"title": ["Alpha"], "sent_id": [1]},
    }
    blocks = lc._paragraph_blocks(item)
    assert [b["gold"] for b in blocks] == [True, False]
    assert blocks[0]["text"].startswith("Title: Alpha")
