import importlib
import json
from pathlib import Path

import pytest

lb = importlib.import_module("benchmarks.10_e2e_longbench")


def test_middle_truncation_keeps_head_and_tail():
    ids = list(range(10_000))
    t = lb.middle_truncate(ids, 7500)
    assert len(t) == 7500 and t[:3750] == ids[:3750] and t[-3750:] == ids[-3750:]
    assert lb.middle_truncate(list(range(10)), 7500) == list(range(10))


def test_budgeted_context_restores_document_order():
    blocks = ["b0", "b1", "b2", "b3"]
    assert lb.budget_context(blocks, ranking=[3, 1, 0], budget_tokens=2, count_tokens=lambda s: 1) == "b1\nb3"


def test_budgeted_context_always_keeps_best_block_even_if_over_budget():
    assert lb.budget_context(["a" * 100, "b"], ranking=[0, 1], budget_tokens=1, count_tokens=len) == "a" * 100


def test_narrativeqa_metrics_perfect_answer():
    m = lb.nqa_generation_metrics(["the red fox"], [["the red fox"]])
    assert m["rouge_l"] == pytest.approx(100.0) and m["bleu_1"] == pytest.approx(100.0)


def test_narrativeqa_metrics_wrong_answer_lower():
    good = lb.nqa_generation_metrics(["the red fox"], [["the red fox"]])
    bad = lb.nqa_generation_metrics(["a blue whale"], [["the red fox"]])
    assert bad["rouge_l"] < good["rouge_l"] and bad["bleu_1"] < good["bleu_1"]


def test_longbench_prompts_file_has_source():
    p = json.loads(Path("benchmarks/longbench_prompts.json").read_text(encoding="utf-8"))
    assert {"hotpotqa", "narrativeqa"} <= set(p["dataset2prompt"]) and p["_source"]["url"].startswith("https://")
    assert set(p["dataset2maxlen"]) >= {"hotpotqa", "narrativeqa"}


def test_chunk_words_splits_story_into_100_word_chunks():
    story = " ".join(f"w{i}" for i in range(250))
    chunks = lb.chunk_words(story, 100)
    assert [len(c.split()) for c in chunks] == [100, 100, 50]
