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


class _FakeTok:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return "answer"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return msgs[0]["content"]

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        import torch
        n = max(len(t.split()) for t in texts)
        return {"input_ids": torch.ones(len(texts), n, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), n, dtype=torch.long)}


class _KernelRecordingModel:
    """Stands in for the 8B reader: records which SDPA kernels are allowed while generate() runs."""
    device = "cpu"

    def __init__(self):
        self.math_enabled, self.cudnn_enabled = [], []

    def generate(self, input_ids, attention_mask, max_new_tokens, **kw):
        import torch
        self.math_enabled.append(torch.backends.cuda.math_sdp_enabled())
        self.cudnn_enabled.append(torch.backends.cuda.cudnn_sdp_enabled())
        return torch.cat([input_ids, torch.zeros(input_ids.shape[0], max_new_tokens, dtype=torch.long)], dim=1)


def test_reader_generates_without_the_quadratic_math_attention_kernel():
    import torch
    reader = lb.Reader.__new__(lb.Reader)
    reader.model, reader.tok, reader.eos = _KernelRecordingModel(), _FakeTok(), [0]
    out = reader.generate(["a long prompt " * 50, "short"], max_new_tokens=3)
    assert out == ["answer", "answer"]
    assert reader.model.math_enabled and not any(reader.model.math_enabled)
    # Llama-3 uses grouped-query attention (enable_gqa=True): with a mask only cuDNN fuses that (torch 2.11)
    assert all(reader.model.cudnn_enabled)
    assert torch.backends.cuda.math_sdp_enabled()  # restored afterwards
