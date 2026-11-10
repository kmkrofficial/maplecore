import importlib

import pytest

mh = importlib.import_module("benchmarks.09_multihop_rerank")


def _q(qid, paras, gold):
    return {"id": qid, "question": f"question {qid}?", "paragraphs": [{"title": t, "text": f"text of {t}"} for t in paras],
            "gold_titles": gold}


def test_corpus_is_union_of_question_paragraphs():
    qs = [_q("q1", ["A", "B"], ["A"]), _q("q2", ["B", "C"], ["C"])]
    corpus = mh.build_corpus(qs)
    assert sorted(p["title"] for p in corpus) == ["A", "B", "C"]          # deduplicated by title + text


def test_metrics_on_known_rerank():
    out = mh.score_reranked(gold={"A", "C"}, ranking=["A", "B", "C", "D"])
    assert out == {"recall@2": 0.5, "recall@5": 1.0, "all_recall@5": 1.0}


def test_all_recall_requires_every_gold():
    assert mh.score_reranked(gold={"A", "Z"}, ranking=["A", "B", "C"])["all_recall@5"] == 0.0


def test_sanity_flag():
    assert mh.first_stage_sanity({"hotpotqa": 0.80}, used="colbertv2")["hotpotqa"]["ok"] is True
    assert mh.first_stage_sanity({"hotpotqa": 0.80}, used="dense")["hotpotqa"]["ok"] is False
    assert mh.first_stage_sanity({"hotpotqa": 0.50}, used="colbertv2")["hotpotqa"]["ok"] is False


def test_seeded_question_sample():
    qs = [_q(f"q{i}", ["A"], ["A"]) for i in range(50)]
    a = mh.sample_questions(qs, 10, seed=3)
    b = mh.sample_questions(qs, 10, seed=3)
    assert [q["id"] for q in a] == [q["id"] for q in b] and len(a) == 10
