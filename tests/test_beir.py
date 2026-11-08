import importlib

import pytest

pytest.importorskip("bm25s")

beir = importlib.import_module("benchmarks.08_beir_rerank")


class OracleQrelRanker:
    """Test ranker: puts the documents with the highest qrel grade first (needs the query text -> grades map)."""

    name = "oracle_qrels"

    def __init__(self, grades_by_question):
        self.grades = grades_by_question

    def rank(self, question, texts, q_emb, block_embs, k=None, sample=None):
        grades = self.grades[question]
        order = sorted(range(len(texts)), key=lambda i: -grades.get(texts[i], 0))
        return order[:k] if k else order


class ReverseRanker:
    name = "reverse"

    def rank(self, question, texts, q_emb, block_embs, k=None, sample=None):
        return list(range(len(texts)))[::-1][:k] if k else list(range(len(texts)))[::-1]


def _toy_beir():
    corpus = {f"d{i}": {"title": "", "text": f"document number {i} about topic{i % 4} words"} for i in range(20)}
    queries = {"q0": "topic0 words", "q1": "topic1 words", "q2": "topic2 words"}
    qrels = {"q0": {"d0": 2, "d4": 1}, "q1": {"d1": 1}, "q2": {"d2": 2, "d6": 1, "d10": 1}}
    return corpus, queries, qrels


def test_beir_pipeline_perfect_reranker_scores_one():
    corpus, queries, qrels = _toy_beir()
    grades = {queries[q]: {corpus[d]["text"]: g for d, g in rel.items()} for q, rel in qrels.items()}
    out = beir.evaluate_rerank(corpus, queries, qrels, rankers=[OracleQrelRanker(grades)], first_stage_k=20)
    assert out["oracle_qrels"]["ndcg@10"]["mean"] == pytest.approx(1.0)
    assert out["oracle_qrels"]["ndcg@10"]["n"] == 3


def test_reversed_ranking_scores_lower_than_oracle():
    corpus, queries, qrels = _toy_beir()
    grades = {queries[q]: {corpus[d]["text"]: g for d, g in rel.items()} for q, rel in qrels.items()}
    out = beir.evaluate_rerank(corpus, queries, qrels, rankers=[OracleQrelRanker(grades), ReverseRanker()],
                               first_stage_k=20)
    assert out["reverse"]["ndcg@10"]["mean"] < out["oracle_qrels"]["ndcg@10"]["mean"]


def test_first_stage_limits_candidates():
    corpus, queries, qrels = _toy_beir()
    grades = {queries[q]: {corpus[d]["text"]: g for d, g in rel.items()} for q, rel in qrels.items()}
    out = beir.evaluate_rerank(corpus, queries, qrels, rankers=[OracleQrelRanker(grades)], first_stage_k=3)
    assert out["oracle_qrels"]["ndcg@10"]["mean"] <= 1.0
    assert out["_first_stage"]["bm25"]["ndcg@10"]["n"] == 3          # BM25 order itself is reported as a row


def test_queries_without_qrels_are_skipped():
    corpus, queries, qrels = _toy_beir()
    qrels = {"q0": qrels["q0"]}
    out = beir.evaluate_rerank(corpus, queries, qrels, rankers=[ReverseRanker()], first_stage_k=20)
    assert out["reverse"]["ndcg@10"]["n"] == 1


def test_max_queries_is_seeded_subsample():
    corpus, queries, qrels = _toy_beir()
    a = beir.evaluate_rerank(corpus, queries, qrels, rankers=[ReverseRanker()], first_stage_k=20, max_queries=2, seed=1)
    b = beir.evaluate_rerank(corpus, queries, qrels, rankers=[ReverseRanker()], first_stage_k=20, max_queries=2, seed=1)
    assert a["reverse"]["ndcg@10"]["n"] == 2 and a["reverse"] == b["reverse"]


def test_bm25_sanity_flag():
    res = beir.bm25_sanity({"scifact": 0.60, "nfcorpus": 0.322})
    assert res["scifact"] == {"ours": 60.0, "published": 67.9, "ok": False}
    assert res["nfcorpus"]["ok"] is True
    assert beir.bm25_sanity({"unknown": 0.5}) == {}
