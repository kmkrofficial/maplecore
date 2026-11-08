import math

from benchmarks import metrics as M


def test_recall_variants():
    assert M.recall_at_k([1, 2, 3], {2, 9}, k=2) == 0.5
    assert M.recall_at_k([1, 2, 3], {1, 2, 3, 4, 5}, k=1, capped=True) == 1.0
    assert M.recall_at_k([1], set(), k=1) is None


def test_hit_and_mrr():
    assert M.hit_all_at_k([1, 2, 3], {1, 3}, 3) == 1.0
    assert M.hit_all_at_k([1, 2, 3], {1, 4}, 3) == 0.0
    assert M.hit_any_at_k([1, 2], {2}, 2) == 1.0
    assert M.mrr_at_k([5, 6, 7], {7}, 10) == 1 / 3
    assert M.mrr_at_k([5, 6, 7], {8}, 10) == 0.0


def test_ndcg_perfect_and_reversed():
    rel = {0: 3.0, 1: 2.0, 2: 1.0}
    assert math.isclose(M.ndcg_at_k([0, 1, 2], rel, 3), 1.0)
    assert M.ndcg_at_k([2, 1, 0], rel, 3) < 1.0


def test_answer_metrics():
    assert M.answer_in_context("It was in the Tower of London.", ["tower of london"]) == 1.0
    assert M.exact_match("The Eiffel Tower", ["eiffel tower"]) == 1.0
    assert M.token_f1("Paris France", ["Paris"]) == 2 * 0.5 * 1 / 1.5


def test_bootstrap_and_paired():
    mean, lo, hi = M.bootstrap_ci([0, 1] * 50)
    assert mean == 0.5 and lo < 0.5 < hi
    assert M.summarize([1.0, None, 0.0])["n"] == 2
    assert M.paired_bootstrap_pvalue([1.0] * 30, [0.0] * 30) < 0.01
    assert M.paired_bootstrap_pvalue([1, 0] * 20, [0, 1] * 20) > 0.5
