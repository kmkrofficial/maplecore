import importlib

import pytest

eff = importlib.import_module("benchmarks.11_efficiency")


class SlowRanker:
    name = "slow"

    def __init__(self, seconds):
        self.seconds = seconds

    def rank(self, question, texts, q_emb, block_embs, k=None, sample=None):
        import time
        time.sleep(self.seconds)
        return list(range(len(texts)))


def test_timer_reports_percentiles():
    t = eff.time_calls(lambda: None, n=50, sync=False)
    assert set(t) == {"p50", "p95"} and 0 <= t["p50"] <= t["p95"]


def test_infeasible_ranker_is_skipped_at_larger_pools():
    out = eff.run_one(SlowRanker(0.2), pool_sizes=[10, 100], device="cpu", n_queries=3, max_seconds_per_query=0.1)
    assert out["10"] == "infeasible" and "100" not in out


def test_feasible_ranker_reports_all_pools():
    out = eff.run_one(SlowRanker(0.0), pool_sizes=[10, 20], device="cpu", n_queries=3, max_seconds_per_query=5.0)
    assert set(out) == {"10", "20"} and {"total_ms", "qps"} <= set(out["10"])
    assert out["10"]["total_ms"]["p50"] >= 0 and out["10"]["qps"] > 0


def test_pareto_frontier():
    pts = {"a": (1.0, 0.5), "b": (2.0, 0.9), "c": (3.0, 0.8)}   # name -> (latency, quality)
    assert eff.pareto_frontier(pts) == ["a", "b"]


def test_pareto_frontier_ties_and_single_point():
    assert eff.pareto_frontier({"only": (1.0, 0.1)}) == ["only"]
    assert eff.pareto_frontier({"a": (1.0, 0.5), "b": (1.0, 0.5)}) == ["a", "b"]
