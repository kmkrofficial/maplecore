import importlib

import pytest
import torch

from benchmarks.rankers import (BM25Ranker, CrossEncoderRanker, DenseRanker, HybridRRFRanker, ModelRanker,
                                TeacherRanker, build_rankers)
from maplecore import MapleResidualNet, save_model


def hf_cached(name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:
        return False


def test_rrf_fuses_rankings():
    assert HybridRRFRanker(rrf_k=60)._fuse([[0, 1, 2], [2, 0, 1]]) == [0, 2, 1]


def test_bm25_exact_term_first():
    texts = ["a cat", "the zebra has stripes", "dogs"]
    assert BM25Ranker().rank("zebra stripes", texts, None, None)[0] == 1


def test_dense_ranks_by_cosine():
    q = torch.tensor([1.0, 0.0])
    b = torch.tensor([[0.0, 1.0], [1.0, 0.1], [-1.0, 0.0]])
    assert DenseRanker().rank("", ["x"] * 3, q, b) == [1, 0, 2]


def test_model_ranker_untrained_residual_equals_dense():
    q, b = torch.randn(16), torch.randn(30, 16)
    model = MapleResidualNet(16).eval()
    assert ModelRanker(model).rank("", [""] * 30, q, b) == DenseRanker().rank("", [""] * 30, q, b)


def test_k_truncates_and_small_pool():
    assert len(DenseRanker().rank("", ["x", "y"], torch.randn(4), torch.randn(2, 4), k=5)) == 2
    assert len(DenseRanker().rank("", ["x"] * 6, torch.randn(4), torch.randn(6, 4), k=3)) == 3


def test_hybrid_combines_lexical_and_dense_evidence():
    texts = ["zebra stripes pattern", "unrelated words here", "another zebra note"]
    q = torch.tensor([1.0, 0.0])
    b = torch.tensor([[0.2, 1.0], [0.9, 0.1], [0.3, 0.9]])      # dense prefers block 1, BM25 prefers 0/2
    ranking = HybridRRFRanker().rank("zebra", texts, q, b)
    assert sorted(ranking) == [0, 1, 2] and ranking[0] in (0, 1)


def test_teacher_ranker_uses_sample_scores():
    sample = {"blocks": [{"id": 0}, {"id": 1}, {"id": 2}], "attention_scores": {"0": 0.2, "1": 0.8, "2": 0.5}}
    r = TeacherRanker("attention_scores", "oracle_last")
    assert r.name == "oracle_last" and r.rank("", ["a", "b", "c"], None, None, sample=sample) == [1, 2, 0]


def test_build_rankers_names_and_unknown_spec(tmp_path):
    save_model(MapleResidualNet(8), tmp_path / "m.pt")
    rankers = build_rankers(["bm25", "dense", "hybrid", "maple:v2"], {"v2": tmp_path / "m.pt"})
    assert [r.name for r in rankers] == ["bm25", "dense", "hybrid", "maple:v2"]
    with pytest.raises(ValueError, match="nope"):
        build_rankers(["nope"], {})
    with pytest.raises(ValueError, match="model_paths"):
        build_rankers(["maple:missing"], {})


def test_cross_encoder_is_lazy():
    r = CrossEncoderRanker("some/model-that-is-not-downloaded")   # constructing must not load weights
    assert r.name == "cross_encoder"


@pytest.mark.skipif(not hf_cached("BAAI/bge-reranker-base"), reason="model not cached")
def test_cross_encoder_ranks_relevant_first():
    # Distractors must not be about Paris: bge-reranker-base scores "The Seine flows through Paris." (logit 8.1)
    # above "Paris is the capital of France." (6.2) for this question, so that pair is not a usable assertion.
    texts = ["Paris is the capital of France.", "Bananas are yellow.", "Photosynthesis takes place in leaves."]
    assert CrossEncoderRanker().rank("What is the capital of France?", texts, None, None)[0] == 0


def test_bm25_ignores_punctuation_and_case():
    texts = ["Bread is baked from flour.", "We saw a ZEBRA.", "Rain fell over the harbour."]
    assert BM25Ranker().rank("zebra", texts, None, None)[0] == 1


def test_bm25_fitted_on_corpus_ranks_candidates_by_corpus_level_scores():
    import bm25s
    import numpy as np

    corpus = ["alpha beta gamma", "alpha alpha delta", "alpha epsilon", "beta zeta eta", "theta iota",
              "kappa lambda", "mu nu alpha", "xi omicron"]
    ranker = BM25Ranker()
    ranker.fit(corpus)
    candidates = [corpus[2], corpus[0], corpus[1]]  # every candidate contains 'alpha'
    order = ranker.rank("alpha beta", candidates, None, None)

    ref = bm25s.BM25()
    ref.index(bm25s.tokenize(corpus, stopwords="en", show_progress=False), show_progress=False)
    full = ref.get_scores(["alpha", "beta"])
    expected = list(np.argsort(-full[[2, 0, 1]], kind="stable"))
    assert order == expected and order[0] == 1  # 'alpha beta gamma' wins on corpus statistics


def test_bm25_fitted_falls_back_to_local_scoring_for_unseen_texts():
    ranker = BM25Ranker()
    ranker.fit(["alpha beta", "gamma delta"])
    assert ranker.rank("zebra", ["no match here", "a zebra appears"], None, None)[0] == 1


def test_fit_rankers_fits_bm25_and_the_bm25_inside_hybrid():
    from benchmarks.rankers import fit_rankers

    bm25, hybrid, dense = BM25Ranker(), HybridRRFRanker(), DenseRanker()
    fit_rankers([bm25, hybrid, dense], ["alpha beta", "gamma delta"])
    assert bm25._index is not None and hybrid._bm25._index is not None
