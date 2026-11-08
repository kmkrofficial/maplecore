"""
Rankers: one interface for every retrieval method compared in the benchmarks
===========================================================================
``rank(question, texts, q_emb, block_embs, k=None, sample=None) -> indices into texts, best first``

* ``bm25``            lexical (rank_bm25, lower-cased whitespace tokens)
* ``dense``           cosine similarity of the supplied embeddings, or of embeddings from its own
                      encoder (``dense:<encoder name or path>``, e.g. an attention-distilled MAPLE-Embed)
* ``hybrid``          reciprocal-rank fusion of bm25 and dense
* ``cross_encoder``   a re-ranker model scoring (question, text) pairs; loaded lazily
* ``maple:<key>``     a trained MAPLE scorer from ``model_paths[key]``
* ``TeacherRanker``   reference rows: rank by a score the oracle file stored per block (not a retriever)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Ranker(Protocol):
    name: str

    def rank(self, question: str, texts: Sequence[str], q_emb: Optional[torch.Tensor],
             block_embs: Optional[torch.Tensor], k: Optional[int] = None, sample: Optional[dict] = None) -> List[int]: ...


def _top(order: Sequence[int], k: Optional[int]) -> List[int]:
    order = [int(i) for i in order]
    return order[:k] if k else order


class BM25Ranker:
    name = "bm25"

    def rank(self, question, texts, q_emb=None, block_embs=None, k=None, sample=None) -> List[int]:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        scores = bm25.get_scores(question.lower().split())
        return _top(np.argsort(-scores, kind="stable"), k)


class DenseRanker:
    """Cosine similarity. With ``encoder`` set it embeds question and texts itself instead of using the
    supplied embeddings (the question gets the encoder's query prefix when it has one)."""

    def __init__(self, encoder: Optional[str] = None, name: Optional[str] = None):
        self.encoder = encoder
        self.name = name or ("dense" if encoder is None else f"dense:{encoder}")
        self._indexer = None

    def _embed(self, question, texts):
        if self._indexer is None:
            from maplecore import MapleIndexer
            self._indexer = MapleIndexer(model_name=self.encoder)
        return self._indexer.encode_query(question), self._indexer.encode_texts(list(texts))

    def rank(self, question, texts, q_emb=None, block_embs=None, k=None, sample=None) -> List[int]:
        if self.encoder is not None:
            q_emb, block_embs = self._embed(question, texts)
        if q_emb is None or block_embs is None:
            raise ValueError("DenseRanker needs embeddings (or an encoder)")
        sims = F.normalize(block_embs.float(), dim=1) @ F.normalize(q_emb.float().to(block_embs.device), dim=0)
        return _top(torch.argsort(sims, descending=True).tolist(), k)


class HybridRRFRanker:
    """Reciprocal-rank fusion of BM25 and dense rankings: score(i) = sum over rankers 1 / (rrf_k + rank_i)."""

    def __init__(self, rrf_k: int = 60, dense: Optional[DenseRanker] = None):
        self.name = "hybrid"
        self.rrf_k = rrf_k
        self._bm25, self._dense = BM25Ranker(), dense or DenseRanker()

    def _fuse(self, rankings: Sequence[Sequence[int]]) -> List[int]:
        scores: Dict[int, float] = {}
        for ranking in rankings:
            for r, i in enumerate(ranking):
                scores[i] = scores.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
        return sorted(scores, key=lambda i: (-scores[i], i))

    def rank(self, question, texts, q_emb=None, block_embs=None, k=None, sample=None) -> List[int]:
        fused = self._fuse([self._bm25.rank(question, texts), self._dense.rank(question, texts, q_emb, block_embs)])
        return _top(fused, k)


class CrossEncoderRanker:
    """Cross-encoder re-ranker (default BAAI/bge-reranker-base). Weights load on first use."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 32,
                 device: Optional[str] = None, name: str = "cross_encoder"):
        self.model_name, self.batch_size, self.device, self.name = model_name, batch_size, device, name
        self._model = None

    def rank(self, question, texts, q_emb=None, block_embs=None, k=None, sample=None) -> List[int]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from maplecore.utils import resolve_device
            self._model = CrossEncoder(self.model_name, device=resolve_device(self.device))
        scores = self._model.predict([(question, t) for t in texts], batch_size=self.batch_size,
                                     show_progress_bar=False)
        return _top(np.argsort(-np.asarray(scores), kind="stable"), k)


class ModelRanker:
    """A trained MAPLE scorer (any module taking [B, 2*dim] -> [B] logits) applied linearly to all texts."""

    def __init__(self, model: torch.nn.Module, name: str = "maple"):
        self.model, self.name = model.eval(), name

    @torch.no_grad()
    def rank(self, question, texts, q_emb=None, block_embs=None, k=None, sample=None) -> List[int]:
        if q_emb is None or block_embs is None:
            raise ValueError("ModelRanker needs embeddings")
        device = next(self.model.parameters()).device
        x = torch.cat([q_emb.float().to(device).unsqueeze(0).expand(len(texts), -1), block_embs.float().to(device)], dim=1)
        return _top(torch.argsort(self.model(x), descending=True).cpu().tolist(), k)


class TeacherRanker:
    """Reference row: rank blocks by a per-block score stored in the oracle sample (e.g. teacher attention)."""

    def __init__(self, field: str, name: str):
        self.field, self.name = field, name

    def rank(self, question, texts, q_emb=None, block_embs=None, k=None, sample=None) -> List[int]:
        if sample is None or not sample.get(self.field):
            raise ValueError(f"TeacherRanker needs a sample with '{self.field}'")
        scores = [float(sample[self.field].get(str(b["id"]), 0.0)) for b in sample["blocks"]]
        return _top(np.argsort(-np.asarray(scores), kind="stable"), k)


def build_rankers(spec: Sequence[str], model_paths: Optional[Dict[str, object]] = None) -> List[Ranker]:
    """Build rankers from names: bm25 | dense[:<encoder>] | hybrid | cross_encoder[:<model>] | maple:<key>."""
    model_paths = model_paths or {}
    out: List[Ranker] = []
    for item in spec:
        kind, _, arg = item.partition(":")
        if kind == "bm25" and not arg:
            out.append(BM25Ranker())
        elif kind == "dense":
            out.append(DenseRanker(arg or None))
        elif kind == "hybrid" and not arg:
            out.append(HybridRRFRanker())
        elif kind == "cross_encoder":
            out.append(CrossEncoderRanker(arg or "BAAI/bge-reranker-base", name=item))
        elif kind == "maple" and arg:
            if arg not in model_paths:
                raise ValueError(f"'{item}' needs model_paths['{arg}']")
            from maplecore import load_model
            out.append(ModelRanker(load_model(model_paths[arg]), name=item))
        else:
            raise ValueError(f"unknown ranker spec '{item}' (bm25 | dense[:enc] | hybrid | cross_encoder[:model] | maple:<key>)")
    return out
