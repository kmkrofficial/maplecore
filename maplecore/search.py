"""
MAPLE Search Engine
====================
Implements Linear, Hierarchical, and Adaptive search strategies
for Memory-Aware Predictive Loading.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch

from .core import MapleNet
from .indexer import Index, compute_chapter_embeddings
from .utils import _compute_entropy, resolve_device

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Available search strategies."""
    LINEAR = "linear"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class SearchResult:
    """Result from a search operation."""
    block_ids: List[int]
    scores: List[float]
    strategy_used: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.block_ids)


def _chapter_embeddings(index, chapter_size: int) -> torch.Tensor:
    """Use the index's cached chapter embeddings when available."""
    if hasattr(index, "chapter_embeddings"):
        return index.chapter_embeddings(chapter_size)
    return compute_chapter_embeddings(index.embeddings, chapter_size)


class MapleScanner:
    """
    MAPLE search engine.

    Supports multiple search strategies:
    - Linear: Score all blocks
    - Hierarchical: Chapter-level filtering then block-level
    - Adaptive: Hierarchical with a confidence-gated wide fallback and
      dynamic K based on cumulative score mass
    """

    def __init__(
        self,
        model: Union[MapleNet, "MapleONNXRunner"],
        device: Optional[str] = None,
        confidence_threshold: float = 0.15,
        mass_target: float = 0.80,
        max_blocks: int = 20,
        chapter_size: int = 100,
        fast_chapters: int = 5,
        fallback_chapters: int = 50,
        entropy_threshold: Optional[float] = None,
        score_batch_size: int = 32768,
    ) -> None:
        """
        Initialize the scanner.

        Args:
            model: MapleNet model or MapleONNXRunner
            device: Compute device (None = auto-detect)
            confidence_threshold: If the best chapter score is below this,
                adaptive search widens to ``fallback_chapters``
            mass_target: Target cumulative score mass for adaptive K
            max_blocks: Maximum blocks returned by adaptive search
            chapter_size: Blocks per chapter for hierarchical/adaptive search
            fast_chapters: Chapters scanned on the adaptive fast path (and by
                hierarchical search)
            fallback_chapters: Chapters scanned on the adaptive fallback path
            entropy_threshold: Optional. If set, adaptive search also falls
                back when the *normalised* chapter-score entropy (0..1) is
                above this value, i.e. the chapter scores are flat.
            score_batch_size: Blocks scored per forward pass
        """
        self.model = model
        self.device = resolve_device(device)

        # Only PyTorch models need to be moved to device and set to eval
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.confidence_threshold = confidence_threshold
        self.mass_target = mass_target
        self.max_blocks = max_blocks
        self.chapter_size = chapter_size
        self.fast_chapters = fast_chapters
        self.fallback_chapters = fallback_chapters
        self.entropy_threshold = entropy_threshold
        self.score_batch_size = score_batch_size

        logger.info(f"MapleScanner initialized (device={self.device})")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_blocks(
        self,
        query_emb: torch.Tensor,
        block_embs: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Score blocks with the MAPLE model in memory-safe batches.

        Args:
            query_emb: Query embedding [dim]
            block_embs: Block embeddings [N, dim]
            batch_size: Blocks per forward pass

        Returns:
            Sigmoid scores tensor [N] on CPU
        """
        batch_size = batch_size or self.score_batch_size
        num_blocks = block_embs.shape[0]
        if num_blocks == 0:
            return torch.empty(0)

        is_onnx = hasattr(self.model, "predict")
        query_emb = query_emb.to(self.device).float().unsqueeze(0)

        all_scores = []
        with torch.no_grad():
            for i in range(0, num_blocks, batch_size):
                batch_embs = block_embs[i:i + batch_size].to(self.device).float()
                query_expanded = query_emb.expand(batch_embs.shape[0], -1)
                combined = torch.cat([query_expanded, batch_embs], dim=1)

                if is_onnx:
                    scores_np = self.model.predict(combined.cpu().numpy())
                    scores = torch.from_numpy(np.asarray(scores_np, dtype=np.float32).reshape(-1))
                else:
                    scores = torch.sigmoid(self.model(combined).reshape(-1)).float().cpu()

                all_scores.append(scores)

        return torch.cat(all_scores, dim=0)

    def _gather_candidates(self, chap_ids: List[int], num_blocks: int) -> List[int]:
        candidate_indices: List[int] = []
        for chap_id in chap_ids:
            start_idx = chap_id * self.chapter_size
            end_idx = min(start_idx + self.chapter_size, num_blocks)
            candidate_indices.extend(range(start_idx, end_idx))
        return candidate_indices

    def _num_chapters(self, num_blocks: int) -> int:
        return math.ceil(num_blocks / self.chapter_size)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def search_linear(
        self,
        query_emb: torch.Tensor,
        index: Index,
        k: int = 5
    ) -> SearchResult:
        """
        Linear search: Score all blocks and return top-k.
        """
        start = time.perf_counter()

        scores = self._score_blocks(query_emb, index.embeddings)
        top_scores, top_ids = torch.topk(scores, min(k, len(scores)))

        latency = (time.perf_counter() - start) * 1000

        return SearchResult(
            block_ids=top_ids.tolist(),
            scores=top_scores.tolist(),
            strategy_used="linear",
            latency_ms=latency,
            metadata={"blocks_scanned": index.num_blocks}
        )

    def search_hierarchical(
        self,
        query_emb: torch.Tensor,
        index: Index,
        k: int = 5,
        top_chapters: Optional[int] = None
    ) -> SearchResult:
        """
        Hierarchical search: score chapters, then blocks within the best ones.
        """
        start = time.perf_counter()
        top_chapters = top_chapters or self.fast_chapters

        num_chapters = self._num_chapters(index.num_blocks)
        if num_chapters <= top_chapters:
            result = self.search_linear(query_emb, index, k)
            result.metadata["fallback_reason"] = "index smaller than chapter budget"
            return result

        chapter_scores = self._score_blocks(query_emb, _chapter_embeddings(index, self.chapter_size))
        _, top_chap_ids = torch.topk(chapter_scores, min(top_chapters, num_chapters))

        candidate_indices = self._gather_candidates(top_chap_ids.tolist(), index.num_blocks)
        candidate_scores = self._score_blocks(query_emb, index.embeddings[candidate_indices])
        top_scores, top_relative = torch.topk(candidate_scores, min(k, len(candidate_scores)))
        top_ids = [candidate_indices[i] for i in top_relative.tolist()]

        latency = (time.perf_counter() - start) * 1000

        return SearchResult(
            block_ids=top_ids,
            scores=top_scores.tolist(),
            strategy_used="hierarchical",
            latency_ms=latency,
            metadata={
                "chapters_scanned": num_chapters,
                "blocks_scanned": len(candidate_indices),
                "top_chapters": top_chap_ids.tolist()
            }
        )

    def search_adaptive(
        self,
        query_emb: torch.Tensor,
        index: Index,
        k: int = 5
    ) -> SearchResult:
        """
        Adaptive search: hierarchical scouting with confidence-gated fallback.

        Algorithm:
        1. Scout: score all chapter embeddings (coarse) -> O(N / C)
        2. Check: max chapter score (and optionally normalised entropy)
        3. Select: confident -> ``fast_chapters``; otherwise ``fallback_chapters``
        4. Score blocks inside the selected chapters and keep blocks until
           ``mass_target`` of the candidate score mass is covered
           (at least ``k``, at most ``max(k, max_blocks)``).
        """
        start = time.perf_counter()

        num_chapters = self._num_chapters(index.num_blocks)

        if num_chapters <= self.fast_chapters:
            # Small index: the fast path would scan everything anyway.
            entropy = 0.0
            norm_entropy = 0.0
            max_conf = float("nan")
            action = "Linear (small index)"
            candidate_indices = list(range(index.num_blocks))
        else:
            chapter_scores = self._score_blocks(
                query_emb, _chapter_embeddings(index, self.chapter_size)
            )
            probs = chapter_scores / chapter_scores.sum().clamp(min=1e-12)
            entropy = _compute_entropy(probs)
            norm_entropy = entropy / math.log(num_chapters) if num_chapters > 1 else 0.0
            max_conf = chapter_scores.max().item()

            low_conf = max_conf < self.confidence_threshold
            flat = self.entropy_threshold is not None and norm_entropy > self.entropy_threshold
            if low_conf or flat:
                top_n_chapters = self.fallback_chapters
                action = "Fallback (Wide Search)"
            else:
                top_n_chapters = self.fast_chapters
                action = "Fast Path"

            _, top_chap_ids = torch.topk(chapter_scores, min(top_n_chapters, num_chapters))
            candidate_indices = self._gather_candidates(top_chap_ids.tolist(), index.num_blocks)

        logger.debug(f"Adaptive: {action} (entropy={entropy:.3f}, max_conf={max_conf:.3f})")

        block_scores = self._score_blocks(query_emb, index.embeddings[candidate_indices])
        order = torch.argsort(block_scores, descending=True)
        probs_blocks = block_scores / block_scores.sum().clamp(min=1e-12)

        limit = max(k, self.max_blocks)
        selected_ids: List[int] = []
        selected_scores: List[float] = []
        cumulative_mass = 0.0
        for idx in order.tolist():
            if len(selected_ids) >= limit:
                break
            if len(selected_ids) >= k and cumulative_mass >= self.mass_target:
                break
            selected_ids.append(candidate_indices[idx])
            selected_scores.append(block_scores[idx].item())
            cumulative_mass += probs_blocks[idx].item()

        latency = (time.perf_counter() - start) * 1000

        return SearchResult(
            block_ids=selected_ids,
            scores=selected_scores,
            strategy_used="adaptive",
            latency_ms=latency,
            metadata={
                "entropy": entropy,
                "normalized_entropy": norm_entropy,
                "max_confidence": max_conf,
                "action": action,
                "chapters_scanned": num_chapters,
                "blocks_scored": len(candidate_indices)
            }
        )

    def search(
        self,
        query_emb: torch.Tensor,
        index: Index,
        k: int = 5,
        strategy: str = "adaptive"
    ) -> SearchResult:
        """
        Search the index using the specified strategy
        ('linear', 'hierarchical', 'adaptive').
        """
        if isinstance(strategy, SearchStrategy):
            strategy = strategy.value
        strategy = strategy.lower()

        if strategy == "linear":
            return self.search_linear(query_emb, index, k)
        elif strategy == "hierarchical":
            return self.search_hierarchical(query_emb, index, k)
        elif strategy == "adaptive":
            return self.search_adaptive(query_emb, index, k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
