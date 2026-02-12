"""
MAPLE Search Engine
====================
Implements Linear, Hierarchical, and Adaptive search strategies
for Memory-Aware Predictive Loading.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch

from .core import MapleNet
from .indexer import Index
from .utils import _compute_entropy, _cosine_similarity

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
    metadata: dict
    
    @property
    def num_results(self) -> int:
        return len(self.block_ids)


class MapleScanner:
    """
    MAPLE search engine.
    
    Supports multiple search strategies:
    - Linear: Score all blocks
    - Hierarchical: Chapter-level filtering then block-level
    - Adaptive: Entropy-aware with dynamic K and fallback
    """
    
    def __init__(
        self,
        model: MapleNet,
        device: str = "cuda",
        confidence_threshold: float = 0.15,
        mass_target: float = 0.80,
        max_blocks: int = 20,
        chapter_size: int = 100
    ) -> None:
        """
        Initialize the scanner.
        
        Args:
            model: MapleNet model
            device: Compute device
            confidence_threshold: Min confidence before fallback
            mass_target: Target attention mass for adaptive K
            max_blocks: Maximum blocks to return
            chapter_size: Blocks per chapter for hierarchical search
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.confidence_threshold = confidence_threshold
        self.mass_target = mass_target
        self.max_blocks = max_blocks
        self.chapter_size = chapter_size
        
        logger.info(f"MapleScanner initialized (device={device})")
    
    def _score_blocks(
        self, 
        query_emb: torch.Tensor, 
        block_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Score all blocks using MAPLE model.
        
        Args:
            query_emb: Query embedding [dim]
            block_embs: Block embeddings [N, dim]
            
        Returns:
            Scores tensor [N]
        """
        query_emb = query_emb.to(self.device)
        block_embs = block_embs.to(self.device)
        
        with torch.no_grad():
            num_blocks = block_embs.shape[0]
            query_expanded = query_emb.unsqueeze(0).expand(num_blocks, -1)
            combined = torch.cat([query_expanded, block_embs], dim=1)
            logits = self.model(combined)
            scores = torch.sigmoid(logits)
        
        return scores
    
    def search_linear(
        self,
        query_emb: torch.Tensor,
        index: Index,
        k: int = 5
    ) -> SearchResult:
        """
        Linear search: Score all blocks and return top-k.
        
        Args:
            query_emb: Query embedding
            index: Document index
            k: Number of results
            
        Returns:
            SearchResult
        """
        start = time.perf_counter()
        
        scores = self._score_blocks(query_emb, index.embeddings)
        top_scores, top_ids = torch.topk(scores, min(k, len(scores)))
        
        latency = (time.perf_counter() - start) * 1000
        
        return SearchResult(
            block_ids=top_ids.cpu().tolist(),
            scores=top_scores.cpu().tolist(),
            strategy_used="linear",
            latency_ms=latency,
            metadata={"blocks_scanned": index.num_blocks}
        )
    
    def search_hierarchical(
        self,
        query_emb: torch.Tensor,
        index: Index,
        k: int = 5,
        top_chapters: int = 5
    ) -> SearchResult:
        """
        Hierarchical search: Filter by chapter, then by block.
        
        Args:
            query_emb: Query embedding
            index: Document index
            k: Number of results
            top_chapters: Number of chapters to consider
            
        Returns:
            SearchResult
        """
        start = time.perf_counter()
        
        # Build chapter embeddings
        num_chapters = index.num_blocks // self.chapter_size
        if num_chapters == 0:
            return self.search_linear(query_emb, index, k)
        
        chapter_embs = []
        for i in range(num_chapters):
            start_idx = i * self.chapter_size
            end_idx = start_idx + self.chapter_size
            chapter_emb = index.embeddings[start_idx:end_idx].mean(dim=0)
            chapter_embs.append(chapter_emb)
        
        chapter_embs = torch.stack(chapter_embs)
        
        # Score chapters
        chapter_scores = self._score_blocks(query_emb, chapter_embs)
        _, top_chap_ids = torch.topk(chapter_scores, min(top_chapters, num_chapters))
        
        # Gather blocks from top chapters
        candidate_indices = []
        for chap_id in top_chap_ids.tolist():
            start_idx = chap_id * self.chapter_size
            end_idx = start_idx + self.chapter_size
            candidate_indices.extend(range(start_idx, min(end_idx, index.num_blocks)))
        
        candidate_embs = index.embeddings[candidate_indices]
        
        # Score candidates
        candidate_scores = self._score_blocks(query_emb, candidate_embs)
        top_scores, top_relative = torch.topk(candidate_scores, min(k, len(candidate_scores)))
        
        # Map back to global indices
        top_ids = [candidate_indices[i] for i in top_relative.tolist()]
        
        latency = (time.perf_counter() - start) * 1000
        
        return SearchResult(
            block_ids=top_ids,
            scores=top_scores.cpu().tolist(),
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
        Adaptive search: Hierarchical scouting with entropy-based confidence.
        
        Algorithm:
        1. Scout: Score chapters (coarse-grained) -> O(N/C)
        2. Check: Compute metrics (entropy, confidence) on chapter scores
        3. Select: 
           - High confidence -> Top-K chapters (Fast Path)
           - Low confidence -> Top-50 chapters (Wide Fallback)
        4. Score: Score blocks within selected chapters -> O(1)
        
        Args:
            query_emb: Query embedding
            index: Document index
            k: Minimum number of results
            
        Returns:
            SearchResult
        """
        start = time.perf_counter()
        
        query_emb = query_emb.to(self.device)
        # Note: We do NOT move all block_embs to device here anymore,
        # to avoid O(N) memory transfer if index is large and on CPU.
        
        # --- Phase 1: Chapter Scouting ---
        num_chapters = index.num_blocks // self.chapter_size
        if num_chapters == 0:
            return self.search_linear(query_emb, index, k)
            
        # Build/Get chapter embeddings (mean pooling)
        # Optimization: In production, these should be pre-computed.
        # Here we compute on-the-fly but only move 1/C tensors to GPU.
        chapter_embs = []
        for i in range(num_chapters):
            start_idx = i * self.chapter_size
            end_idx = start_idx + self.chapter_size
            # tensordot/mean on CPU is fast enough for 10M/100 = 100k vectors?
            # Or move batch to GPU? 
            # For simplicity & memory safety, compute on device where embeddings are.
            chunk = index.embeddings[start_idx:end_idx]
            chapter_emb = chunk.mean(dim=0)
            chapter_embs.append(chapter_emb)
            
        chapter_embs = torch.stack(chapter_embs).to(self.device)
        
        # Score chapters
        chapter_scores = self._score_blocks(query_emb, chapter_embs)
        
        # --- Phase 2: Entropy Check ---
        # Normalize to probability distribution
        probs = chapter_scores / chapter_scores.sum()
        entropy = _compute_entropy(probs)
        max_conf = chapter_scores.max().item()
        
        # --- Phase 3: Selection ---
        if max_conf < self.confidence_threshold:
            # Low confidence: Widen search window significantly
            # Still avoids full linear scan (which might be 10M blocks)
            top_n_chapters = 50  # Scan top 50 chapters (~5000 blocks)
            action = "Fallback (Wide Search)"
        else:
            # High confidence: Focused search
            top_n_chapters = 5   # Scan top 5 chapters (~500 blocks)
            action = "Fast Path"

        # Instrumentation
        logger.debug(f"Adaptive: {action} (Entropy {entropy:.2f})")
        
        # Select top chapters
        _, top_chap_ids = torch.topk(chapter_scores, min(top_n_chapters, num_chapters))
        
        # Cleanup intermediate tensors to prevent VRAM spikes
        del chapter_embs, chapter_scores
        # torch.cuda.empty_cache() # Optional, but can slow down loop
        
        # Gather candidate blocks
        candidate_indices = []
        for chap_id in top_chap_ids.tolist():
            start_idx = chap_id * self.chapter_size
            end_idx = start_idx + self.chapter_size
            candidate_indices.extend(range(start_idx, min(end_idx, index.num_blocks)))
            
        # Get candidate embeddings (transfer only needed blocks to device)
        candidate_embs = index.embeddings[candidate_indices].to(self.device)
        
        # --- Phase 4: Block Scoring & Result Selection ---
        block_scores = self._score_blocks(query_emb, candidate_embs)
        
        # Apply Mass Target Logic on Candidates
        # (Same logic as original adaptive, but on subset)
        probs_blocks = block_scores / block_scores.sum()
        
        sorted_indices_local = torch.argsort(block_scores, descending=True)
        sorted_probs = probs_blocks[sorted_indices_local]
        
        cumulative_mass = 0.0
        selected_ids = []
        selected_scores = []
        
        for i, idx in enumerate(sorted_indices_local):
            if cumulative_mass >= self.mass_target or len(selected_ids) >= self.max_blocks:
                break
            
            global_idx = candidate_indices[idx.item()]
            selected_ids.append(global_idx)
            selected_scores.append(block_scores[idx].item())
            cumulative_mass += sorted_probs[i].item()
            
        # Ensure at least k results
        while len(selected_ids) < k and len(selected_ids) < len(candidate_indices):
            next_local_idx = sorted_indices_local[len(selected_ids)].item()
            global_idx = candidate_indices[next_local_idx]
            selected_ids.append(global_idx)
            selected_scores.append(block_scores[next_local_idx].item())
            
        latency = (time.perf_counter() - start) * 1000
        
        return SearchResult(
            block_ids=selected_ids,
            scores=selected_scores,
            strategy_used="adaptive",
            latency_ms=latency,
            metadata={
                "entropy": entropy,
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
        Search the index using specified strategy.
        
        Args:
            query_emb: Query embedding
            index: Document index
            k: Number of results
            strategy: Search strategy ('linear', 'hierarchical', 'adaptive')
            
        Returns:
            SearchResult
        """
        strategy = strategy.lower()
        
        if strategy == "linear":
            return self.search_linear(query_emb, index, k)
        elif strategy == "hierarchical":
            return self.search_hierarchical(query_emb, index, k)
        elif strategy == "adaptive":
            return self.search_adaptive(query_emb, index, k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
