#!/usr/bin/env python3
"""
run_adaptive_inference.py
=========================
Scout-KV Phase 10: Adaptive Inference

Implements Entropy-Aware Adaptive Paging:
- Confidence detection via score entropy
- Dynamic K selection (80% mass coverage)
- RAG fallback for low-confidence queries
"""

import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


# =============================================================================
# ScoutBGE Model
# =============================================================================

class ScoutBGE(nn.Module):
    """Scout model for BGE embeddings."""
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# =============================================================================
# Adaptive Inference Result
# =============================================================================

@dataclass
class AdaptiveResult:
    """Result from adaptive Scout inference."""
    block_ids: List[int]
    action: str  # "Adaptive Selection" or "RAG Fallback"
    entropy: float
    max_confidence: float
    mass_coverage: float
    num_blocks: int


# =============================================================================
# Adaptive Inference Function
# =============================================================================

def compute_entropy(probs: torch.Tensor) -> float:
    """Compute entropy of probability distribution."""
    # Avoid log(0)
    probs = probs.clamp(min=1e-10)
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()


def cosine_similarity_fallback(query_emb: torch.Tensor, 
                                block_embs: torch.Tensor, 
                                k: int = 5) -> List[int]:
    """RAG fallback using cosine similarity."""
    query_norm = query_emb / query_emb.norm()
    block_norms = block_embs / block_embs.norm(dim=1, keepdim=True)
    similarities = torch.matmul(block_norms, query_norm)
    _, top_ids = torch.topk(similarities, k)
    return top_ids.tolist()


def adaptive_scout_inference(
    scout: nn.Module,
    query_emb: torch.Tensor,
    block_embs: torch.Tensor,
    confidence_threshold: float = 0.15,
    mass_target: float = 0.80,
    max_blocks: int = 20,
    device: str = "cuda"
) -> AdaptiveResult:
    """
    Entropy-Aware Adaptive Scout Inference.
    
    Args:
        scout: ScoutBGE model
        query_emb: Query embedding [384]
        block_embs: Block embeddings [N, 384]
        confidence_threshold: Min max_score to avoid fallback
        mass_target: Target attention mass to cover
        max_blocks: Maximum blocks to select
        device: cuda or cpu
    
    Returns:
        AdaptiveResult with selected blocks and metadata
    """
    scout = scout.to(device)
    query_emb = query_emb.to(device)
    block_embs = block_embs.to(device)
    
    with torch.no_grad():
        # Batch inference
        num_blocks = block_embs.shape[0]
        query_expanded = query_emb.unsqueeze(0).expand(num_blocks, -1)
        combined = torch.cat([query_expanded, block_embs], dim=1)
        
        logits = scout(combined)
        scores = torch.sigmoid(logits)
    
    # Normalize to probability distribution
    probs = scores / scores.sum()
    
    # Calculate entropy
    entropy = compute_entropy(probs)
    max_confidence = scores.max().item()
    
    # Check 1: Low Confidence → RAG Fallback
    if max_confidence < confidence_threshold:
        fallback_ids = cosine_similarity_fallback(query_emb, block_embs, k=5)
        return AdaptiveResult(
            block_ids=fallback_ids,
            action="RAG Fallback",
            entropy=entropy,
            max_confidence=max_confidence,
            mass_coverage=0.0,
            num_blocks=5
        )
    
    # Check 2: Dynamic K - accumulate until mass_target covered
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_probs = probs[sorted_indices]
    
    cumulative_mass = 0.0
    selected_ids = []
    
    for i, idx in enumerate(sorted_indices):
        if cumulative_mass >= mass_target or len(selected_ids) >= max_blocks:
            break
        selected_ids.append(idx.item())
        cumulative_mass += sorted_probs[i].item()
    
    return AdaptiveResult(
        block_ids=selected_ids,
        action="Adaptive Selection",
        entropy=entropy,
        max_confidence=max_confidence,
        mass_coverage=cumulative_mass,
        num_blocks=len(selected_ids)
    )


# =============================================================================
# Main Demo
# =============================================================================

def main():
    print("="*70)
    print("Scout-KV Phase 10: Adaptive Inference")
    print("="*70)
    
    # =========================================================================
    # Setup
    # =========================================================================
    print("\n[Setup] Loading models...")
    
    # Load BGE
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    print("  ✓ BGE loaded")
    
    # Load Scout
    scout = ScoutBGE(input_dim=768, hidden_dim=128, dropout=0.3)
    scout.load_state_dict(torch.load("scout_bge.pth", weights_only=True))
    scout.eval()
    print("  ✓ ScoutBGE loaded")
    
    # Load Sherlock Holmes
    cache_file = Path("sherlock_holmes.txt")
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        import requests
        url = "https://www.gutenberg.org/files/1661/1661-0.txt"
        response = requests.get(url, timeout=30)
        text = response.text
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
    
    # Clean
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    text = text[start:end] if end != -1 else text[start:]
    
    # Chunk
    BLOCK_CHARS = 500
    blocks = []
    for i in range(0, len(text), BLOCK_CHARS):
        block = text[i:i+BLOCK_CHARS].strip()
        if block:
            blocks.append(block)
    
    print(f"  ✓ Loaded {len(blocks)} blocks from Sherlock Holmes")
    
    # Encode all blocks
    print("  Encoding blocks...")
    block_embs = bge.encode(blocks, convert_to_tensor=True, device="cuda", 
                            batch_size=32, show_progress_bar=True)
    
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    # =========================================================================
    # Test Cases
    # =========================================================================
    test_cases = [
        {
            "name": "A: The Needle",
            "query": "What was the speckled band?",
            "expectation": "High Confidence, Low Entropy → Specific blocks"
        },
        {
            "name": "B: The Ambiguous",
            "query": "Explain the socio-economic impact of the industrial revolution in London.",
            "expectation": "High Entropy → Many blocks OR Fallback"
        },
        {
            "name": "C: The Garbage",
            "query": "sdfsdf sdf sdf",
            "expectation": "Very Low Confidence → RAG Fallback"
        }
    ]
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    for case in test_cases:
        print(f"\n{'─'*70}")
        print(f"Case {case['name']}")
        print(f"{'─'*70}")
        print(f"Query: \"{case['query']}\"")
        print(f"Expected: {case['expectation']}")
        
        # Encode query
        query_emb = bge.encode(QUERY_PREFIX + case["query"], 
                               convert_to_tensor=True, device="cuda")
        
        # Run adaptive inference
        result = adaptive_scout_inference(
            scout=scout,
            query_emb=query_emb,
            block_embs=block_embs,
            confidence_threshold=0.15,
            mass_target=0.80,
            max_blocks=20
        )
        
        print(f"\n┌─ Results ─────────────────────────────────")
        print(f"│ Scout Entropy:    {result.entropy:.4f}")
        print(f"│ Max Confidence:   {result.max_confidence:.4f}")
        print(f"│ Action:           {result.action}")
        print(f"│ Blocks Selected:  {result.num_blocks}")
        if result.action == "Adaptive Selection":
            print(f"│ Mass Coverage:    {result.mass_coverage*100:.1f}%")
        print(f"│ Block IDs:        {result.block_ids[:5]}{'...' if len(result.block_ids) > 5 else ''}")
        print(f"└────────────────────────────────────────────")
        
        # Show block preview for first result
        if result.block_ids:
            preview = blocks[result.block_ids[0]][:80].replace("\n", " ")
            print(f"  Preview: \"{preview}...\"")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("ADAPTIVE INFERENCE SUMMARY")
    print("="*70)
    print("• Low Confidence (<0.15) triggers RAG Fallback")
    print("• Dynamic K accumulates blocks until 80% mass coverage")
    print("• Maximum 20 blocks to prevent context explosion")
    print("="*70)


if __name__ == "__main__":
    main()
