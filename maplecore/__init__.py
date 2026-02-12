"""
MAPLE: Memory-Aware Predictive Loading Engine
==============================================

A learned retrieval system for infinite-context LLM inference.
MAPLE achieves 71% Recall@5 compared to 30% for standard RAG
while maintaining sub-millisecond search latency.

Quick Start::

    from maplecore import Maple

    client = Maple(model_path="maple.pth")
    client.index_file("books/sherlock_holmes.txt")
    results = client.query("Who is the killer?")
    context = client.get_context(results, max_blocks=5)

Public API
----------

Classes:

    Maple
        High-level client. Use for indexing, querying, and context extraction.
        
    MapleNet
        Lightweight MLP for relevance scoring.
        Architecture: 768 → 128 → 1 (sigmoid output).
        
    MapleIndexer
        Chunking + BGE embedding engine.
        Handles text → blocks → embeddings pipeline.
        
    MapleScanner
        Search engine with three strategies:
        Linear, Hierarchical, and Adaptive.
        
    MapleTrainer
        Training loop for fine-tuning MapleNet from oracle data.

Data Classes:

    Block
        A single text chunk with position metadata.
        
    Index
        Collection of Blocks + their embeddings.
        
    SearchResult
        Query result containing block IDs, scores, and latency.
        
    SearchStrategy
        Enum: LINEAR, HIERARCHICAL, ADAPTIVE.

Utilities:

    setup_logging()  — Configure structured logging.
    get_device()     — Auto-detect best available device.
"""

from .client import Maple
from .core import MapleNet
from .indexer import Block, Index, MapleIndexer
from .search import MapleScanner, SearchResult, SearchStrategy
from .trainer import MapleTrainer
from .utils import get_device, setup_logging

__version__ = "0.1.0-alpha"
__author__ = "Keerthi Raajan K M"

__all__ = [
    # Primary API
    "Maple",
    # Model & Search
    "MapleNet",
    "MapleScanner",
    "MapleIndexer",
    "MapleTrainer",
    # Data types
    "Index",
    "Block",
    "SearchResult",
    "SearchStrategy",
    # Utilities
    "setup_logging",
    "get_device",
]
