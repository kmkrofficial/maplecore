"""
Scout-KV: Speculative Paging for Infinite Context LLMs
=======================================================

Scout-KV is a learned retrieval system that achieves 71% Recall@5
compared to 30% for standard RAG, while maintaining sub-millisecond latency.

Quick Start:
>>> from scoutkv import ScoutKV
>>> client = ScoutKV(model_path="scout_bge.pth")
>>> client.index_file("books/holmes.txt")
>>> results = client.query("Who is the killer?")
>>> print(results.blocks)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from .core import ScoutBGE
from .indexer import Block, Index, Indexer
from .search import Scanner, SearchResult, SearchStrategy
from .trainer import ScoutTrainer
from .utils import get_device, setup_logging

__version__ = "0.1.0"
__author__ = "Scout-KV Team"

# Default logger
logger = logging.getLogger(__name__)


class ScoutKV:
    """
    High-level Scout-KV client.
    
    Provides a simple interface for document indexing and search.
    
    Example:
        >>> client = ScoutKV(model_path="scout_bge.pth")
        >>> client.index_file("document.txt")
        >>> results = client.query("What is the main topic?")
        >>> for block_id in results.block_ids[:5]:
        ...     print(client.get_block(block_id).text)
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        chunk_size: int = 500,
        strategy: str = "adaptive"
    ) -> None:
        """
        Initialize Scout-KV client.
        
        Args:
            model_path: Path to Scout model weights (.pth)
            device: Compute device ('cuda', 'cpu', or None for auto)
            chunk_size: Default chunk size for indexing
            strategy: Default search strategy ('linear', 'hierarchical', 'adaptive')
        """
        self.device = device or str(get_device())
        self.chunk_size = chunk_size
        self.strategy = strategy
        
        # Initialize components
        self.indexer = Indexer(device=self.device)
        self._index: Optional[Index] = None
        self._scanner: Optional[Scanner] = None
        
        # Load model if provided
        if model_path is not None:
            self.load_model(model_path)
        
        logger.info(f"ScoutKV initialized (device={self.device}, strategy={strategy})")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a Scout model.
        
        Args:
            path: Path to model weights
        """
        model = ScoutBGE.load(path, device=self.device)
        self._scanner = Scanner(model, device=self.device)
        logger.info(f"Model loaded from {path}")
    
    def index_text(self, text: str, chunk_size: Optional[int] = None) -> Index:
        """
        Index a text document.
        
        Args:
            text: Document text
            chunk_size: Override default chunk size
            
        Returns:
            Created Index
        """
        chunk_size = chunk_size or self.chunk_size
        self._index = self.indexer.create_index(text, chunk_size=chunk_size)
        return self._index
    
    def index_file(
        self, 
        path: Union[str, Path], 
        chunk_size: Optional[int] = None
    ) -> Index:
        """
        Index a text file.
        
        Args:
            path: Path to text file
            chunk_size: Override default chunk size
            
        Returns:
            Created Index
        """
        chunk_size = chunk_size or self.chunk_size
        self._index = self.indexer.index_file(path, chunk_size=chunk_size)
        return self._index
    
    def save_index(self, path: Union[str, Path]) -> None:
        """
        Save the current index to disk.
        
        Args:
            path: Output path (without extension)
        """
        if self._index is None:
            raise ValueError("No index to save. Call index_text() or index_file() first.")
        self.indexer.save_index(self._index, path)
    
    def load_index(self, path: Union[str, Path]) -> Index:
        """
        Load an index from disk.
        
        Args:
            path: Index path (without extension)
            
        Returns:
            Loaded Index
        """
        self._index = self.indexer.load_index(path)
        return self._index
    
    def query(
        self,
        question: str,
        k: int = 5,
        strategy: Optional[str] = None
    ) -> SearchResult:
        """
        Query the indexed document.
        
        Args:
            question: Query string
            k: Number of results
            strategy: Override default strategy
            
        Returns:
            SearchResult with block IDs and scores
        """
        if self._index is None:
            raise ValueError("No index loaded. Call index_text() or index_file() first.")
        if self._scanner is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        query_emb = self.indexer.encode_query(question)
        strategy = strategy or self.strategy
        
        return self._scanner.search(query_emb, self._index, k=k, strategy=strategy)
    
    def get_block(self, block_id: int) -> Block:
        """
        Get a block by ID.
        
        Args:
            block_id: Block ID
            
        Returns:
            Block object
        """
        if self._index is None:
            raise ValueError("No index loaded.")
        return self._index.blocks[block_id]
    
    def get_context(self, result: SearchResult, max_blocks: Optional[int] = None) -> str:
        """
        Get concatenated text from search results.
        
        Args:
            result: SearchResult from query()
            max_blocks: Limit number of blocks
            
        Returns:
            Concatenated block texts
        """
        block_ids = result.block_ids
        if max_blocks is not None:
            block_ids = block_ids[:max_blocks]
        
        texts = [self.get_block(bid).text for bid in block_ids]
        return "\n\n".join(texts)
    
    @property
    def index(self) -> Optional[Index]:
        """Return the current index."""
        return self._index
    
    @property
    def num_blocks(self) -> int:
        """Return number of indexed blocks."""
        if self._index is None:
            return 0
        return self._index.num_blocks


# Expose main classes
__all__ = [
    "ScoutKV",
    "ScoutBGE",
    "Indexer",
    "Index",
    "Block",
    "Scanner",
    "SearchResult",
    "SearchStrategy",
    "ScoutTrainer",
    "setup_logging",
    "get_device",
]
