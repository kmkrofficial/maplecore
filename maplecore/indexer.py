"""
MAPLE Indexer
=============
Handles document chunking, BGE embedding, and index I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# BGE query instruction prefix
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Block:
    """Represents a single text block."""
    id: int
    text: str
    start_char: int = 0
    end_char: int = 0


@dataclass
class Index:
    """Document index containing blocks and embeddings."""
    blocks: List[Block] = field(default_factory=list)
    embeddings: Optional[torch.Tensor] = None
    chunk_size: int = 500
    source_path: Optional[str] = None
    
    @property
    def num_blocks(self) -> int:
        """Return number of blocks in index."""
        return len(self.blocks)
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self.embeddings is not None:
            return self.embeddings.shape[1]
        return 0


class MapleIndexer:
    """
    Document indexer using BGE embeddings.
    
    Handles text chunking, embedding generation, and index serialization.
    """
    
    BGE_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 384
    
    def __init__(
        self, 
        device: str = "cuda",
        batch_size: int = 32
    ) -> None:
        """
        Initialize the indexer.
        
        Args:
            device: Device for embedding computation
            batch_size: Batch size for encoding
        """
        self.device = device
        self.batch_size = batch_size
        self._model = None
        
        logger.info(f"MapleIndexer initialized (device={device})")
    
    @property
    def model(self):
        """Lazy-load the BGE model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading BGE model: {self.BGE_MODEL}")
            self._model = SentenceTransformer(self.BGE_MODEL, device=self.device)
        return self._model
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 500
    ) -> List[Block]:
        """
        Split text into fixed-size character blocks.
        
        Args:
            text: Input text
            chunk_size: Characters per block
            
        Returns:
            List of Block objects
        """
        blocks = []
        for i in range(0, len(text), chunk_size):
            block_text = text[i:i+chunk_size].strip()
            if block_text:
                blocks.append(Block(
                    id=len(blocks),
                    text=block_text,
                    start_char=i,
                    end_char=min(i + chunk_size, len(text))
                ))
        
        logger.debug(f"Created {len(blocks)} blocks (chunk_size={chunk_size})")
        return blocks
    
    def encode_blocks(
        self, 
        blocks: List[Block],
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode blocks using BGE model.
        
        Args:
            blocks: List of Block objects
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding tensor [N, 384]
        """
        texts = [b.text for b in blocks]
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            batch_size=self.batch_size,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query using BGE model with instruction prefix.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding tensor [384]
        """
        prefixed = _QUERY_PREFIX + query
        embedding = self.model.encode(
            prefixed,
            convert_to_tensor=True,
            device=self.device
        )
        return embedding
    
    def create_index(
        self, 
        text: str, 
        chunk_size: int = 500,
        source_path: Optional[str] = None
    ) -> Index:
        """
        Create an index from text.
        
        Args:
            text: Input document text
            chunk_size: Characters per block
            source_path: Optional source file path
            
        Returns:
            Index object with blocks and embeddings
        """
        logger.info(f"Creating index (text length: {len(text):,} chars)")
        
        blocks = self.chunk_text(text, chunk_size)
        embeddings = self.encode_blocks(blocks)
        
        index = Index(
            blocks=blocks,
            embeddings=embeddings,
            chunk_size=chunk_size,
            source_path=source_path
        )
        
        logger.info(f"Index created: {index.num_blocks} blocks, dim={index.embedding_dim}")
        return index
    
    def index_file(
        self, 
        path: Union[str, Path], 
        chunk_size: int = 500,
        encoding: str = "utf-8"
    ) -> Index:
        """
        Create an index from a file.
        
        Args:
            path: Path to text file
            chunk_size: Characters per block
            encoding: File encoding
            
        Returns:
            Index object
        """
        path = Path(path)
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        
        return self.create_index(text, chunk_size, str(path))
    
    def save_index(self, index: Index, path: Union[str, Path]) -> None:
        """
        Save index to disk.
        
        Args:
            index: Index object to save
            path: Output path (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        emb_path = path.with_suffix(".embeddings.pt")
        torch.save(index.embeddings, emb_path)
        
        # Save metadata
        meta_path = path.with_suffix(".meta.npz")
        np.savez(
            meta_path,
            block_texts=np.array([b.text for b in index.blocks], dtype=object),
            block_starts=np.array([b.start_char for b in index.blocks]),
            block_ends=np.array([b.end_char for b in index.blocks]),
            chunk_size=np.array([index.chunk_size]),
            source_path=np.array([index.source_path or ""])
        )
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: Union[str, Path]) -> Index:
        """
        Load index from disk.
        
        Args:
            path: Index path (without extension)
            
        Returns:
            Loaded Index object
        """
        path = Path(path)
        
        # Load embeddings
        emb_path = path.with_suffix(".embeddings.pt")
        embeddings = torch.load(emb_path, map_location=self.device, weights_only=True)
        
        # Load metadata
        meta_path = path.with_suffix(".meta.npz")
        meta = np.load(meta_path, allow_pickle=True)
        
        blocks = []
        for i, (text, start, end) in enumerate(zip(
            meta["block_texts"], meta["block_starts"], meta["block_ends"]
        )):
            blocks.append(Block(id=i, text=str(text), start_char=int(start), end_char=int(end)))
        
        index = Index(
            blocks=blocks,
            embeddings=embeddings,
            chunk_size=int(meta["chunk_size"][0]),
            source_path=str(meta["source_path"][0]) or None
        )
        
        logger.info(f"Index loaded: {index.num_blocks} blocks")
        return index
