"""
MAPLE Indexer
=============
Handles document chunking, embedding, and index I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .utils import resolve_device

logger = logging.getLogger(__name__)

# BGE query instruction prefix (only applied to BGE-family encoders)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def query_prefix_for(model_name: str) -> str:
    """Return the query instruction prefix recommended for an encoder."""
    name = model_name.lower()
    if "bge" in name and "-en" in name:
        return _QUERY_PREFIX
    return ""


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
    embedding_model: Optional[str] = None
    chunk_overlap: int = 0
    _chapter_cache: Dict[int, torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )
    _normalized: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)

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

    def normalized_embeddings(self) -> torch.Tensor:
        """Unit-norm copy of the block embeddings (float32), computed once and cached.

        Dense pre-filtering is then a single matrix-vector product. Do not mutate ``embeddings`` afterwards."""
        if self._normalized is None:
            self._normalized = torch.nn.functional.normalize(self.embeddings.float(), dim=1)
        return self._normalized

    def chapter_embeddings(self, chapter_size: int) -> torch.Tensor:
        """
        Mean-pooled chapter embeddings, cached per ``chapter_size``.

        The final chapter may be partial, so every block belongs to
        exactly one chapter (``ceil(num_blocks / chapter_size)`` chapters).
        """
        if chapter_size not in self._chapter_cache:
            self._chapter_cache[chapter_size] = compute_chapter_embeddings(
                self.embeddings, chapter_size
            )
        return self._chapter_cache[chapter_size]


def compute_chapter_embeddings(embeddings: torch.Tensor, chapter_size: int) -> torch.Tensor:
    """
    Mean-pool consecutive groups of ``chapter_size`` block embeddings.

    Includes the trailing partial chapter so no block is unreachable.

    Returns:
        Tensor of shape ``[ceil(N / chapter_size), dim]``.
    """
    if chapter_size <= 0:
        raise ValueError("chapter_size must be positive")
    n, dim = embeddings.shape
    n_full = n // chapter_size
    parts = []
    if n_full > 0:
        full = embeddings[: n_full * chapter_size].reshape(n_full, chapter_size, dim)
        parts.append(full.float().mean(dim=1))
    if n % chapter_size:
        parts.append(embeddings[n_full * chapter_size:].float().mean(dim=0, keepdim=True))
    return torch.cat(parts, dim=0).to(embeddings.dtype)


class MapleIndexer:
    """
    Document indexer using sentence-transformer embeddings (BGE by default).

    Handles text chunking, embedding generation, and index serialization.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32
    ) -> None:
        """
        Initialize the indexer.

        Args:
            model_name: HuggingFace model identifier
            device: Device for embedding computation (None = auto-detect)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.query_prefix = query_prefix_for(model_name)
        self._model = None

        logger.info(f"MapleIndexer initialized (device={self.device})")

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
        return self._model

    def get_embedding_dimension(self) -> int:
        """Dynamically get the dimension of the loaded embedding model."""
        return self.model.get_sentence_embedding_dimension()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 0
    ) -> List[Block]:
        """
        Split text into fixed-size character blocks.

        Args:
            text: Input text
            chunk_size: Characters per block
            overlap: Characters shared between consecutive blocks
                (0 reproduces the original non-overlapping chunking used
                to build the oracle datasets).

        Returns:
            List of Block objects
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")

        step = chunk_size - overlap
        blocks = []
        for i in range(0, len(text), step):
            block_text = text[i:i + chunk_size].strip()
            if block_text:
                blocks.append(Block(
                    id=len(blocks),
                    text=block_text,
                    start_char=i,
                    end_char=min(i + chunk_size, len(text))
                ))
            if i + chunk_size >= len(text):
                break

        logger.debug(f"Created {len(blocks)} blocks (chunk_size={chunk_size}, overlap={overlap})")
        return blocks

    def encode_texts(self, texts: List[str], show_progress: bool = False) -> torch.Tensor:
        """Encode raw passages (no query prefix)."""
        if not texts:
            return torch.empty(0, self.get_embedding_dimension())
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            batch_size=self.batch_size,
            show_progress_bar=show_progress
        )

    def encode_blocks(
        self,
        blocks: List[Block],
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode blocks using the embedding model.

        Args:
            blocks: List of Block objects
            show_progress: Whether to show progress bar

        Returns:
            Embedding tensor [N, dim]
        """
        return self.encode_texts([b.text for b in blocks], show_progress=show_progress)

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query, adding the encoder's instruction prefix if it has one.

        Args:
            query: Query string

        Returns:
            Query embedding tensor [dim]
        """
        return self.model.encode(
            self.query_prefix + query,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )

    def create_index(
        self,
        text: str,
        chunk_size: int = 500,
        source_path: Optional[str] = None,
        overlap: int = 0,
        show_progress: bool = True
    ) -> Index:
        """
        Create an index from text.

        Args:
            text: Input document text
            chunk_size: Characters per block
            source_path: Optional source file path
            overlap: Characters shared between consecutive blocks
            show_progress: Whether to show an encoding progress bar

        Returns:
            Index object with blocks and embeddings
        """
        logger.info(f"Creating index (text length: {len(text):,} chars)")

        blocks = self.chunk_text(text, chunk_size, overlap)
        embeddings = self.encode_blocks(blocks, show_progress=show_progress)

        index = Index(
            blocks=blocks,
            embeddings=embeddings,
            chunk_size=chunk_size,
            source_path=source_path,
            embedding_model=self.model_name,
            chunk_overlap=overlap,
        )

        logger.info(f"Index created: {index.num_blocks} blocks, dim={index.embedding_dim}")
        return index

    def index_file(
        self,
        path: Union[str, Path],
        chunk_size: int = 500,
        encoding: str = "utf-8",
        overlap: int = 0
    ) -> Index:
        """
        Create an index from a file.

        Args:
            path: Path to document file (.txt, .pdf, .docx, .html)
            chunk_size: Characters per block
            encoding: Kept for backwards compatibility (text files are read as UTF-8)
            overlap: Characters shared between consecutive blocks

        Returns:
            Index object
        """
        from .parsers import extract_text_from_file

        path = Path(path)
        text = extract_text_from_file(path)

        return self.create_index(text, chunk_size, str(path), overlap=overlap)

    def save_index(self, index: Index, path: Union[str, Path]) -> None:
        """
        Save index to disk.

        Args:
            index: Index object to save
            path: Output path (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        emb_path = path.with_suffix(".embeddings.pt")
        torch.save(index.embeddings.cpu(), emb_path)

        meta_path = path.with_suffix(".meta.npz")
        np.savez(
            meta_path,
            block_texts=np.array([b.text for b in index.blocks], dtype=object),
            block_starts=np.array([b.start_char for b in index.blocks]),
            block_ends=np.array([b.end_char for b in index.blocks]),
            chunk_size=np.array([index.chunk_size]),
            chunk_overlap=np.array([index.chunk_overlap]),
            source_path=np.array([index.source_path or ""]),
            embedding_model=np.array([index.embedding_model or ""]),
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

        emb_path = path.with_suffix(".embeddings.pt")
        embeddings = torch.load(emb_path, map_location=self.device, weights_only=True)

        meta_path = path.with_suffix(".meta.npz")
        meta = np.load(meta_path, allow_pickle=True)

        blocks = []
        for i, (text, start, end) in enumerate(zip(
            meta["block_texts"], meta["block_starts"], meta["block_ends"]
        )):
            blocks.append(Block(id=i, text=str(text), start_char=int(start), end_char=int(end)))

        embedding_model = str(meta["embedding_model"][0]) if "embedding_model" in meta else ""
        if embedding_model and embedding_model != self.model_name:
            logger.warning(
                f"Index was built with '{embedding_model}' but this indexer uses "
                f"'{self.model_name}'. Query embeddings will not match."
            )

        index = Index(
            blocks=blocks,
            embeddings=embeddings,
            chunk_size=int(meta["chunk_size"][0]),
            source_path=str(meta["source_path"][0]) or None,
            embedding_model=embedding_model or None,
            chunk_overlap=int(meta["chunk_overlap"][0]) if "chunk_overlap" in meta else 0,
        )

        logger.info(f"Index loaded: {index.num_blocks} blocks")
        return index
