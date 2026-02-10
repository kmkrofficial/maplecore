"""
MAPLE Performance Lab -- Data Loader
======================================
Download and cache datasets for benchmarking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from .config import DATA_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------

def get_gutenberg_books(book_ids: List[int]) -> Dict[int, str]:
    """
    Download and cache Project Gutenberg books.

    Each book is saved to ``DATA_DIR/gutenberg/<id>.txt`` on first download.
    Subsequent calls return the cached version.

    Args:
        book_ids: List of Gutenberg book IDs (e.g. ``[1661]`` for
                  *The Adventures of Sherlock Holmes*).

    Returns:
        Dict mapping each ``book_id`` to its full text content.

    Example::

        >>> books = get_gutenberg_books([1661, 1342])
        >>> print(len(books[1661]))   # Sherlock Holmes char count
        593731
    """
    import requests

    cache_dir = DATA_DIR / "gutenberg"
    cache_dir.mkdir(parents=True, exist_ok=True)

    books: Dict[int, str] = {}

    for book_id in book_ids:
        cache_path = cache_dir / f"{book_id}.txt"

        # Return cached version if available
        if cache_path.exists():
            logger.info(f"[cache hit] Gutenberg #{book_id}")
            books[book_id] = cache_path.read_text(encoding="utf-8")
            continue

        # Try primary URL, then fallback
        urls = [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        ]

        text = None
        for url in urls:
            try:
                logger.info(f"Downloading Gutenberg #{book_id} from {url}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                text = resp.text
                break
            except Exception as exc:
                logger.debug(f"  failed ({exc}), trying next URL...")

        if text is None:
            logger.warning(f"Could not download Gutenberg #{book_id}")
            continue

        cache_path.write_text(text, encoding="utf-8")
        books[book_id] = text
        logger.info(
            f"  Gutenberg #{book_id}: {len(text):,} chars -> {cache_path.name}"
        )

    return books


# ---------------------------------------------------------------------------
# NarrativeQA
# ---------------------------------------------------------------------------

def get_narrative_qa(split: str = "test"):
    """
    Load the NarrativeQA dataset via HuggingFace ``datasets``.

    The dataset is streamed on first call and cached locally by the
    ``datasets`` library for subsequent runs.

    Args:
        split: Dataset split -- ``'train'``, ``'test'``, or ``'validation'``.

    Returns:
        A HuggingFace ``Dataset`` (or ``IterableDataset`` if streaming).

    Example::

        >>> ds = get_narrative_qa("test")
        >>> sample = next(iter(ds))
        >>> print(sample["question"]["text"])
    """
    from datasets import load_dataset

    logger.info(f"Loading NarrativeQA (split={split})")
    dataset = load_dataset(
        "deepmind/narrativeqa",
        split=split,
        trust_remote_code=True,
    )
    logger.info(f"NarrativeQA loaded: {len(dataset)} samples")
    return dataset
