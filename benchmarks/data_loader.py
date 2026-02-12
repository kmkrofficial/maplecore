"""
MAPLE Performance Lab -- Data Loader
======================================
Download and cache datasets for benchmarking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

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
    """
    import requests

    cache_dir = DATA_DIR / "gutenberg"
    cache_dir.mkdir(parents=True, exist_ok=True)

    books: Dict[int, str] = {}

    for book_id in book_ids:
        cache_path = cache_dir / f"{book_id}.txt"

        if cache_path.exists():
            logger.info(f"[cache hit] Gutenberg #{book_id}")
            books[book_id] = cache_path.read_text(encoding="utf-8")
            continue

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


def build_large_corpus(target_chars: int) -> str:
    """
    Build a large text corpus by tiling Gutenberg books.

    Downloads a set of classic novels and repeats them until the
    target character count is reached.

    Args:
        target_chars: Desired corpus length in characters.

    Returns:
        Corpus string of approximately ``target_chars`` length.
    """
    # Classic books with diverse styles
    book_ids = [1661, 1342, 84, 11]  # Sherlock, P&P, Frankenstein, Alice
    books = get_gutenberg_books(book_ids)

    if not books:
        # Fallback: generate synthetic filler
        logger.warning("No books available, generating synthetic corpus")
        paragraph = (
            "The morning sun cast long shadows across the ancient stone walls "
            "of the library. Dust motes danced in the amber light that filtered "
            "through stained glass windows, illuminating row upon row of leather "
            "bound volumes. The scent of old paper and dried ink permeated the air. "
        ) * 10 + "\n\n"
        return (paragraph * (target_chars // len(paragraph) + 1))[:target_chars]

    combined = "\n\n".join(books.values())

    # Tile if needed
    if len(combined) < target_chars:
        repeats = target_chars // len(combined) + 1
        combined = (combined + "\n\n") * repeats

    return combined[:target_chars]


# ---------------------------------------------------------------------------
# NarrativeQA
# ---------------------------------------------------------------------------

def get_narrative_qa(split: str = "test"):
    """
    Load the NarrativeQA dataset via HuggingFace ``datasets``.

    Args:
        split: Dataset split -- ``'train'``, ``'test'``, or ``'validation'``.

    Returns:
        A HuggingFace ``Dataset``.
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


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def get_hotpot_qa(
    split: str = "validation",
    max_samples: Optional[int] = None,
):
    """
    Load HotpotQA (distractor setting) via HuggingFace ``datasets``.

    Each sample contains a question, a set of context paragraphs
    (2 gold + 8 distractors), and supporting facts.

    Args:
        split: Dataset split -- ``'train'`` or ``'validation'``.
        max_samples: Optional cap on number of samples.

    Returns:
        A HuggingFace ``Dataset``.

    Example::

        >>> ds = get_hotpot_qa("validation", max_samples=100)
        >>> sample = ds[0]
        >>> print(sample["question"])
    """
    from datasets import load_dataset

    logger.info(f"Loading HotpotQA distractor (split={split})")
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split=split,
        trust_remote_code=True,
    )

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    logger.info(f"HotpotQA loaded: {len(dataset)} samples")
    return dataset
