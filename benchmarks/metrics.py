"""
MAPLE evaluation metrics
========================
Pure functions (no torch / model dependencies) used by every benchmark.
Definitions are documented in docs/METRICS.md — keep the two in sync.

Conventions:
    ranking : list of block ids, best first
    gold    : set of relevant block ids (may be empty -> metric undefined)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------- retrieval

def recall_at_k(ranking: Sequence, gold: Set, k: int, capped: bool = False) -> Optional[float]:
    """
    |top-k ∩ gold| / |gold|.

    ``capped=True`` divides by ``min(|gold|, k)`` instead, so a perfect
    ranking scores 1.0 even when k < |gold| (used for oracle top-5 labels,
    matching the v0.2.0 definition).
    """
    if not gold:
        return None
    hits = len(set(ranking[:k]) & gold)
    denom = min(len(gold), k) if capped else len(gold)
    return hits / denom


def hit_all_at_k(ranking: Sequence, gold: Set, k: int) -> Optional[float]:
    """1.0 if *every* gold block is in the top-k (multi-hop completeness)."""
    if not gold:
        return None
    return float(gold.issubset(set(ranking[:k])))


def hit_any_at_k(ranking: Sequence, gold: Set, k: int) -> Optional[float]:
    """1.0 if at least one gold block is in the top-k."""
    if not gold:
        return None
    return float(bool(set(ranking[:k]) & gold))


def mrr_at_k(ranking: Sequence, gold: Set, k: int = 10) -> Optional[float]:
    """Reciprocal rank of the first gold block within the top-k (0 if none)."""
    if not gold:
        return None
    for i, bid in enumerate(ranking[:k]):
        if bid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranking: Sequence, relevance: Dict, k: int = 10) -> Optional[float]:
    """nDCG@k with graded relevance (e.g. teacher attention mass per block)."""
    if not relevance or max(relevance.values()) <= 0:
        return None
    dcg = sum(relevance.get(bid, 0.0) / math.log2(i + 2) for i, bid in enumerate(ranking[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


# --------------------------------------------------------------- answers

def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, drop punctuation and articles, squash spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_in_context(context: str, answers: Iterable[str], min_len: int = 1) -> Optional[float]:
    """1.0 if any normalised answer string occurs in the normalised context."""
    answers = [normalize_answer(a) for a in answers]
    answers = [a for a in answers if len(a) >= min_len]
    if not answers:
        return None
    ctx = normalize_answer(context)
    return float(any(a in ctx for a in answers))


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(a) for a in answers))


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    """Max SQuAD token-F1 over the reference answers."""
    best = 0.0
    p_toks = normalize_answer(prediction).split()
    for a in answers:
        a_toks = normalize_answer(a).split()
        common = Counter(p_toks) & Counter(a_toks)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p_toks), same / len(a_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ------------------------------------------------------------- statistics

def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI. Ignores None."""
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def summarize(values: Sequence[Optional[float]]) -> Dict[str, float]:
    """{mean, ci_low, ci_high, n} in percent-free units (0..1)."""
    mean, lo, hi = bootstrap_ci(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi,
            "n": int(sum(v is not None for v in values))}


def paired_bootstrap_pvalue(a: Sequence[Optional[float]], b: Sequence[Optional[float]],
                            n_boot: int = 5000, seed: int = 0) -> float:
    """
    Two-sided paired bootstrap p-value for mean(a) != mean(b) over items where
    both are defined.
    """
    pairs = np.array([(x, y) for x, y in zip(a, b) if x is not None and y is not None], dtype=float)
    if len(pairs) < 2:
        return float("nan")
    diff = pairs[:, 0] - pairs[:, 1]
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    centered = diff - observed
    boots = rng.choice(centered, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float((np.abs(boots) >= abs(observed)).mean())
