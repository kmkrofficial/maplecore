#!/usr/bin/env python3
"""
Benchmark 04: Cost Analysis
=============================

Simulates 100 queries on a book-length document and compares:
  - Full Context: Send entire book to LLM every query
  - MAPLE: Send only Top-5 retrieved blocks + query

Calculates cumulative token cost using GPT-4o and GPT-4o-Mini pricing.

Output: results/cost_savings.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    RESULTS_DIR, PRICING, COST_NUM_QUERIES, CHUNK_SIZE,
)
from benchmarks.data_loader import download_gutenberg_book

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Approximate tokens per character (English text, GPT tokenizer)
CHARS_PER_TOKEN = 4.0
MAPLE_K = 5          # top-k blocks retrieved
QUERY_TOKENS = 50    # average query length in tokens


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return int(len(text) / CHARS_PER_TOKEN)


def run():
    """Run the cost analysis benchmark."""
    print("=" * 70)
    print("BENCHMARK 04: Cost Analysis")
    print("=" * 70)

    # ---- Load a book ----
    logger.info("Loading test document...")
    book_text = download_gutenberg_book(1661)  # Sherlock Holmes
    book_tokens = _estimate_tokens(book_text)
    logger.info(f"Document: {len(book_text):,} chars ~ {book_tokens:,} tokens")

    # MAPLE block budget per query
    maple_context_chars = MAPLE_K * CHUNK_SIZE
    maple_context_tokens = _estimate_tokens(str("." * maple_context_chars))

    results_all = {}

    for model_name, prices in PRICING.items():
        print(f"\n--- {model_name} ---")
        price_per_input_token = prices["input"] / 1_000_000

        full_context_costs = []
        maple_costs = []
        full_cumulative = []
        maple_cumulative = []

        for q in range(1, COST_NUM_QUERIES + 1):
            # Full context: send entire book + query each time
            full_input_tokens = book_tokens + QUERY_TOKENS
            full_cost = full_input_tokens * price_per_input_token
            full_context_costs.append(full_cost)
            full_cumulative.append(sum(full_context_costs))

            # MAPLE: send only top-k blocks + query
            maple_input_tokens = maple_context_tokens + QUERY_TOKENS
            maple_cost = maple_input_tokens * price_per_input_token
            maple_costs.append(maple_cost)
            maple_cumulative.append(sum(maple_costs))

        total_full = sum(full_context_costs)
        total_maple = sum(maple_costs)
        savings = total_full - total_maple
        savings_pct = (savings / total_full) * 100 if total_full > 0 else 0

        results_all[model_name] = {
            "model": model_name,
            "book_tokens": book_tokens,
            "maple_context_tokens": maple_context_tokens,
            "num_queries": COST_NUM_QUERIES,
            "full_context_total": round(total_full, 4),
            "maple_total": round(total_maple, 4),
            "savings": round(savings, 4),
            "savings_pct": round(savings_pct, 1),
            "full_context_cumulative": [round(x, 6) for x in full_cumulative],
            "maple_cumulative": [round(x, 6) for x in maple_cumulative],
        }

        print(f"  Full Context: ${total_full:.4f}")
        print(f"  MAPLE:        ${total_maple:.4f}")
        print(f"  Savings:      ${savings:.4f} ({savings_pct:.1f}%)")

    # ---- Save primary model results (GPT-4o) for plotting ----
    primary = results_all.get("gpt-4o", list(results_all.values())[0])
    output_path = RESULTS_DIR / "cost_savings.json"
    with open(output_path, "w") as f:
        json.dump(primary, f, indent=2)

    # Save all model results
    output_all = RESULTS_DIR / "cost_savings_all.json"
    with open(output_all, "w") as f:
        json.dump(results_all, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved -> {output_path}")
    print("=" * 70)

    return results_all


if __name__ == "__main__":
    run()
