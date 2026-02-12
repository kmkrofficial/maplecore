#!/usr/bin/env python3
"""
Demo: Infinite Context with MAPLE
===================================

This example demonstrates how MAPLE handles a book-length document
and performs precise retrieval using learned attention patterns
(Memory-Aware Predictive Loading).
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from maplecore import Maple


def main():
    print("="*70)
    print("MAPLE: Infinite Context Demo")
    print("="*70)
    
    # Initialize client
    client = Maple(
        model_path="maple.pth",
        device="cuda",
        strategy="adaptive"
    )
    
    # Download Sherlock Holmes if not present
    book_path = Path("sherlock_holmes.txt")
    if not book_path.exists():
        print("\nDownloading 'The Adventures of Sherlock Holmes'...")
        import requests
        url = "https://www.gutenberg.org/files/1661/1661-0.txt"
        response = requests.get(url, timeout=30)
        book_path.write_text(response.text, encoding="utf-8")
    
    # Index the book
    print(f"\n[1] Indexing {book_path}...")
    index = client.index_file(book_path)
    print(f"    Created {index.num_blocks} blocks")
    
    # Example queries
    queries = [
        "What was the speckled band?",
        "Who is Dr. Watson?",
        "Describe Sherlock Holmes' methods of deduction"
    ]
    
    print("\n[2] Running queries...\n")
    
    for query in queries:
        print(f"Query: \"{query}\"")
        
        result = client.query(query, k=5)
        
        print(f"  Strategy: {result.strategy_used}")
        print(f"  Latency:  {result.latency_ms:.2f} ms")
        print(f"  Blocks:   {result.block_ids[:5]}")
        
        # Show preview of top block
        top_block = client.get_block(result.block_ids[0])
        preview = top_block.text[:100].replace("\n", " ")
        print(f"  Preview:  \"{preview}...\"")
        print()
    
    print("="*70)
    print("Done! MAPLE successfully handled infinite context retrieval.")
    print("="*70)


if __name__ == "__main__":
    main()
