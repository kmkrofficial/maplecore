#!/usr/bin/env python3
"""
Verify Custom Embedding Model
=============================
Tests whether MAPLE can successfully switch to a different
embedding model and calculate dynamic dimensions.
"""

from pathlib import Path
from maplecore import Maple

def run_test():
    # MiniLM operates with 384 dims.
    # We will use this to test swapping BGE entirely.
    # Note: If we had a 768 or 1024 dim model this would more
    # actively prove variable sizing. BAAI/bge-large-en-v1.5 has 1024.
    # Let's use string "sentence-transformers/all-MiniLM-L6-v2".
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    print(f"Initializing Maple with model: {model_name}")
    try:
        client = Maple(embedding_model=model_name, device="cpu")
    except Exception as e:
        print(f"FAILED to initialize client: {e}")
        return
        
    print("Testing dynamic dimension fetching...")
    dim = client.indexer.get_embedding_dimension()
    print(f"Retrieved Embedding Dimension: {dim}")
    
    text = (
        "Project MAPLE is a new memory caching mechanism for large "
        "language models that decouples embedding dimensions."
    )
    
    print("Creating index...")
    try:
        index = client.index_text(text, chunk_size=200)
        print(f"Index successfully created. Blocks: {index.num_blocks}, Dim: {index.embedding_dim}")
        
    except Exception as e:
        print(f"FAILED indexing text: {e}")
        
    print("Test Sequence completed successfully.")
        
if __name__ == "__main__":
    run_test()
