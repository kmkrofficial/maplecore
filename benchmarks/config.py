"""
MAPLE Performance Lab — Configuration
=======================================
Central constants for benchmarks, pricing, and paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results"

# Ensure directories exist at import time
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM Pricing ($ per 1M input tokens)
# ---------------------------------------------------------------------------
LLM_PRICING = {
    "gpt-4o": 5.00,
}

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
