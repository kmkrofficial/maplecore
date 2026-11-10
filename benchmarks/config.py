"""
MAPLE Performance Lab — Configuration
=======================================
Central constants for benchmarks, pricing, and paths.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR.parent / "data"  # Root data directory
RESULTS_DIR = BENCHMARK_DIR / "results"

# Ensure directories exist at import time
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model & Device
# ---------------------------------------------------------------------------
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = BENCHMARK_DIR.parent / "models" / "maple_generalist.pth"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500  # characters per block
MAX_TOKENS = 1536 # max tokens for oracle context

# Scaling counts for latency benchmark
SCALING_BLOCK_COUNTS = [10_000, 50_000, 100_000, 500_000, 1_000_000]

# Recall K values
RECALL_K_VALUES = [1, 5, 10]

# Needle benchmark settings
NEEDLE_DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
NEEDLE_CONTEXT_SIZES = [5_000, 10_000, 25_000]

# ---------------------------------------------------------------------------
# LLM Pricing — ILLUSTRATIVE ONLY. Edit to current list prices before quoting
# any dollar figure. The cost benchmark's primary output is token reduction,
# which does not depend on these numbers.
# ($ per 1M input tokens)
# ---------------------------------------------------------------------------
PRICING = {
    "example-large-model": {"input": 2.50},
    "example-small-model": {"input": 0.15},
}
LLM_PRICING = {k: v["input"] for k, v in PRICING.items()}  # backwards compatibility
COST_NUM_QUERIES = 100

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
