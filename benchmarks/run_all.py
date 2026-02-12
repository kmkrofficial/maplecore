#!/usr/bin/env python3
"""
MAPLE Benchmark Runner
=======================

Master runner that executes all benchmarks and generates plots.

Usage:
    python -m benchmarks.run_all              # Run everything
    python -m benchmarks.run_all --only 01    # Run specific benchmark
    python -m benchmarks.run_all --plots-only # Only generate plots
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BENCHMARKS = {
    "01": ("Recall@K (NarrativeQA)", "benchmarks.01_recall_narrativeqa"),
    "02": ("Latency Scaling", "benchmarks.02_latency_scaling"),
    "03": ("Needle in a Haystack", "benchmarks.03_needle_in_haystack"),
    "04": ("Cost Analysis", "benchmarks.04_cost_analysis"),
}


def run_benchmark(key: str) -> None:
    """Import and run a single benchmark."""
    name, module_path = BENCHMARKS[key]
    print(f"\n{'#' * 70}")
    print(f"# BENCHMARK {key}: {name}")
    print(f"{'#' * 70}\n")

    import importlib
    module = importlib.import_module(module_path)

    start = time.perf_counter()
    module.run()
    elapsed = time.perf_counter() - start

    print(f"\n[*] Completed in {elapsed:.1f}s\n")


def generate_plots() -> None:
    """Generate all available plots."""
    print(f"\n{'#' * 70}")
    print("# GENERATING PLOTS")
    print(f"{'#' * 70}\n")

    from benchmarks.plotting import generate_all_plots
    generate_all_plots(RESULTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="MAPLE Benchmark Suite")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only a specific benchmark (e.g., '01', '02', '03', '04')"
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Only generate plots from existing results"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MAPLE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")
    print()

    overall_start = time.perf_counter()

    if args.plots_only:
        generate_plots()
    elif args.only:
        if args.only not in BENCHMARKS:
            print(f"Unknown benchmark: {args.only}")
            print(f"Available: {', '.join(BENCHMARKS.keys())}")
            sys.exit(1)
        run_benchmark(args.only)
        generate_plots()
    else:
        for key in sorted(BENCHMARKS.keys()):
            try:
                run_benchmark(key)
            except Exception as e:
                logger.error(f"Benchmark {key} failed: {e}")
                continue

        generate_plots()

    total_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 70)
    print(f"MAPLE: ALL BENCHMARKS COMPLETE -- {total_elapsed:.1f}s total")
    print(f"   Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
