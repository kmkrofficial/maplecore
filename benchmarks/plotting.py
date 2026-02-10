"""
Benchmark Plotting
===================
Matplotlib helpers for generating benchmark visualizations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _setup_style():
    """Apply consistent plot styling."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 10,
    })


def plot_recall_curve(results_path: Path, output_path: Optional[Path] = None) -> None:
    """
    Plot Recall@K comparison (MAPLE vs RAG).

    Args:
        results_path: Path to recall_curve.json
        output_path: Where to save the plot (PNG)
    """
    import matplotlib.pyplot as plt
    _setup_style()

    with open(results_path, "r") as f:
        data = json.load(f)

    k_values = data["k_values"]
    maple_recalls = data["maple_recalls"]
    rag_recalls = data["rag_recalls"]

    fig, ax = plt.subplots()
    ax.plot(k_values, maple_recalls, "o-", color="#e74c3c", linewidth=2,
            markersize=8, label="MAPLE (Adaptive)")
    ax.plot(k_values, rag_recalls, "s--", color="#3498db", linewidth=2,
            markersize=8, label="Standard RAG (Cosine)")

    ax.set_xlabel("K (Top-K Retrieved)")
    ax.set_ylabel("Recall@K (%)")
    ax.set_title("Recall@K — MAPLE vs Standard RAG (NarrativeQA)")
    ax.legend()
    ax.set_ylim(0, 100)

    output_path = output_path or results_path.parent / "recall_curve.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved recall curve -> {output_path}")


def plot_latency_scaling(results_path: Path, output_path: Optional[Path] = None) -> None:
    """
    Plot latency vs block count (log-log scale).

    Args:
        results_path: Path to latency_scaling.json
        output_path: Where to save the plot (PNG)
    """
    import matplotlib.pyplot as plt
    _setup_style()

    with open(results_path, "r") as f:
        data = json.load(f)

    block_counts = data["block_counts"]
    strategies = data["strategies"]

    colors = {"linear": "#e74c3c", "hierarchical": "#2ecc71", "adaptive": "#9b59b6"}
    markers = {"linear": "o", "hierarchical": "s", "adaptive": "D"}

    fig, ax = plt.subplots()

    for strategy_name, latencies in strategies.items():
        ax.loglog(
            block_counts, latencies,
            f"{markers.get(strategy_name, 'o')}-",
            color=colors.get(strategy_name, "#333"),
            linewidth=2, markersize=8,
            label=strategy_name.capitalize()
        )

    ax.set_xlabel("Number of Blocks")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Search Latency Scaling (Log-Log)")
    ax.legend()

    output_path = output_path or results_path.parent / "latency_scaling.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved latency scaling plot -> {output_path}")


def plot_needle_heatmap(results_path: Path, output_path: Optional[Path] = None) -> None:
    """
    Plot Needle-in-Haystack precision heatmap.

    Args:
        results_path: Path to needle_heatmap.json
        output_path: Where to save the plot (PNG)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_style()

    with open(results_path, "r") as f:
        data = json.load(f)

    depths = data["depths"]
    context_sizes = data["context_sizes"]
    heatmap = np.array(data["heatmap"])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"{d:.0%}" for d in depths])
    ax.set_yticks(range(len(context_sizes)))
    ax.set_yticklabels([f"{s:,}" for s in context_sizes])

    ax.set_xlabel("Needle Depth")
    ax.set_ylabel("Context Size (chars)")
    ax.set_title("Needle-in-Haystack Retrieval Accuracy")

    # Add text annotations
    for i in range(len(context_sizes)):
        for j in range(len(depths)):
            val = heatmap[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Retrieval Accuracy")

    output_path = output_path or results_path.parent / "needle_heatmap.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved needle heatmap -> {output_path}")


def plot_cost_savings(results_path: Path, output_path: Optional[Path] = None) -> None:
    """
    Plot cumulative cost comparison (Full Context vs MAPLE).

    Args:
        results_path: Path to cost_savings.json
        output_path: Where to save the plot (PNG)
    """
    import matplotlib.pyplot as plt
    _setup_style()

    with open(results_path, "r") as f:
        data = json.load(f)

    queries = list(range(1, len(data["full_context_cumulative"]) + 1))
    full_cost = data["full_context_cumulative"]
    maple_cost = data["maple_cumulative"]

    fig, ax = plt.subplots()

    ax.fill_between(queries, full_cost, maple_cost, alpha=0.2, color="#e74c3c",
                    label="Cost Saved")
    ax.plot(queries, full_cost, "-", color="#e74c3c", linewidth=2,
            label="Full Context")
    ax.plot(queries, maple_cost, "-", color="#2ecc71", linewidth=2,
            label="MAPLE (Top-5 Blocks)")

    ax.set_xlabel("Number of Queries")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title(f"Cost Comparison — {data.get('model', 'GPT-4o')}")
    ax.legend()

    output_path = output_path or results_path.parent / "cost_savings.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cost savings plot -> {output_path}")


def generate_all_plots(results_dir: Path) -> None:
    """Generate all plots from results directory."""
    recall_path = results_dir / "recall_curve.json"
    latency_path = results_dir / "latency_scaling.json"
    needle_path = results_dir / "needle_heatmap.json"
    cost_path = results_dir / "cost_savings.json"

    if recall_path.exists():
        plot_recall_curve(recall_path)
    if latency_path.exists():
        plot_latency_scaling(latency_path)
    if needle_path.exists():
        plot_needle_heatmap(needle_path)
    if cost_path.exists():
        plot_cost_savings(cost_path)
