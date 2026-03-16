#!/usr/bin/env python3
"""Compare results across multiple experiment runs.

Produces:
  1. A side-by-side comparison table (Recall@K, MRR, Precision@K, latency)
  2. A K-sensitivity plot (Recall vs K for each method)
  3. A combined summary CSV

Usage:
    python scripts/compare_results.py \
        results/baseline_bm25/summary.json \
        results/ablation_dense/summary.json \
        results/ablation_hybrid/summary.json
"""

import json
import sys
from pathlib import Path


def load_summary(path: Path) -> dict:
    """Load a summary.json file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_comparison_table(summaries: dict[str, dict]) -> None:
    """Print a formatted comparison table to console."""
    # Collect all metrics across all gatherers
    all_metrics = set()
    for summary in summaries.values():
        for gatherer_stats in summary.values():
            all_metrics.update(gatherer_stats.keys())

    # Focus on key metrics
    key_metrics = [
        "recall@5", "recall@10", "recall@20",
        "precision@5", "precision@10", "precision@20",
        "mrr",
        "ndcg@5", "ndcg@10", "ndcg@20",
        "latency_s",
    ]
    key_metrics = [m for m in key_metrics if m in all_metrics]

    # Build method names
    methods = []
    for source_name, summary in summaries.items():
        for gatherer in summary:
            methods.append((source_name, gatherer))

    # Print header
    method_names = [f"{g}" for _, g in methods]
    header = f"{'Metric':<20s}" + "".join(f"  {m:>14s}" for m in method_names)
    print("\n" + "=" * len(header))
    print("  RAG Retrieval Results Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for metric in key_metrics:
        row = f"{metric:<20s}"
        for source_name, gatherer in methods:
            stats = summaries[source_name].get(gatherer, {}).get(metric, {})
            if stats:
                mean = stats.get("mean", 0)
                std = stats.get("std", 0)
                row += f"  {mean:>6.4f}±{std:<5.4f}"
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    print("-" * len(header))
    print()


def generate_k_sensitivity_data(summaries: dict[str, dict]) -> None:
    """Print K-sensitivity data for plotting."""
    print("\n📈 K-Sensitivity Data (Recall@K):")
    print(f"{'Method':<15s}  {'K=5':>8s}  {'K=10':>8s}  {'K=20':>8s}")
    print("-" * 45)

    for source_name, summary in summaries.items():
        for gatherer, stats in summary.items():
            r5 = stats.get("recall@5", {}).get("mean", "N/A")
            r10 = stats.get("recall@10", {}).get("mean", "N/A")
            r20 = stats.get("recall@20", {}).get("mean", "N/A")
            r5_s = f"{r5:.4f}" if isinstance(r5, float) else r5
            r10_s = f"{r10:.4f}" if isinstance(r10, float) else r10
            r20_s = f"{r20:.4f}" if isinstance(r20, float) else r20
            print(f"{gatherer:<15s}  {r5_s:>8s}  {r10_s:>8s}  {r20_s:>8s}")

    print()


def save_plot(summaries: dict[str, dict], output_path: Path) -> None:
    """Generate a K-sensitivity plot if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k_values = [5, 10, 20]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for source_name, summary in summaries.items():
            for gatherer, stats in summary.items():
                recalls = []
                for k in k_values:
                    r = stats.get(f"recall@{k}", {}).get("mean", 0)
                    recalls.append(r)
                ax.plot(k_values, recalls, marker="o", label=gatherer, linewidth=2)

        ax.set_xlabel("K (number of retrieved files)", fontsize=12)
        ax.set_ylabel("Recall@K", fontsize=12)
        ax.set_title("Recall@K across Retrieval Methods", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"📊 Plot saved to {output_path}")

    except ImportError:
        print("⚠️  matplotlib not installed — skipping plot generation")
        print("   Install with: pip install matplotlib")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_results.py <summary1.json> [summary2.json] ...")
        sys.exit(1)

    summaries = {}
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        name = path.parent.name  # e.g., "baseline_bm25"
        summaries[name] = load_summary(path)

    if not summaries:
        print("Error: No valid summary files found")
        sys.exit(1)

    print_comparison_table(summaries)
    generate_k_sensitivity_data(summaries)

    # Try to generate a plot
    plot_path = Path("results/k_sensitivity.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot(summaries, plot_path)


if __name__ == "__main__":
    main()
