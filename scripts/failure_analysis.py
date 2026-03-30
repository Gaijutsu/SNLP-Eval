#!/usr/bin/env python3
"""Failure analysis for RAG retrieval experiments.

Identifies instances where retrieval completely failed (Recall@K = 0)
and categorizes failure modes.

Usage:
    python scripts/failure_analysis.py results/baseline_bm25/
"""

import csv
import json
import sys
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    """Load results from a results directory."""
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_instance_details(results_dir: Path, instance_id: str, gatherer: str) -> dict | None:
    """Load per-instance JSON for detailed analysis."""
    safe_id = instance_id.replace("/", "__").replace("\\", "__")
    json_path = results_dir / "instances" / f"{safe_id}_{gatherer}.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def analyze_failures(results: list[dict], results_dir: Path, k: int = 10) -> None:
    """Find and categorize instances with Recall@K = 0."""
    recall_key = f"recall@{k}"

    failures = []
    successes = []
    for row in results:
        try:
            recall = float(row.get(recall_key, 0))
        except (ValueError, TypeError):
            recall = 0.0

        if recall == 0.0:
            failures.append(row)
        else:
            successes.append(row)

    total = len(results)
    n_fail = len(failures)
    n_success = len(successes)

    print(f"\n{'=' * 70}")
    print(f"  Failure Analysis — Recall@{k}")
    print(f"{'=' * 70}")
    print(f"  Total instances: {total}")
    print(f"  Recall@{k} = 0 (complete failures): {n_fail} ({100*n_fail/max(total,1):.1f}%)")
    print(f"  Recall@{k} > 0 (at least partial): {n_success} ({100*n_success/max(total,1):.1f}%)")
    print()

    if not failures:
        print("  🎉 No complete failures found!")
        return

    # Analyze each failure
    print(f"  {'Instance ID':<50s}  {'Gatherer':<15s}  {'#Gold':<6s}")
    print(f"  {'-'*50}  {'-'*15}  {'-'*6}")

    failure_categories = {
        "multi_file": 0,
        "single_file": 0,
        "unknown_gold_count": 0,
    }

    for row in failures[:20]:  # Cap at 20 for readability
        instance_id = row.get("instance_id", "?")
        gatherer = row.get("gatherer", "?")

        # Try to load instance details
        detail = load_instance_details(results_dir, instance_id, gatherer)

        gold_count = "?"
        if detail:
            # Instance JSONs don't store gold_context directly,
            # but we can infer from other metrics
            pass

        print(f"  {instance_id:<50s}  {gatherer:<15s}  {gold_count:<6s}")

    if n_fail > 20:
        print(f"  ... and {n_fail - 20} more failures")

    print()
    print("  📋 Failure Mode Checklist:")
    print("  ─────────────────────────")
    print("  [ ] Path mismatch: retrieved paths vs gold paths not normalized")
    print("  [ ] Query too vague: problem statement lacks code-specific terms")
    print("  [ ] Gold file not indexed: test files, config files, non-code files")
    print("  [ ] Gold file unusual naming: names don't match query terms")
    print("  [ ] Large repo: too many files dilute BM25 scores")
    print()

    # Distribution analysis
    print("  📊 Recall Distribution:")
    recall_values = []
    for row in results:
        try:
            recall_values.append(float(row.get(recall_key, 0)))
        except (ValueError, TypeError):
            recall_values.append(0.0)

    bins = [(0, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.01)]
    labels = ["= 0.0", "(0, 0.25]", "(0.25, 0.5]", "(0.5, 0.75]", "(0.75, 1.0)", "= 1.0"]

    for (lo, hi), label in zip(bins, labels):
        if label == "= 0.0":
            count = sum(1 for v in recall_values if v == 0.0)
        elif label == "= 1.0":
            count = sum(1 for v in recall_values if v == 1.0)
        else:
            count = sum(1 for v in recall_values if lo < v <= hi)
        bar = "█" * count
        print(f"  {label:>14s}  {count:3d}  {bar}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/failure_analysis.py <results_dir/> [K]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    results = load_results(results_dir)
    analyze_failures(results, results_dir, k=k)


if __name__ == "__main__":
    main()
