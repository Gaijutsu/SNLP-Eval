#!/usr/bin/env python3
"""Extract a stable, reproducible N=30 task subset from SWE-bench Lite.

Owned by RAG subteam – Sara (Dataset Plumbing + Ground Truth Extraction)

This script:
  1. Loads SWE-bench Lite (test split) from HuggingFace.
  2. Extracts per-task ground truth: task_id, repo_id, query_text, gold_files.
  3. Selects N=30 tasks using a diversity-aware strategy (spread across repos,
     varying gold-file counts) with a fixed random seed for reproducibility.
  4. Saves:
       - data/subset_30.json          — pinned list of 30 instance IDs
       - data/subset_30_full.json     — full dataset artifact (used by benchmark adapter)
       - data/sanity_stats.json       — sanity statistics
       - data/sanity_stats_report.txt — human-readable sanity report
  5. Prints sanity stats to stdout.

Usage:
    python scripts/extract_subset.py [--n 30] [--seed 42] [--output-dir data]

Requirements:
    pip install datasets
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


def extract_files_from_patch(patch: str) -> list[str]:
    """Parse a unified diff to extract the list of modified file paths.

    Mirrors SWEBenchAdapter._extract_files_from_patch exactly to ensure
    consistency with the harness evaluation.
    """
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                if path not in files:
                    files.append(path)
    return files


def load_swebench_lite(split: str = "test") -> list[dict]:
    """Load all instances from SWE-bench Lite."""
    os.environ["DATASETS_NO_TORCH"] = "1"
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    return list(ds)


def extract_ground_truth(row: dict) -> dict:
    """Extract the ground-truth fields we need from a single HF row."""
    instance_id: str = row["instance_id"]
    repo: str = row["repo"]
    patch: str = row.get("patch", "")
    problem_statement: str = row.get("problem_statement", "")
    hints_text: str = row.get("hints_text", "")
    base_commit: str = row.get("base_commit", "")

    gold_files = extract_files_from_patch(patch)

    # Build query text the same way the SWE-bench adapter does
    query = problem_statement
    if hints_text:
        query += f"\n\nHints:\n{hints_text}"

    return {
        "task_id": instance_id,
        "repo_id": repo,
        "base_commit": base_commit,
        "query_text": query,
        "gold_files": gold_files,
        "num_gold_files": len(gold_files),
    }


def select_diverse_subset(
    all_tasks: list[dict],
    n: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Select N tasks with diversity across repos and gold-file counts.

    Strategy:
      1. Group tasks by repo.
      2. Sample proportionally from each repo (so no single repo dominates).
      3. Within each repo, prefer a spread of gold-file counts.
      4. Use a fixed seed for reproducibility.
    """
    import random

    rng = random.Random(seed)

    # Group by repo
    by_repo: dict[str, list[dict]] = {}
    for t in all_tasks:
        by_repo.setdefault(t["repo_id"], []).append(t)

    # Sort repos by size (descending) for deterministic ordering
    repos_sorted = sorted(by_repo.keys(), key=lambda r: (-len(by_repo[r]), r))

    # Proportional allocation: each repo gets floor(n * repo_size / total)
    total = len(all_tasks)
    allocation: dict[str, int] = {}
    assigned = 0
    for repo in repos_sorted:
        share = max(1, int(n * len(by_repo[repo]) / total))
        allocation[repo] = share
        assigned += share

    # Distribute remaining slots round-robin to largest repos
    remaining = n - assigned
    idx = 0
    while remaining > 0:
        repo = repos_sorted[idx % len(repos_sorted)]
        if allocation[repo] < len(by_repo[repo]):
            allocation[repo] += 1
            remaining -= 1
        idx += 1
        if idx > len(repos_sorted) * 2:
            break  # safety valve

    # If we over-allocated, trim from smallest repos
    while sum(allocation.values()) > n:
        for repo in reversed(repos_sorted):
            if allocation[repo] > 1:
                allocation[repo] -= 1
                break

    # Sample from each repo
    selected: list[dict] = []
    for repo in repos_sorted:
        pool = by_repo[repo]
        k = min(allocation.get(repo, 0), len(pool))
        if k == 0:
            continue
        # Sort pool by num_gold_files to get spread, then sample evenly
        pool_sorted = sorted(pool, key=lambda t: t["num_gold_files"])
        if k >= len(pool_sorted):
            selected.extend(pool_sorted[:k])
        else:
            # Take evenly spaced items for diversity in gold-file counts
            step = len(pool_sorted) / k
            indices = [int(i * step) for i in range(k)]
            selected.extend(pool_sorted[i] for i in indices)

    # If we still don't have enough, fill randomly from remaining
    selected_ids = {t["task_id"] for t in selected}
    remaining_pool = [t for t in all_tasks if t["task_id"] not in selected_ids]
    rng.shuffle(remaining_pool)
    while len(selected) < n and remaining_pool:
        selected.append(remaining_pool.pop())

    # Trim if over
    selected = selected[:n]

    # Sort by task_id for stable ordering
    selected.sort(key=lambda t: t["task_id"])
    return selected


def compute_sanity_stats(tasks: list[dict]) -> dict:
    """Compute sanity statistics over the selected subset."""
    gold_counts = [t["num_gold_files"] for t in tasks]
    repo_counts = Counter(t["repo_id"] for t in tasks)

    # Distribution of gold file counts
    gold_dist = Counter(gold_counts)

    avg_gold = sum(gold_counts) / len(gold_counts) if gold_counts else 0
    min_gold = min(gold_counts) if gold_counts else 0
    max_gold = max(gold_counts) if gold_counts else 0
    median_gold = sorted(gold_counts)[len(gold_counts) // 2] if gold_counts else 0

    # Parsing caveats
    caveats = []
    zero_gold = [t["task_id"] for t in tasks if t["num_gold_files"] == 0]
    if zero_gold:
        caveats.append(
            f"{len(zero_gold)} task(s) have 0 gold files (empty patch?): "
            f"{zero_gold[:5]}"
        )

    large_gold = [t for t in tasks if t["num_gold_files"] > 10]
    if large_gold:
        caveats.append(
            f"{len(large_gold)} task(s) touch >10 files — may be noisy for "
            f"file-level retrieval: "
            f"{[t['task_id'] for t in large_gold[:5]]}"
        )

    # Check for path normalisation concerns
    all_gold_files = [f for t in tasks for f in t["gold_files"]]
    dotslash = [f for f in all_gold_files if f.startswith("./")]
    leading_slash = [f for f in all_gold_files if f.startswith("/")]
    if dotslash:
        caveats.append(
            f"{len(dotslash)} gold file path(s) have './' prefix — "
            f"verify normalisation against retriever output"
        )
    if leading_slash:
        caveats.append(
            f"{len(leading_slash)} gold file path(s) have leading '/' — "
            f"verify normalisation against retriever output"
        )
    if not caveats:
        caveats.append("No parsing caveats detected — paths look clean.")

    return {
        "n_tasks": len(tasks),
        "n_unique_repos": len(repo_counts),
        "repos": dict(repo_counts.most_common()),
        "gold_files": {
            "mean": round(avg_gold, 2),
            "median": median_gold,
            "min": min_gold,
            "max": max_gold,
            "std": (
                round(
                    (sum((x - avg_gold) ** 2 for x in gold_counts) / len(gold_counts))
                    ** 0.5,
                    2,
                )
                if gold_counts
                else 0
            ),
            "distribution": {str(k): v for k, v in sorted(gold_dist.items())},
        },
        "caveats": caveats,
    }


def format_stats_report(stats: dict, tasks: list[dict]) -> str:
    """Format a human-readable sanity stats report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SANITY STATS — SWE-bench Lite Task Subset (Sara's Deliverable)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total tasks:       {stats['n_tasks']}")
    lines.append(f"Unique repos:      {stats['n_unique_repos']}")
    lines.append("")
    lines.append("--- Gold File Statistics ---")
    gf = stats["gold_files"]
    lines.append(f"  Mean:   {gf['mean']}")
    lines.append(f"  Median: {gf['median']}")
    lines.append(f"  Min:    {gf['min']}")
    lines.append(f"  Max:    {gf['max']}")
    lines.append(f"  Std:    {gf['std']}")
    lines.append("")
    lines.append("--- Gold File Count Distribution ---")
    for count, freq in sorted(gf["distribution"].items(), key=lambda x: int(x[0])):
        bar = "#" * freq
        lines.append(f"  {count:>3} files: {freq:>3} tasks  {bar}")
    lines.append("")
    lines.append("--- Repo Distribution ---")
    for repo, count in stats["repos"].items():
        lines.append(f"  {repo:<40s} {count:>3} tasks")
    lines.append("")
    lines.append("--- Parsing Caveats ---")
    for c in stats["caveats"]:
        lines.append(f"  ⚠ {c}")
    lines.append("")
    lines.append("--- Task Summary (first 10) ---")
    lines.append(f"  {'task_id':<50s} {'repo':<30s} #gold")
    lines.append(f"  {'-'*50} {'-'*30} -----")
    for t in tasks[:10]:
        lines.append(
            f"  {t['task_id']:<50s} {t['repo_id']:<30s} {t['num_gold_files']:>5}"
        )
    if len(tasks) > 10:
        lines.append(f"  ... ({len(tasks) - 10} more)")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract a stable N-task subset from SWE-bench Lite"
    )
    parser.add_argument(
        "--n", type=int, default=30, help="Number of tasks to select (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SWE-bench Lite ({args.split} split)...")
    rows = load_swebench_lite(split=args.split)
    print(f"  Loaded {len(rows)} instances.")

    print("Extracting ground truth...")
    all_tasks = [extract_ground_truth(row) for row in rows]

    # Filter out any tasks with 0 gold files (unusable for retrieval eval)
    valid_tasks = [t for t in all_tasks if t["num_gold_files"] > 0]
    skipped = len(all_tasks) - len(valid_tasks)
    if skipped:
        print(f"  ⚠ Skipped {skipped} tasks with 0 gold files.")
    print(f"  {len(valid_tasks)} valid tasks available.")

    print(f"Selecting diverse subset of N={args.n}...")
    subset = select_diverse_subset(valid_tasks, n=args.n, seed=args.seed)
    print(
        f"  Selected {len(subset)} tasks across "
        f"{len(set(t['repo_id'] for t in subset))} repos."
    )

    # --- Save outputs ---

    # 1. Pinned ID list (consumed by runner via task_ids_file)
    id_list = [t["task_id"] for t in subset]
    ids_path = output_dir / f"subset_{args.n}.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(id_list, f, indent=2)
    print(f"  ✅ Pinned IDs      → {ids_path}")

    # 2. Full dataset artifact (task_id, repo_id, query_text, gold_files)
    artifact_path = output_dir / f"subset_{args.n}_full.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Full artifact   → {artifact_path}")

    # 3. Sanity stats
    stats = compute_sanity_stats(subset)
    stats_path = output_dir / "sanity_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ Stats (JSON)    → {stats_path}")

    report = format_stats_report(stats, subset)
    report_path = output_dir / "sanity_stats_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✅ Stats (report)  → {report_path}")

    # Print report to stdout
    print()
    print(report)


if __name__ == "__main__":
    main()
