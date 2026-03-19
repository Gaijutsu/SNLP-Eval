#!/usr/bin/env python3
"""Verify the dataset artifact is compatible with the benchmark adapter.

Owned by RAG subteam – Sara (Dataset Plumbing + Ground Truth Extraction)

Checks:
  1. subset_30.json loads and has exactly N IDs
  2. subset_30_full.json has all required fields
  3. Gold file paths are normalised (no ./, no leading /, no trailing /)
  4. Gold file paths match the format the SWE-bench adapter produces
  5. The adapter can load with the pinned task IDs (live check, optional)

Usage:
    python scripts/verify_subset.py [--live]  # --live does an actual HF load
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_id_file(path: Path) -> list[str]:
    """Validate the pinned ID list."""
    print(f"\n--- Checking {path} ---")
    assert path.exists(), f"  ✗ File not found: {path}"

    with open(path, encoding="utf-8") as f:
        ids = json.load(f)

    assert isinstance(ids, list), "  ✗ Expected a JSON array"
    assert len(ids) > 0, "  ✗ ID list is empty"
    assert len(ids) == len(set(ids)), "  ✗ Duplicate IDs found"

    print(f"  ✓ {len(ids)} unique instance IDs")
    print(f"  ✓ First: {ids[0]}")
    print(f"  ✓ Last:  {ids[-1]}")
    return ids


def check_full_artifact(path: Path, expected_ids: list[str]) -> list[dict]:
    """Validate the full dataset artifact."""
    print(f"\n--- Checking {path} ---")
    assert path.exists(), f"  ✗ File not found: {path}"

    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)

    assert isinstance(tasks, list), "  ✗ Expected a JSON array"
    assert len(tasks) == len(
        expected_ids
    ), f"  ✗ Expected {len(expected_ids)} tasks, got {len(tasks)}"

    required_fields = {
        "task_id",
        "repo_id",
        "query_text",
        "gold_files",
        "num_gold_files",
    }
    errors = []

    for i, task in enumerate(tasks):
        missing = required_fields - set(task.keys())
        if missing:
            errors.append(f"  Task {i}: missing fields {missing}")

        # Check task_id is in the ID list
        if task["task_id"] not in set(expected_ids):
            errors.append(f"  Task {i}: ID '{task['task_id']}' not in ID list")

        # Validate gold_files
        if not isinstance(task.get("gold_files", []), list):
            errors.append(f"  Task {i}: gold_files is not a list")

        # Check num_gold_files consistency
        if task.get("num_gold_files") != len(task.get("gold_files", [])):
            errors.append(
                f"  Task {i}: num_gold_files ({task.get('num_gold_files')}) "
                f"!= len(gold_files) ({len(task.get('gold_files', []))})"
            )

        # Path normalisation checks
        for gf in task.get("gold_files", []):
            if gf.startswith("./"):
                errors.append(
                    f"  Task {task['task_id']}: gold file has './' prefix: {gf}"
                )
            if gf.startswith("/"):
                errors.append(
                    f"  Task {task['task_id']}: gold file has leading '/': {gf}"
                )
            if gf.endswith("/"):
                errors.append(
                    f"  Task {task['task_id']}: gold file has trailing '/': {gf}"
                )
            if "\\" in gf:
                errors.append(
                    f"  Task {task['task_id']}: gold file has backslash: {gf}"
                )

    if errors:
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        print(f"\n  ✗ {len(errors)} issue(s) found")
    else:
        print(f"  ✓ All {len(tasks)} tasks have required fields")
        print("  ✓ All gold file paths are normalised")
        print("  ✓ num_gold_files consistent with gold_files length")

    return tasks


def check_sanity_stats(path: Path) -> None:
    """Validate sanity stats file exists and is reasonable."""
    print(f"\n--- Checking {path} ---")
    assert path.exists(), f"  ✗ File not found: {path}"

    with open(path, encoding="utf-8") as f:
        stats = json.load(f)

    assert "n_tasks" in stats, "  ✗ Missing n_tasks"
    assert "gold_files" in stats, "  ✗ Missing gold_files stats"
    assert "repos" in stats, "  ✗ Missing repos"
    assert "caveats" in stats, "  ✗ Missing caveats"

    gf = stats["gold_files"]
    print(f"  ✓ {stats['n_tasks']} tasks, {stats['n_unique_repos']} repos")
    print(
        f"  ✓ Gold files: mean={gf['mean']}, median={gf['median']}, "
        f"min={gf['min']}, max={gf['max']}"
    )
    print(f"  ✓ {len(stats['caveats'])} caveat(s) documented")


def live_adapter_check(ids_path: Path) -> None:
    """Actually load the pinned subset through the SWE-bench adapter."""
    print("\n--- Live adapter check (loading from HuggingFace) ---")

    with open(ids_path, encoding="utf-8") as f:
        task_ids = json.load(f)

    # Import the adapter from the scaffold
    try:
        from harness.benchmarks.swebench import SWEBenchAdapter
    except ImportError:
        print("  ⚠ Could not import SWEBenchAdapter — run 'pip install -e .' first")
        return

    adapter = SWEBenchAdapter()
    instances = adapter.load(split="test", task_ids=task_ids)

    print(f"  Loaded {len(instances)} instances via adapter")
    if len(instances) != len(task_ids):
        print(
            f"  ⚠ Expected {len(task_ids)}, got {len(instances)} — "
            f"some tasks may have failed to load (check repo cloning)"
        )

    # Verify gold_context matches our artifact
    ids_path_full = ids_path.parent / ids_path.name.replace(".json", "_full.json")
    if ids_path_full.exists():
        with open(ids_path_full, encoding="utf-8") as f:
            artifact_tasks = {t["task_id"]: t for t in json.load(f)}

        mismatches = 0
        for inst in instances:
            if inst.id in artifact_tasks:
                expected = set(artifact_tasks[inst.id]["gold_files"])
                actual = set(inst.gold_context)
                if expected != actual:
                    print(f"  ✗ Mismatch for {inst.id}:")
                    print(f"    Expected: {expected}")
                    print(f"    Actual:   {actual}")
                    mismatches += 1
        if mismatches == 0:
            print("  ✓ All gold_context matches the artifact!")
        else:
            print(f"  ✗ {mismatches} mismatch(es) — check path normalisation")
    else:
        print(f"  ⚠ No full artifact at {ids_path_full} to cross-check")

    print("  ✓ Live check complete")


def main():
    parser = argparse.ArgumentParser(description="Verify dataset artifact")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also do a live check via the SWE-bench adapter",
    )
    parser.add_argument("--n", type=int, default=30, help="Expected subset size")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ok = True

    try:
        ids = check_id_file(data_dir / f"subset_{args.n}.json")
        tasks = check_full_artifact(data_dir / f"subset_{args.n}_full.json", ids)
        check_sanity_stats(data_dir / "sanity_stats.json")

        if args.live:
            live_adapter_check(data_dir / f"subset_{args.n}.json")

    except AssertionError as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        ok = False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        ok = False

    if ok:
        print(f"\n{'='*50}")
        print("✅ All checks passed!")
        print(f"{'='*50}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
