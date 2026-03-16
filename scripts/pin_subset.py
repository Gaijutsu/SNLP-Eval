#!/usr/bin/env python3
"""Pin the task-ID subset after a successful experiment run.

Reads instance IDs from a results CSV and writes them to a JSON file
for reproducible subsequent experiments.

Usage:
    python scripts/pin_subset.py results/baseline_bm25/results.csv data/subset_30.json
"""

import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/pin_subset.py <results.csv> <output.json>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist")
        sys.exit(1)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ids = sorted(set(row["instance_id"] for row in reader))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)

    print(f"✅ Pinned {len(ids)} instance IDs → {output_path}")
    print(f"   First 5: {ids[:5]}")


if __name__ == "__main__":
    main()
