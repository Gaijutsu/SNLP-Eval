"""Result reporting — aggregated CSV, per-instance JSON, summary tables."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResultStore:
    """Collects per-instance results and writes aggregated reports."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def store(
        self,
        instance_id: str,
        gatherer_name: str,
        metrics: dict[str, Any],
    ) -> None:
        """Store a single result record."""
        record = {
            "instance_id": instance_id,
            "gatherer": gatherer_name,
            **metrics,
        }
        self.records.append(record)

        # Also write per-instance JSON
        instance_dir = self.output_dir / "instances"
        instance_dir.mkdir(exist_ok=True)
        safe_id = instance_id.replace("/", "__").replace("\\", "__")
        fpath = instance_dir / f"{safe_id}_{gatherer_name}.json"
        fpath.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    def generate_report(self) -> Path:
        """Write aggregated CSV and summary to output_dir."""
        if not self.records:
            logger.warning("No results to report.")
            return self.output_dir

        # ── Aggregated CSV ──
        csv_path = self.output_dir / "results.csv"
        all_keys = sorted({k for r in self.records for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        logger.info("Wrote results CSV to %s", csv_path)

        # ── Summary table (mean ± std per gatherer) ──
        summary = self._compute_summary()
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

        # Print to console
        self._print_summary(summary)

        return self.output_dir

    def _compute_summary(self) -> dict[str, Any]:
        """Compute mean ± std for each metric, grouped by gatherer."""
        import numpy as np

        grouped: dict[str, list[dict]] = defaultdict(list)
        for r in self.records:
            grouped[r["gatherer"]].append(r)

        summary: dict[str, Any] = {}
        numeric_keys = set()
        for records in grouped.values():
            for r in records:
                for k, v in r.items():
                    if isinstance(v, (int, float)) and k not in ("instance_id",):
                        numeric_keys.add(k)

        for gatherer, records in grouped.items():
            stats = {}
            for key in sorted(numeric_keys):
                values = [r.get(key) for r in records if r.get(key) is not None]
                if values:
                    values = [float(v) for v in values]
                    stats[key] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std": round(float(np.std(values)), 4),
                        "min": round(float(np.min(values)), 4),
                        "max": round(float(np.max(values)), 4),
                        "n": len(values),
                    }
            summary[gatherer] = stats

        return summary

    def _print_summary(self, summary: dict[str, Any]) -> None:
        """Print a formatted summary table using rich."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            for gatherer, stats in summary.items():
                table = Table(title=f"📊 {gatherer}", show_lines=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Mean", style="green", justify="right")
                table.add_column("Std", style="yellow", justify="right")
                table.add_column("N", justify="right")

                for metric, values in sorted(stats.items()):
                    table.add_row(
                        metric,
                        f"{values['mean']:.4f}",
                        f"{values['std']:.4f}",
                        str(values["n"]),
                    )

                console.print(table)
                console.print()

        except ImportError:
            # Fallback without rich
            for gatherer, stats in summary.items():
                print(f"\n{'='*60}")
                print(f"  {gatherer}")
                print(f"{'='*60}")
                for metric, values in sorted(stats.items()):
                    print(f"  {metric:30s}  mean={values['mean']:.4f}  std={values['std']:.4f}")
