"""Thread-safe experiment state shared between the runner and the dashboard."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GathererStats:
    """Running statistics for a single gatherer."""

    instances_completed: int = 0
    total_tokens: int = 0
    total_latency_s: float = 0.0

    # Running sums for computing means
    sum_precision: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    sum_recall: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    sum_ndcg: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    sum_mrr: float = 0.0

    # Per-instance metric history (for live charts)
    metric_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return self.total_latency_s / max(self.instances_completed, 1)

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / max(self.instances_completed, 1)

    def avg_metric(self, metric: str, k: int) -> float:
        store = {
            "precision": self.sum_precision,
            "recall": self.sum_recall,
            "ndcg": self.sum_ndcg,
        }.get(metric, {})
        if isinstance(store, dict):
            return store.get(k, 0.0) / max(self.instances_completed, 1)
        return 0.0

    @property
    def avg_mrr(self) -> float:
        return self.sum_mrr / max(self.instances_completed, 1)


class DashboardState:
    """Thread-safe container for live experiment progress.

    The runner writes to this; the dashboard WebSocket reads from it.
    """

    def __init__(self, total: int = 0):
        self._lock = threading.Lock()
        self.total = total
        self.completed = 0
        self.current_gatherer: str = ""
        self.current_instance: str = ""
        self.start_time: float = time.time()

        # Per-gatherer running stats
        self.gatherer_stats: dict[str, GathererStats] = defaultdict(GathererStats)

        # Timeline for tokens-per-minute chart
        self.token_timeline: list[dict[str, Any]] = []

        # Cumulative totals
        self.total_tokens: int = 0

    def record(
        self,
        gatherer_name: str,
        token_usage: int,
        latency_s: float,
        metrics: dict[str, float],
        instance_id: str = "",
    ) -> None:
        """Record completion of one instance."""
        with self._lock:
            self.completed += 1
            self.current_gatherer = gatherer_name
            self.current_instance = instance_id
            self.total_tokens += token_usage

            # Per-gatherer stats
            stats = self.gatherer_stats[gatherer_name]
            stats.instances_completed += 1
            stats.total_tokens += token_usage
            stats.total_latency_s += latency_s

            # Update metric sums
            for key, value in metrics.items():
                if key.startswith("precision@"):
                    k = int(key.split("@")[1])
                    stats.sum_precision[k] += value
                elif key.startswith("recall@"):
                    k = int(key.split("@")[1])
                    stats.sum_recall[k] += value
                elif key.startswith("ndcg@"):
                    k = int(key.split("@")[1])
                    stats.sum_ndcg[k] += value
                elif key == "mrr":
                    stats.sum_mrr += value

            # Record per-instance snapshot for history charts
            snapshot_entry: dict[str, Any] = {
                "idx": stats.instances_completed,
                "instance_id": instance_id,
                "latency_s": round(latency_s, 3),
                "tokens": token_usage,
                "mrr": round(metrics.get("mrr", 0.0), 4),
            }
            # Add instance precision/recall/ndcg
            for k in sorted(stats.sum_precision.keys()):
                snapshot_entry[f"p@{k}"] = round(metrics.get(f"precision@{k}", 0.0), 4)
            for k in sorted(stats.sum_recall.keys()):
                snapshot_entry[f"r@{k}"] = round(metrics.get(f"recall@{k}", 0.0), 4)
            for k in sorted(stats.sum_ndcg.keys()):
                snapshot_entry[f"ndcg@{k}"] = round(metrics.get(f"ndcg@{k}", 0.0), 4)
            stats.metric_history.append(snapshot_entry)

            # Token timeline
            self.token_timeline.append({
                "timestamp": time.time(),
                "tokens": token_usage,
                "cumulative_tokens": self.total_tokens,
                "gatherer": gatherer_name,
            })

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the current state."""
        with self._lock:
            elapsed = time.time() - self.start_time
            tokens_per_min = (self.total_tokens / elapsed * 60) if elapsed > 0 else 0

            # Compute rolling tokens/min (last 60 seconds)
            cutoff = time.time() - 60
            recent_tokens = sum(
                entry["tokens"]
                for entry in self.token_timeline
                if entry["timestamp"] > cutoff
            )

            gatherers = {}
            for name, stats in self.gatherer_stats.items():
                gatherers[name] = {
                    "instances_completed": stats.instances_completed,
                    "total_tokens": stats.total_tokens,
                    "avg_tokens": round(stats.avg_tokens, 1),
                    "avg_latency_s": round(stats.avg_latency, 2),
                    "avg_mrr": round(stats.avg_mrr, 4),
                    "avg_precision": {
                        str(k): round(stats.avg_metric("precision", k), 4)
                        for k in sorted(stats.sum_precision.keys())
                    },
                    "avg_recall": {
                        str(k): round(stats.avg_metric("recall", k), 4)
                        for k in sorted(stats.sum_recall.keys())
                    },
                    "avg_ndcg": {
                        str(k): round(stats.avg_metric("ndcg", k), 4)
                        for k in sorted(stats.sum_ndcg.keys())
                    },
                    "metric_history": stats.metric_history[-200:],  # last 200 points
                }

            return {
                "progress": {
                    "completed": self.completed,
                    "total": self.total,
                    "percent": round(self.completed / max(self.total, 1) * 100, 1),
                    "current_gatherer": self.current_gatherer,
                    "current_instance": self.current_instance,
                },
                "tokens": {
                    "total": self.total_tokens,
                    "tokens_per_min": round(tokens_per_min, 1),
                    "rolling_tokens_per_min": round(recent_tokens, 0),
                },
                "elapsed_s": round(elapsed, 1),
                "gatherers": gatherers,
            }

