"""Abstract base class and data types for benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkInstance:
    """A single benchmark item (SWE-bench issue or CrossCodeEval completion)."""

    id: str
    query: str  # issue text (SWE-bench) or code prefix (CrossCodeEval)
    repo_snapshot: Path  # path to checked-out repo at the right commit
    gold_context: list[str]  # ground-truth relevant files / snippets
    gold_patch: str | None = None  # reference patch (SWE-bench only)
    metadata: dict = field(default_factory=dict)  # extra benchmark-specific data


class BenchmarkAdapter(ABC):
    """Load benchmark instances and (optionally) evaluate generated patches."""

    @abstractmethod
    def load(
        self,
        split: str = "test",
        limit: int | None = None,
        task_ids: list[str] | None = None,
    ) -> list[BenchmarkInstance]:
        """Return a list of benchmark instances.

        Args:
            split: Dataset split to load (e.g. "test", "dev").
            limit: Cap the number of instances (``None`` for all).
            task_ids: Optional list of specific instance IDs to load
                (for reproducible subsets).
        """
        ...

    def evaluate_patch(
        self,
        instance: BenchmarkInstance,
        patch: str | None,
    ) -> dict:
        """Evaluate a candidate patch against the instance.

        Returns a dict of metric_name → value.  Default returns empty dict
        (appropriate for benchmarks without patch evaluation like CrossCodeEval).
        """
        return {}
