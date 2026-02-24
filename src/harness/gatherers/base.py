"""Abstract base class and data types for context-gathering strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.benchmarks.base import BenchmarkInstance


@dataclass
class GatherResult:
    """Uniform output from every context-gathering strategy."""

    # Ordered list of retrieved file paths / snippet identifiers (most relevant first)
    retrieved_contexts: list[str] = field(default_factory=list)

    # Total tokens consumed (prompt + completion)
    token_usage: int = 0

    # Wall-clock seconds for the entire gather call
    latency_s: float = 0.0

    # Time-to-first-token (if streaming was used), else None
    ttft_s: float | None = None

    # Generated patch text (agentic methods only)
    generated_patch: str | None = None

    # Step-by-step trace for post-hoc analysis
    trace: list[dict[str, Any]] = field(default_factory=list)


class ContextGatherer(ABC):
    """Base class for all context-gathering strategies.

    Subclasses implement ``gather()`` which receives a benchmark instance
    and returns a :class:`GatherResult` with the retrieved contexts, token
    usage, latency, and optionally a generated patch.
    """

    name: str = "base"

    @abstractmethod
    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        """Gather context for the given benchmark instance."""
        ...
