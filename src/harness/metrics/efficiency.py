"""Efficiency metrics: token usage, latency, time-to-first-token."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.gatherers.base import GatherResult


def compute_efficiency_metrics(result: GatherResult) -> dict[str, float | None]:
    """Extract efficiency metrics from a GatherResult.

    Returns:
        Dict with keys ``token_usage``, ``latency_s``, ``ttft_s``.
    """
    return {
        "token_usage": float(result.token_usage),
        "latency_s": result.latency_s,
        "ttft_s": result.ttft_s,
    }

