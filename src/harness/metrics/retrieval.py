"""Retrieval-quality metrics: Precision@K, Recall@K, MRR, NDCG@K."""

from __future__ import annotations

import math
from typing import Sequence


def _deduplicate(items: Sequence[str]) -> list[str]:
    """Remove duplicates from *items* while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def precision_at_k(
    retrieved: Sequence[str],
    gold: Sequence[str],
    k: int,
) -> float:
    """Precision@K — fraction of retrieved items up to rank K that are relevant.

    Args:
        retrieved: Ordered list of retrieved item identifiers (most relevant first).
        gold: Set of ground-truth relevant identifiers.
        k: Cutoff rank.

    Returns:
        Precision value in [0, 1].
    """
    if k <= 0:
        return 0.0
    effective_k = min(k, len(retrieved))
    if effective_k == 0:
        return 0.0
    top_k = _deduplicate(retrieved)[:k]
    gold_set = set(gold)
    relevant_count = sum(1 for item in top_k if item in gold_set)
    return relevant_count / effective_k


def recall_at_k(
    retrieved: Sequence[str],
    gold: Sequence[str],
    k: int,
) -> float:
    """Recall@K — fraction of relevant items recoverable within rank K that are found.

    Returns:
        Recall value in [0, 1].  Returns 0.0 if gold is empty.
    """
    if not gold or k <= 0:
        return 0.0
    top_k = _deduplicate(retrieved)[:k]
    gold_set = set(gold)
    relevant_count = sum(1 for item in top_k if item in gold_set)
    max_relevant_at_k = min(len(gold_set), k)
    return relevant_count / max_relevant_at_k


def mrr(
    retrieved: Sequence[str],
    gold: Sequence[str],
) -> float:
    """Mean Reciprocal Rank — 1 / rank of the first relevant result.

    Returns:
        MRR value in (0, 1] or 0.0 if no relevant item is found.
    """
    gold_set = set(gold)
    for rank, item in enumerate(_deduplicate(retrieved), start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    gold: Sequence[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if in gold set, 0 otherwise).

    Returns:
        NDCG value in [0, 1].  Returns 0.0 if gold is empty.
    """
    if not gold or k <= 0:
        return 0.0

    gold_set = set(gold)

    # DCG@K
    deduped = _deduplicate(retrieved)
    dcg = 0.0
    for i, item in enumerate(deduped[:k]):
        if item in gold_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG@K — all relevant items at top positions
    ideal_relevant = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_all_retrieval_metrics(
    retrieved: Sequence[str],
    gold: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute all retrieval metrics for a single instance.

    Returns a flat dict like::

        {
            "precision@1": 1.0,
            "precision@3": 0.33,
            "recall@1": 0.5,
            ...
            "mrr": 0.5,
            "ndcg@5": 0.72,
        }
    """
    results: dict[str, float] = {}

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        results[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)

    results["mrr"] = mrr(retrieved, gold)
    return results
