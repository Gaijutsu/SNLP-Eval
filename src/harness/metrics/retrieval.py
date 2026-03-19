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
    """Precision@K — fraction of top-K retrieved items that are relevant.

    Args:
        retrieved: Ordered list of retrieved item identifiers (most relevant first).
        gold: Set of ground-truth relevant identifiers.
        k: Cutoff rank.

    Returns:
        Precision value in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = _deduplicate(retrieved)[:k]
    gold_set = set(gold)
    relevant_count = sum(1 for item in top_k if item in gold_set)
    return relevant_count / k


def recall_at_k(
    retrieved: Sequence[str],
    gold: Sequence[str],
    k: int,
) -> float:
    """Recall@K — fraction of relevant items found in top-K results.

    Returns:
        Recall value in [0, 1].  Returns 0.0 if gold is empty.
    """
    if not gold or k <= 0:
        return 0.0
    top_k = _deduplicate(retrieved)[:k]
    gold_set = set(gold)
    relevant_count = sum(1 for item in top_k if item in gold_set)
    return relevant_count / len(gold_set)


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


def success_at_k(
    retrieved: Sequence[str],
    gold: Sequence[str],
    k: int,
) -> float:
    """Success@K — 1.0 if ALL gold items are found in top-K, else 0.0.

    This is a stricter metric than Recall@K: it requires *every* relevant
    item to appear in the top-K results.  Useful when downstream tasks
    (e.g., patch generation) need *all* modified files to succeed.

    Returns:
        1.0 or 0.0.  Returns 1.0 when gold is empty (vacuously true).
    """
    if not gold or k <= 0:
        return 0.0 if (gold and k <= 0) else (1.0 if not gold else 0.0)
    top_k = set(retrieved[:k])
    return 1.0 if set(gold).issubset(top_k) else 0.0


def f1_at_k(
    retrieved: Sequence[str],
    gold: Sequence[str],
    k: int,
) -> float:
    """F1@K — harmonic mean of Precision@K and Recall@K.

    Balances the trade-off between retrieving relevant items (recall)
    and avoiding irrelevant ones (precision).

    Returns:
        F1 value in [0, 1].  Returns 0.0 if both precision and recall are 0.
    """
    p = precision_at_k(retrieved, gold, k)
    r = recall_at_k(retrieved, gold, k)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


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
            "success@5": 1.0,
            "f1@5": 0.33,
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
        results[f"success@{k}"] = success_at_k(retrieved, gold, k)
        results[f"f1@{k}"] = f1_at_k(retrieved, gold, k)

    results["mrr"] = mrr(retrieved, gold)
    return results
