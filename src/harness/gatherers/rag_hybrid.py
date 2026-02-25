"""Hybrid RAG context gatherer — BM25 + Dense with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import torch

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult
from harness.gatherers.rag_bm25 import ChunkedIndex
from harness.gatherers.rag_dense import DenseIndex

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF) of multiple ranked result lists.

    RRF score for document d = Σ 1/(k + rank_i(d))
    where rank_i is the rank in the i-th list (1-indexed).

    Args:
        *ranked_lists: Each is a list of (item_id, score) sorted descending.
        k: RRF constant (default 60, as per the original paper).

    Returns:
        Fused list of (item_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRAGGatherer(ContextGatherer):
    """Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion."""

    name = "rag_hybrid"

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
        rrf_k: int = 60,
        **kwargs: Any,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.top_k = top_k
        self.rrf_k = rrf_k
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)
        logger.info("HybridRAGGatherer using device: %s", device)

    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        t0 = time.perf_counter()

        # Build both indexes
        bm25_index = ChunkedIndex(instance.repo_snapshot)
        dense_index = DenseIndex(
            instance.repo_snapshot,
            model=self._model,
        )

        # Retrieve from each
        # Fetch more than top_k from each to give RRF enough candidates
        fetch_k = self.top_k * 3
        bm25_results = bm25_index.search(instance.query, top_k=fetch_k)
        dense_results = dense_index.search(instance.query, top_k=fetch_k)

        # Fuse
        fused = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            k=self.rrf_k,
        )
        top_results = fused[: self.top_k]
        retrieved = [path for path, _ in top_results]

        latency = time.perf_counter() - t0

        return GatherResult(
            retrieved_contexts=retrieved,
            token_usage=0,
            latency_s=latency,
            ttft_s=None,
            generated_patch=None,
            trace=[
                {
                    "step": "hybrid_search",
                    "bm25_results": len(bm25_results),
                    "dense_results": len(dense_results),
                    "fused_top_k": self.top_k,
                    "results": [
                        {"file": path, "score": round(score, 6)}
                        for path, score in top_results
                    ],
                }
            ],
        )
