"""Dense (embedding-based) RAG context gatherer."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from harness.gatherers.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult
from harness.gatherers.rag_bm25 import _read_file_safe

logger = logging.getLogger(__name__)


class DenseIndex:
    """Dense vector index over repo files."""

    def __init__(
        self,
        repo_path: Path,
        model: Any,  # Pre-loaded SentenceTransformer model
        extensions: tuple[str, ...] = (".py", ".java", ".ts", ".cs", ".js"),
    ):
        self.model = model
        self.file_paths: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._build(repo_path, extensions)

    def _build(self, repo_path: Path, extensions: tuple[str, ...]) -> None:
        """Walk the repo, read files, and compute embeddings."""
        logger.info("Building dense index: %s", repo_path.name)

        file_contents: list[str] = []
        for fpath in sorted(repo_path.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix not in extensions:
                continue

            rel = fpath.relative_to(repo_path).as_posix()
            if any(part.startswith(".") for part in rel.split("/")):
                continue

            content = _read_file_safe(fpath)
            if not content.strip():
                continue

            self.file_paths.append(rel)
            # Keep inputs bounded to avoid tokenizer / tensor blowups on very large files.
            file_contents.append(content[:8000])

        if not file_contents:
            logger.warning("Dense index empty: %s", repo_path.name)
            return

        self.embeddings = self.model.encode(
            file_contents,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        file_contents.clear()
        gc.collect()

        logger.info(
            "Built dense index: %s (%d files)",
            repo_path.name,
            len(self.file_paths),
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return top-K file paths with cosine similarity scores."""
        if self.embeddings is None or not self.file_paths:
            return []

        query_emb = self.model.encode(
            [query[:8000]],
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Cosine similarity since embeddings are normalized.
        scores = (self.embeddings @ query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.file_paths[i], float(scores[i])) for i in top_indices]


class DenseRAGGatherer(ContextGatherer):
    """Dense embedding-based retrieval context gatherer."""

    name = "rag_dense"

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
        **kwargs: Any,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.top_k = top_k

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)
        self._index_cache: dict[str, DenseIndex] = {}

        logger.info(
            "DenseRAG ready | model=%s | device=%s | top_k=%d",
            self.model_name,
            device,
            self.top_k,
        )

    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        t0 = time.perf_counter()

        repo_key = str(instance.repo_snapshot.resolve())

        if repo_key not in self._index_cache:
            self._index_cache[repo_key] = DenseIndex(
                instance.repo_snapshot,
                model=self._model,
            )
        else:
            logger.debug("Dense index cache hit: %s", instance.repo_snapshot.name)

        index = self._index_cache[repo_key]
        results = index.search(instance.query, top_k=self.top_k)
        retrieved = [path for path, _ in results]

        latency = time.perf_counter() - t0

        logger.debug(
            "Dense search complete: repo=%s results=%d latency=%.2fs",
            instance.repo_snapshot.name,
            len(retrieved),
            latency,
        )

        return GatherResult(
            retrieved_contexts=retrieved,
            token_usage=0,
            latency_s=latency,
            ttft_s=None,
            generated_patch=None,
            trace=[{
                "step": "dense_search",
                "model": self.model_name,
                "top_k": self.top_k,
                "num_indexed_files": len(index.file_paths),
                "results": [
                    {"file": path, "score": round(score, 4)}
                    for path, score in results
                ],
            }],
        )