"""BM25 (sparse) RAG context gatherer."""

from __future__ import annotations

import math
import re
import time
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult


logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + camelCase-aware tokenizer for code."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace(".", " ").replace("/", " ")
    return [t for t in text.lower().split() if len(t) > 1]


def _read_file_safe(path: Path) -> str:
    """Read a file, returning empty string on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class _BM25:
    """Minimal BM25-Okapi implementation (no external dependencies).

    score(Q, D) = Σ IDF(q) · (tf · (k1+1)) / (tf + k1 · (1 - b + b · |D|/avgdl))
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)

        # Inverted index: word -> {doc_id: tf}
        self.inv_index: dict[str, dict[int, int]] = {}

        for doc_idx, doc in enumerate(corpus):
            tf = dict(Counter(doc))
            for word, count in tf.items():
                if word not in self.inv_index:
                    self.inv_index[word] = {}
                self.inv_index[word][doc_idx] = count

        # Pre-compute IDF
        self.idf: dict[str, float] = {}
        for word, doc_map in self.inv_index.items():
            freq = len(doc_map)
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query: list[str]) -> list[float]:
        """Score all documents against a tokenized query using an inverted index."""
        scores = [0.0] * self.corpus_size

        for q in query:
            if q not in self.idf:
                continue

            idf = self.idf[q]
            doc_map = self.inv_index.get(q, {})

            for doc_idx, q_tf in doc_map.items():
                doc_len = self.doc_lens[doc_idx]
                numerator = q_tf * (self.k1 + 1)
                denominator = q_tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )
                scores[doc_idx] += idf * numerator / denominator

        return scores


class ChunkedIndex:
    """Indexes a repository by file, storing file paths and their content."""

    def __init__(
        self,
        repo_path: Path,
        extensions: tuple[str, ...] = (".py", ".java", ".ts", ".cs", ".js"),
    ):
        self.file_paths: list[str] = []
        tokenized_corpus: list[list[str]] = []
        self._build(repo_path, extensions, tokenized_corpus)
        if tokenized_corpus:
            self.bm25 = _BM25(tokenized_corpus)
        else:
            self.bm25 = None

    def _build(
        self,
        repo_path: Path,
        extensions: tuple[str, ...],
        tokenized_corpus: list[list[str]],
    ) -> None:
        """Walk the repo and index each eligible file."""
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

            tokens = _tokenize(content)
            if not tokens:
                continue

            self.file_paths.append(rel)
            tokenized_corpus.append(tokens)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return top-K file paths with BM25 scores."""
        if not self.bm25:
            return []

        # VERY IMPORTANT: `query` is a raw string here! It MUST be tokenized!
        tokenized_query = _tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.file_paths, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


class BM25RAGGatherer(ContextGatherer):
    """BM25-based sparse retrieval context gatherer."""

    name = "rag_bm25"

    _index_cache: dict[Path, ChunkedIndex] = {}

    def __init__(self, top_k: int = 10, **kwargs: Any):
        self.top_k = top_k

    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        t0 = time.perf_counter()

        repo_path = instance.repo_snapshot.resolve()

        if repo_path not in self._index_cache:
            logger.info("Building BM25 index for repo: %s", repo_path)
            self._index_cache[repo_path] = ChunkedIndex(repo_path)
        else:
            logger.info("Using cached BM25 index for repo: %s", repo_path)

        index = self._index_cache[repo_path]

        results = index.search(instance.query, top_k=self.top_k)
        retrieved = [path for path, _ in results]

        latency = time.perf_counter() - t0

        return GatherResult(
            retrieved_contexts=retrieved,
            token_usage=0,
            latency_s=latency,
            ttft_s=None,
            generated_patch=None,
            trace=[
                {
                    "step": "bm25_search",
                    "top_k": self.top_k,
                    "num_indexed_files": len(index.file_paths),
                    "results": [
                        {"file": path, "score": round(score, 4)}
                        for path, score in results
                    ],
                }
            ],
        )
