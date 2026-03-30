"""CrossCodeEval benchmark adapter."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkAdapter, BenchmarkInstance

logger = logging.getLogger(__name__)


class CrossCodeEvalAdapter(BenchmarkAdapter):
    """Adapter for the CrossCodeEval benchmark.

    CrossCodeEval evaluates cross-file code completion — each instance
    requires retrieving context from *other* files in the repository to
    correctly complete a code snippet.

    Loads from the HuggingFace ``microsoft/CrossCodeEval`` dataset.
    """

    HF_DATASET = "microsoft/CrossCodeEval"

    def __init__(
        self,
        language: str = "python",
        cache_dir: str | Path | None = None,
    ):
        self.language = language
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(tempfile.gettempdir()) / "crosscodeeval_repos"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        split: str = "test",
        limit: int | None = None,
        task_ids: list[str] | None = None,
    ) -> list[BenchmarkInstance]:
        from datasets import load_dataset

        ds = load_dataset(self.HF_DATASET, self.language, split=split)

        if task_ids:
            task_id_set = set(task_ids)
            ds = ds.filter(lambda row: row.get("task_id", "") in task_id_set)
        elif limit:
            ds = ds.select(range(min(limit, len(ds))))

        instances: list[BenchmarkInstance] = []
        for idx, row in enumerate(ds):
            instance = self._row_to_instance(row, idx)
            if instance:
                instances.append(instance)

        logger.info(
            "Loaded %d CrossCodeEval instances (lang=%s, split=%s)",
            len(instances),
            self.language,
            split,
        )
        return instances

    def _row_to_instance(
        self,
        row: dict[str, Any],
        idx: int,
    ) -> BenchmarkInstance | None:
        """Convert a dataset row to a BenchmarkInstance."""
        instance_id = row.get("task_id", f"crosscodeeval_{self.language}_{idx}")
        prompt = row.get("prompt", "")
        groundtruth = row.get("groundtruth", "")

        # Cross-file context references identified by static analysis
        cross_file_context = row.get("cross_file_context", [])
        if isinstance(cross_file_context, str):
            cross_file_context = [cross_file_context]

        # Gold context = the cross-file snippets the completion depends on
        gold_context: list[str] = []
        for ctx in cross_file_context:
            if isinstance(ctx, dict):
                path = ctx.get("path", ctx.get("file", ""))
                if path:
                    gold_context.append(path)
            elif isinstance(ctx, str):
                gold_context.append(ctx)

        # Repository path (CrossCodeEval bundles repo files in the dataset)
        repo_path = self.cache_dir / instance_id.replace("/", "__")
        repo_path.mkdir(parents=True, exist_ok=True)

        # Write repo files if provided in the dataset row
        repo_files = row.get("repo_files", {})
        if isinstance(repo_files, dict):
            for fpath, content in repo_files.items():
                full_path = repo_path / fpath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")

        return BenchmarkInstance(
            id=instance_id,
            query=prompt,
            repo_snapshot=repo_path,
            gold_context=gold_context,
            gold_patch=None,  # CrossCodeEval doesn't have patches
            metadata={
                "language": self.language,
                "groundtruth": groundtruth,
            },
        )

    def evaluate_patch(
        self,
        instance: BenchmarkInstance,
        patch: str | None,
    ) -> dict:
        """CrossCodeEval doesn't evaluate patches — retrieval metrics only."""
        return {}
