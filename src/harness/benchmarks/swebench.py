"""SWE-bench Lite benchmark adapter."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkAdapter, BenchmarkInstance
from harness.metrics.patch import apply_and_test_patch, patch_similarity

logger = logging.getLogger(__name__)


class SWEBenchAdapter(BenchmarkAdapter):
    """Adapter for the SWE-bench Lite dataset.

    Loads instances from the HuggingFace ``princeton-nlp/SWE-bench_Lite``
    dataset.  Each instance contains a GitHub issue, the repo to clone,
    the base commit, the gold patch, and test information.
    """

    HF_DATASET = "princeton-nlp/SWE-bench_Lite"

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(tempfile.gettempdir()) / "swebench_repos"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        split: str = "test",
        limit: int | None = None,
    ) -> list[BenchmarkInstance]:
        import os

        os.environ["DATASETS_NO_TORCH"] = "1"
        from datasets import load_dataset

        ds = load_dataset(self.HF_DATASET, split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        instances: list[BenchmarkInstance] = []
        for row in ds:
            instance = self._row_to_instance(row)
            if instance:
                instances.append(instance)

        logger.info(
            "Loaded %d SWE-bench Lite instances (split=%s)", len(instances), split
        )
        return instances

    def _row_to_instance(self, row: dict[str, Any]) -> BenchmarkInstance | None:
        """Convert a HuggingFace dataset row to a BenchmarkInstance."""
        instance_id: str = row["instance_id"]
        repo_name: str = row["repo"]
        base_commit: str = row["base_commit"]
        patch: str = row.get("patch", "")
        test_patch: str = row.get("test_patch", "")
        problem_statement: str = row.get("problem_statement", "")
        hints_text: str = row.get("hints_text", "")

        # Derive gold context from the files touched by the reference patch
        gold_files = self._extract_files_from_patch(patch)

        # Prepare repo snapshot path (clone will happen lazily or upfront)
        repo_dir = self.cache_dir / instance_id.replace("/", "__")

        # Clone / checkout if not already present
        if not repo_dir.exists():
            try:
                self._prepare_repo(repo_name, base_commit, repo_dir)
            except Exception as e:
                logger.warning("Failed to prepare repo for %s: %s", instance_id, e)
                return None

        query = problem_statement
        if hints_text:
            query += f"\n\nHints:\n{hints_text}"

        return BenchmarkInstance(
            id=instance_id,
            query=query,
            repo_snapshot=repo_dir,
            gold_context=gold_files,
            gold_patch=patch,
            metadata={
                "repo": repo_name,
                "base_commit": base_commit,
                "test_patch": test_patch,
                "test_cmd": "pytest",
            },
        )

    def evaluate_patch(
        self,
        instance: BenchmarkInstance,
        patch: str | None,
    ) -> dict[str, Any]:
        """Apply candidate patch, run tests, and compute similarity to gold."""
        results: dict[str, Any] = {}

        # Patch similarity to gold
        if instance.gold_patch and patch:
            results.update(patch_similarity(patch, instance.gold_patch))

        # Apply and run tests
        test_results = apply_and_test_patch(instance, patch)
        results.update(test_results)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_files_from_patch(patch: str) -> list[str]:
        """Parse a unified diff to extract the list of modified file paths."""
        files: list[str] = []
        for line in patch.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    # "diff --git a/path b/path" → extract b/path
                    path = parts[3]
                    if path.startswith("b/"):
                        path = path[2:]
                    if path not in files:
                        files.append(path)
        return files

    @staticmethod
    def _prepare_repo(repo_name: str, base_commit: str, dest: Path) -> None:
        """Clone a repo at a specific commit."""
        url = f"https://github.com/{repo_name}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            capture_output=True,
            check=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", base_commit],
            cwd=str(dest),
            capture_output=True,
            check=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(dest),
            capture_output=True,
            check=True,
            timeout=60,
        )
