"""SWE-bench Lite benchmark adapter."""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkAdapter, BenchmarkInstance
from harness.metrics.patch import apply_and_test_patch, patch_similarity

logger = logging.getLogger(__name__)

# Valid gold-context strategies (see README § Gold Set Construction)
GOLD_STRATEGIES = ("patch_only", "patch_and_tests", "patch_tests_and_imports")


class SWEBenchAdapter(BenchmarkAdapter):
    """Adapter for the SWE-bench Lite dataset.

    Loads instances from the HuggingFace ``princeton-nlp/SWE-bench_Lite``
    dataset.  Each instance contains a GitHub issue, the repo to clone,
    the base commit, the gold patch, and test information.

    The ``gold_context_strategy`` controls how the ground-truth relevant
    file set is built:

    * ``patch_only`` – only files modified in the reference patch.
    * ``patch_and_tests`` – patch files **plus** files from the test patch.
    * ``patch_tests_and_imports`` – the above **plus** first-party files
      directly imported by the patched files.
    """

    HF_DATASET = "princeton-nlp/SWE-bench_Lite"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        gold_context_strategy: str = "patch_and_tests",
    ):
        if gold_context_strategy not in GOLD_STRATEGIES:
            raise ValueError(
                f"Unknown gold_context_strategy '{gold_context_strategy}'. "
                f"Choose from: {GOLD_STRATEGIES}"
            )
        self.gold_context_strategy = gold_context_strategy
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
        total = len(ds)
        for idx, row in enumerate(ds, 1):
            logger.info(
                "Preparing instance %d/%d: %s", idx, total, row.get("instance_id", "?")
            )
            instance = self._row_to_instance(row)
            if instance:
                instances.append(instance)

        logger.info(
            "Loaded %d SWE-bench Lite instances (split=%s, gold_strategy=%s)",
            len(instances),
            split,
            self.gold_context_strategy,
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

        # Prepare repo snapshot path (clone will happen lazily or upfront)
        repo_dir = self.cache_dir / instance_id.replace("/", "__")

        # Clone / checkout if not already present
        if not repo_dir.exists():
            try:
                self._prepare_repo(repo_name, base_commit, repo_dir)
            except Exception as e:
                logger.warning("Failed to prepare repo for %s: %s", instance_id, e)
                return None

        # ── Build gold context using the configured strategy ──────────
        gold_files = self._build_gold_context(
            patch=patch,
            test_patch=test_patch,
            repo_dir=repo_dir,
        )

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
                "gold_context_strategy": self.gold_context_strategy,
            },
        )

    def _build_gold_context(
        self,
        patch: str,
        test_patch: str,
        repo_dir: Path,
    ) -> list[str]:
        """Build the gold-context file list according to the active strategy.

        Strategies (cumulative):
          1. ``patch_only``              – files from the reference patch.
          2. ``patch_and_tests``         – (1) + files from the test patch.
          3. ``patch_tests_and_imports`` – (2) + first-party imports of (1).
        """
        seen: set[str] = set()
        gold: list[str] = []

        def _add(paths: list[str]) -> None:
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    gold.append(p)

        # Tier 1: files modified by the fix
        patch_files = self._extract_files_from_patch(patch)
        _add(patch_files)

        if self.gold_context_strategy in ("patch_and_tests", "patch_tests_and_imports"):
            # Tier 2: files modified by the test patch
            test_files = self._extract_files_from_patch(test_patch)
            _add(test_files)

        if self.gold_context_strategy == "patch_tests_and_imports":
            # Tier 3: first-party imports of patched files
            import_files = self._collect_imports(patch_files, repo_dir)
            _add(import_files)

        return gold

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
    def _collect_imports(
        source_files: list[str],
        repo_dir: Path,
    ) -> list[str]:
        """Resolve first-party imports from *source_files* in *repo_dir*.

        For each file in *source_files* that exists in the repo, parse its
        ``import`` / ``from ... import`` statements and attempt to map them
        back to concrete files in the repository.  Only first-party imports
        (files that actually exist in the repo) are returned.

        This deliberately does **not** recurse — it returns only the direct
        (depth-1) imports to keep the gold set focused.
        """
        imported: list[str] = []
        seen: set[str] = set()

        for src in source_files:
            full = repo_dir / src
            if not full.exists() or full.suffix != ".py":
                continue

            try:
                source = full.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=src)
            except (SyntaxError, ValueError):
                # Fall back to regex for files that are not valid Python 3
                imported.extend(
                    SWEBenchAdapter._imports_via_regex(src, full, repo_dir, seen)
                )
                continue

            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        modules = [node.module]

                for mod in modules:
                    candidates = SWEBenchAdapter._module_to_paths(mod, src)
                    for cand in candidates:
                        if cand not in seen and (repo_dir / cand).is_file():
                            seen.add(cand)
                            imported.append(cand)

        return imported

    @staticmethod
    def _imports_via_regex(
        src: str,
        full_path: Path,
        repo_dir: Path,
        seen: set[str],
    ) -> list[str]:
        """Regex fallback for extracting imports from non-parseable files."""
        results: list[str] = []
        try:
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return results

        pattern = re.compile(
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
        )
        for m in pattern.finditer(text):
            mod = m.group(1) or m.group(2)
            if mod:
                for cand in SWEBenchAdapter._module_to_paths(mod, src):
                    if cand not in seen and (repo_dir / cand).is_file():
                        seen.add(cand)
                        results.append(cand)
        return results

    @staticmethod
    def _module_to_paths(module: str, source_file: str) -> list[str]:
        """Convert a dotted module name to candidate file paths.

        Given ``module = "django.db.models"`` we produce:
          - ``django/db/models.py``
          - ``django/db/models/__init__.py``

        For relative-looking short names we also try resolving relative to
        the source file's directory.
        """
        parts = module.split(".")
        base = "/".join(parts)
        candidates = [f"{base}.py", f"{base}/__init__.py"]

        # Also try relative to the source file's package
        src_dir = str(Path(source_file).parent)
        if src_dir and src_dir != ".":
            rel_base = f"{src_dir}/{'/'.join(parts)}"
            candidates.append(f"{rel_base}.py")
            candidates.append(f"{rel_base}/__init__.py")

        return candidates

    @staticmethod
    def _prepare_repo(repo_name: str, base_commit: str, dest: Path) -> None:
        """Clone a repo at a specific commit."""
        url = f"https://github.com/{repo_name}.git"
        logger.info("  Cloning %s …", url)
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            capture_output=True,
            check=True,
            timeout=120,
        )
        logger.info("  Fetching commit %s …", base_commit[:10])
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", base_commit],
            cwd=str(dest),
            capture_output=True,
            check=True,
            timeout=120,
        )
        logger.info("  Checking out %s …", base_commit[:10])
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(dest),
            capture_output=True,
            check=True,
            timeout=60,
        )
