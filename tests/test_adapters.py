"""Smoke tests for benchmark adapters — verifies schema, not live data."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.benchmarks.base import BenchmarkAdapter, BenchmarkInstance


class TestBenchmarkInstance:
    def test_creation(self):
        inst = BenchmarkInstance(
            id="test_1",
            query="Fix the login bug",
            repo_snapshot=Path("/tmp/repo"),
            gold_context=["src/auth.py", "src/models/user.py"],
            gold_patch="--- a/src/auth.py\n+++ b/src/auth.py\n",
        )
        assert inst.id == "test_1"
        assert len(inst.gold_context) == 2
        assert inst.gold_patch is not None

    def test_default_fields(self):
        inst = BenchmarkInstance(
            id="test_2",
            query="query",
            repo_snapshot=Path("/tmp"),
            gold_context=[],
        )
        assert inst.gold_patch is None
        assert inst.metadata == {}


class TestBenchmarkAdapterBase:
    def test_evaluate_patch_default(self):
        """Default evaluate_patch returns empty dict."""

        class DummyAdapter(BenchmarkAdapter):
            def load(self, split="test", limit=None):
                return []

        adapter = DummyAdapter()
        inst = BenchmarkInstance(
            id="x", query="q", repo_snapshot=Path("/tmp"), gold_context=[]
        )
        assert adapter.evaluate_patch(inst, "some patch") == {}


class TestSWEBenchPatchExtraction:
    def test_extract_files(self):
        from harness.benchmarks.swebench import SWEBenchAdapter

        patch = (
            "diff --git a/src/auth.py b/src/auth.py\n"
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -10,3 +10,4 @@\n"
            " def login():\n"
            "+    pass\n"
            "diff --git a/src/models.py b/src/models.py\n"
            "--- a/src/models.py\n"
            "+++ b/src/models.py\n"
        )
        files = SWEBenchAdapter._extract_files_from_patch(patch)
        assert files == ["src/auth.py", "src/models.py"]

    def test_empty_patch(self):
        from harness.benchmarks.swebench import SWEBenchAdapter

        assert SWEBenchAdapter._extract_files_from_patch("") == []
