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


class TestGoldContextStrategy:
    """Tests for the configurable gold_context_strategy."""

    PATCH = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    TEST_PATCH = (
        "diff --git a/tests/test_auth.py b/tests/test_auth.py\n"
        "--- a/tests/test_auth.py\n"
        "+++ b/tests/test_auth.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    def _make_repo(self, tmp_path):
        """Create a minimal repo with imports for testing."""
        # src/auth.py imports src/models.py and os (stdlib, should be ignored)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        (src_dir / "auth.py").write_text(
            "import os\nfrom src.models import User\nfrom src.utils import helper\n"
        )
        (src_dir / "models.py").write_text("class User: pass\n")
        (src_dir / "utils.py").write_text("def helper(): pass\n")

        # tests/
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_auth.py").write_text("def test_login(): pass\n")

        return tmp_path

    def test_patch_only(self, tmp_path):
        from harness.benchmarks.swebench import SWEBenchAdapter

        repo = self._make_repo(tmp_path)
        adapter = SWEBenchAdapter(gold_context_strategy="patch_only")
        gold = adapter._build_gold_context(self.PATCH, self.TEST_PATCH, repo)
        assert gold == ["src/auth.py"]

    def test_patch_and_tests(self, tmp_path):
        from harness.benchmarks.swebench import SWEBenchAdapter

        repo = self._make_repo(tmp_path)
        adapter = SWEBenchAdapter(gold_context_strategy="patch_and_tests")
        gold = adapter._build_gold_context(self.PATCH, self.TEST_PATCH, repo)
        assert "src/auth.py" in gold
        assert "tests/test_auth.py" in gold
        assert len(gold) == 2

    def test_patch_tests_and_imports(self, tmp_path):
        from harness.benchmarks.swebench import SWEBenchAdapter

        repo = self._make_repo(tmp_path)
        adapter = SWEBenchAdapter(gold_context_strategy="patch_tests_and_imports")
        gold = adapter._build_gold_context(self.PATCH, self.TEST_PATCH, repo)

        # Patch + test files
        assert "src/auth.py" in gold
        assert "tests/test_auth.py" in gold

        # First-party imports of src/auth.py should be included
        assert "src/models.py" in gold
        assert "src/utils.py" in gold

        # Stdlib imports (os) should NOT be included
        assert not any("os.py" in f for f in gold)

    def test_no_duplicates(self, tmp_path):
        from harness.benchmarks.swebench import SWEBenchAdapter

        repo = self._make_repo(tmp_path)
        adapter = SWEBenchAdapter(gold_context_strategy="patch_tests_and_imports")
        gold = adapter._build_gold_context(self.PATCH, self.TEST_PATCH, repo)
        assert len(gold) == len(set(gold)), "Gold set contains duplicates"

    def test_invalid_strategy_raises(self):
        from harness.benchmarks.swebench import SWEBenchAdapter

        with pytest.raises(ValueError, match="Unknown gold_context_strategy"):
            SWEBenchAdapter(gold_context_strategy="invalid_strategy")
