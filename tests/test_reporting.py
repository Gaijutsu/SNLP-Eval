"""Unit tests for the enriched ResultStore."""

import json

import pytest

from harness.reporting import ResultStore
from harness.gatherers.base import GatherResult


class TestVersionedResultStore:
    """Tests for versioned run directories and enriched output."""

    def test_creates_versioned_directory(self, tmp_path):
        store = ResultStore(tmp_path, run_id="2026-02-26_130000")
        assert store.output_dir == tmp_path / "2026-02-26_130000"
        assert store.output_dir.exists()

    def test_auto_generates_run_id(self, tmp_path):
        store = ResultStore(tmp_path)
        # Should create a timestamped directory
        assert store.run_id is not None
        assert store.output_dir.exists()
        assert store.output_dir.parent == tmp_path

    def test_save_run_meta(self, tmp_path):
        store = ResultStore(tmp_path, run_id="test_run")
        config = {"benchmark": {"name": "swebench_lite"}, "llm": {"model": "gpt-5"}}
        store.save_run_meta(config)

        meta_path = store.output_dir / "run_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == "test_run"
        assert meta["config"]["llm"]["model"] == "gpt-5"
        assert "timestamp" in meta


class TestEnrichedStore:
    """Tests for enriched per-instance JSON output."""

    def _make_result(self) -> GatherResult:
        return GatherResult(
            retrieved_contexts=["src/auth.py", "src/models.py"],
            token_usage=5000,
            latency_s=12.5,
            ttft_s=2.3,
            generated_patch="--- a/auth.py\n+++ b/auth.py\n",
            trace=[
                {"step": 1, "action": "grep", "args": ["login"], "tokens": 500}
            ],
            conversation=[
                {"role": "system", "content": "You are a code agent."},
                {"role": "user", "content": "Find relevant files."},
                {"role": "assistant", "content": "I'll search for login."},
            ],
        )

    def test_store_with_enriched_data(self, tmp_path):
        store = ResultStore(tmp_path, run_id="test_enriched")
        result = self._make_result()
        metrics = {"precision@1": 1.0, "recall@1": 0.5, "mrr": 1.0}

        store.store(
            "test__instance-1",
            "react_agent",
            metrics,
            result=result,
            gold_context=["src/auth.py", "src/utils.py"],
            model="gpt-5-mini",
        )

        # Check the file exists
        instance_file = store.output_dir / "instances" / "test__instance-1_react_agent.json"
        assert instance_file.exists()

        data = json.loads(instance_file.read_text())

        # Verify enriched fields
        assert data["instance_id"] == "test__instance-1"
        assert data["gatherer"] == "react_agent"
        assert data["model"] == "gpt-5-mini"
        assert data["metrics"]["precision@1"] == 1.0
        assert data["retrieved_documents"] == ["src/auth.py", "src/models.py"]
        assert data["gold_context"] == ["src/auth.py", "src/utils.py"]
        assert len(data["conversation"]) == 3
        assert data["conversation"][0]["role"] == "system"
        assert len(data["trace"]) == 1
        assert data["trace"][0]["action"] == "grep"
        assert data["generated_patch"] is not None
        assert data["token_usage"] == 5000
        assert data["latency_s"] == 12.5
        assert data["ttft_s"] == 2.3

    def test_store_without_enriched_data(self, tmp_path):
        """Backwards compatibility: calling store() without result/gold."""
        store = ResultStore(tmp_path, run_id="test_basic")
        metrics = {"precision@1": 0.5, "mrr": 0.25}

        store.store("inst_1", "rag_bm25", metrics)

        instance_file = store.output_dir / "instances" / "inst_1_rag_bm25.json"
        data = json.loads(instance_file.read_text())

        assert data["instance_id"] == "inst_1"
        assert data["gatherer"] == "rag_bm25"
        assert data["metrics"]["precision@1"] == 0.5
        # Enriched fields should not be present
        assert "retrieved_documents" not in data
        assert "conversation" not in data

    def test_generate_report(self, tmp_path):
        store = ResultStore(tmp_path, run_id="test_report")
        store.store("i1", "g1", {"mrr": 1.0, "precision@1": 1.0})
        store.store("i2", "g1", {"mrr": 0.5, "precision@1": 0.0})

        report_dir = store.generate_report()

        assert (report_dir / "results.csv").exists()
        assert (report_dir / "summary.json").exists()

        summary = json.loads((report_dir / "summary.json").read_text())
        assert "g1" in summary
        assert summary["g1"]["mrr"]["mean"] == 0.75


class TestGatherResultConversation:
    """Tests for the conversation field on GatherResult."""

    def test_default_empty(self):
        result = GatherResult()
        assert result.conversation == []

    def test_with_conversation(self):
        msgs = [{"role": "system", "content": "Hello"}]
        result = GatherResult(conversation=msgs)
        assert result.conversation == msgs
        assert len(result.conversation) == 1
