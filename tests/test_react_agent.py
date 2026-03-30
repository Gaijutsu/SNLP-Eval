from pathlib import Path

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.react_agent import (
    ReActGatherer,
    _parse_action,
    _finalize_retrieved_paths,
)
from harness.llm_client import LLMResponse


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of responses")
        return self._responses.pop(0)


def test_finalize_retrieved_paths_adds_matching_tests(tmp_path):
    repo = tmp_path
    source = repo / "astropy" / "io" / "ascii" / "qdp.py"
    test_file = repo / "astropy" / "io" / "ascii" / "tests" / "test_qdp.py"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source.write_text("def parse_qdp():\n    pass\n")
    test_file.write_text("def test_parse_qdp():\n    pass\n")

    result = _finalize_retrieved_paths(["astropy/io/ascii/qdp.py"], repo)

    assert result == [
        "astropy/io/ascii/qdp.py",
        "astropy/io/ascii/tests/test_qdp.py",
    ]


def test_react_gatherer_forces_early_wrap_up_and_keeps_tests(tmp_path):
    repo = tmp_path
    source = repo / "src" / "foo.py"
    test_file = repo / "tests" / "test_foo.py"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source.write_text("class Foo:\n    pass\n")
    test_file.write_text("def test_foo():\n    pass\n")

    gatherer = object.__new__(ReActGatherer)
    gatherer.max_steps = 6
    gatherer.llm = FakeLLM(
        [
            LLMResponse(
                content='Thought: Find Foo.\nAction: grep("Foo", ".")',
                total_tokens=10,
                latency_s=0.01,
            ),
            LLMResponse(
                content='Thought: Search again.\nAction: grep("Foo", ".")',
                total_tokens=10,
                latency_s=0.01,
            ),
            LLMResponse(
                content='Thought: Confirm implementation.\nAction: read_file("src/foo.py")',
                total_tokens=10,
                latency_s=0.01,
            ),
            LLMResponse(
                content='Thought: The main file is clear.\nAction: finish(src/foo.py)',
                total_tokens=10,
                latency_s=0.01,
            ),
        ]
    )

    instance = BenchmarkInstance(
        id="inst-1",
        query="Foo is broken; identify the relevant implementation and tests.",
        repo_snapshot=repo,
        gold_context=[],
    )

    result = gatherer.gather(instance)

    assert result.retrieved_contexts == ["src/foo.py", "tests/test_foo.py"]
    assert len(result.trace) == 4
    assert result.trace[-1]["action"] == "finish"
    assert not any(
        "no longer making useful progress" in message["content"]
        for message in result.conversation
        if message["role"] == "user"
    )


def test_react_gatherer_supports_find_tests_tool(tmp_path):
    repo = tmp_path
    source = repo / "pkg" / "core.py"
    test_file = repo / "pkg" / "tests" / "test_core.py"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source.write_text("def core():\n    return 1\n")
    test_file.write_text("def test_core():\n    assert True\n")

    gatherer = object.__new__(ReActGatherer)
    gatherer.max_steps = 5
    gatherer.llm = FakeLLM(
        [
            LLMResponse(
                content='Thought: Find core implementation.\nAction: grep("def core", ".")',
                total_tokens=10,
                latency_s=0.01,
            ),
            LLMResponse(
                content='Thought: Find matching tests.\nAction: find_tests("pkg/core.py")',
                total_tokens=10,
                latency_s=0.01,
            ),
            LLMResponse(
                content='Thought: Done.\nAction: finish(pkg/core.py, pkg/tests/test_core.py)',
                total_tokens=10,
                latency_s=0.01,
            ),
        ]
    )

    instance = BenchmarkInstance(
        id="inst-2",
        query="Core behavior is broken.",
        repo_snapshot=repo,
        gold_context=[],
    )

    result = gatherer.gather(instance)

    assert result.retrieved_contexts == ["pkg/core.py", "pkg/tests/test_core.py"]
    assert [step["action"] for step in result.trace] == ["grep", "find_tests", "finish"]


def test_finalize_retrieved_paths_uses_finish_paths_only(tmp_path):
    repo = tmp_path
    source = repo / "src" / "main.py"
    test_main = repo / "tests" / "test_main.py"
    distracting = repo / "src" / "distracting.py"
    source.parent.mkdir(parents=True)
    test_main.parent.mkdir(parents=True)
    source.write_text("def main():\n    pass\n")
    test_main.write_text("def test_main():\n    pass\n")
    distracting.write_text("def distracting():\n    pass\n")

    result = _finalize_retrieved_paths(
        ["src/main.py"],
        repo,
        candidate_paths=["src/distracting.py"],
        messages=[{"role": "assistant", "content": "src/distracting.py"}],
    )

    assert result == ["src/main.py", "tests/test_main.py"]


def test_parse_action_rejects_shell_style_actions():
    tool, args = _parse_action('Thought: x\nAction: grep "foo" src')
    assert tool == "error"
    assert args == []
