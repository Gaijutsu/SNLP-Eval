from pathlib import Path

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.react_agent import (
    ReActGatherer,
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
    assert len(result.trace) == 3
    assert result.trace[-1]["action"] == "finish"
    assert any(
        "Do NOT call the same tool again" in message["content"]
        for message in result.conversation
        if message["role"] == "user"
    )
