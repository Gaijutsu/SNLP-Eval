"""ReAct-style agentic context gatherer.

Implements a Thought → Action → Observation loop where the agent uses
tools (list_dir, read_file, grep) to explore a repo
and locate relevant context for a given issue.
"""

from __future__ import annotations

import logging
import re
import shlex
import time
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult
from harness.llm_client import LLMClient, LLMConfig

from harness.gatherers.prompts import REACT_TOOL_DESCRIPTIONS, REACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

WRAP_UP_PROMPT = """\
You have used all of your investigation steps. You MUST now provide your \
final answer by calling finish() with the file paths you believe are most \
relevant to the issue, and tests for those files if they exist. Base your answer on everything you have found so far.

Respond in EXACTLY this format:
Thought: <brief summary of relevant files found>
Action: finish(file1.py, file2.py, ...)
"""

STAGNATION_PROMPT = """\
You are repeating actions or not discovering new files.
Do NOT call the same tool again.
Based on the evidence already gathered, provide your final answer now by calling finish() with the most relevant source file(s) and matching test file(s) if they exist.

Likely candidate files already seen:
{candidates}

Respond in EXACTLY this format:
Thought: <brief summary of relevant files found>
Action: finish(file1.py, file2.py, ...)
"""

SOURCE_FILE_SUFFIXES = {".py", ".java", ".ts", ".js", ".cs"}
MAX_REPEAT_ACTIONS = 2
MAX_STAGNATION_STEPS = 3


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (e.g. Qwen3 chain-of-thought)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _tool_list_dir(repo: Path, path: str) -> str:
    target = repo / path.strip().strip("/")
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if target.is_file():
        return (
            f"Error: '{path}' is a file, not a directory. "
            f"Use read_file(\"{path}\") to read its contents."
        )
    entries = []
    for item in sorted(target.iterdir()):
        rel = item.relative_to(repo).as_posix()
        suffix = "/" if item.is_dir() else ""
        entries.append(f"  {rel}{suffix}")
    return "\n".join(entries) if entries else "(empty directory)"


def _tool_read_file(
    repo: Path,
    path: str,
    start_line: int = 1,
    end_line: int = 200,
) -> str:
    target = repo / path.strip().strip("/")
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if target.is_dir():
        return (
            f"Error: '{path}' is a directory, not a file. "
            f"Use list_dir(\"{path}\") to list its contents."
        )
    try:
        all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(all_lines)
        # Clamp bounds
        start_line = max(1, start_line)
        end_line = min(total, end_line)
        if start_line > total:
            return f"Error: start_line {start_line} exceeds file length ({total} lines)."

        selected = all_lines[start_line - 1 : end_line]
        numbered = [
            f"{start_line + i:>4}: {line}" for i, line in enumerate(selected)
        ]
        result = "\n".join(numbered)
        if end_line < total:
            result += f"\n... ({total - end_line} more lines, total {total})"
        return result
    except OSError as e:
        return f"Error reading file: {e}"


def _tool_grep(repo: Path, pattern: str, path: str = ".") -> str:
    import subprocess

    target = repo / path.strip().strip("/")
    try:
        result = subprocess.run(
            [
                "grep",
                "-rnI",
                "--include=*.py",
                "--include=*.java",
                "--include=*.ts",
                "--include=*.js",
                "--include=*.cs",
                pattern,
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(repo),
        )
        output = result.stdout.strip()
        lines = output.splitlines()
        if len(lines) > 100:
            return "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more matches)"
        return output if output else "No matches found."
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback: simple Python grep
        matches = []
        compiled = re.compile(pattern, re.IGNORECASE)
        search_path = repo / path.strip().strip("/")
        if search_path.is_file():
            files = [search_path]
        else:
            files = list(search_path.rglob("*"))
        for f in files[:500]:
            if not f.is_file() or f.suffix not in {".py", ".java", ".ts", ".js", ".cs"}:
                continue
            try:
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if compiled.search(line):
                        rel = f.relative_to(repo).as_posix()
                        matches.append(f"{rel}:{i}: {line.strip()}")
                        if len(matches) >= 100:
                            break
            except OSError:
                continue
            if len(matches) >= 100:
                break
        return "\n".join(matches) if matches else "No matches found."


def _is_file_path(s: str) -> bool:
    """Return True if s looks like a source file path (not a sentence/description)."""
    s = s.strip()
    if not s:
        return False
    # Must contain a known source extension
    if not re.search(r'\.(py|java|ts|js|cs|rb|go|cpp|c|h)$', s):
        return False
    # Must not look like a natural-language sentence (contains common sentence words)
    if len(s.split()) > 3:
        return False
    return True


def _parse_action(text: str) -> tuple[str, list[str]]:
    """Parse 'Action: tool_name(arg1, arg2)' from agent output.

    Supports function-style (tool(arg1, ...)) and shell-style (tool "arg1" arg2).
    Also handles list-syntax calls like finish(['a.py', 'b.py']) and filters
    non-path arguments from finish() to prevent sentence-in-finish failures.
    """
    # Try function-style first: Action: tool(arg1, arg2)
    # Use a greedy match to capture content including nested brackets
    match = re.search(r"Action:\s*(\w+)\(([^)]*(?:\)[^)]*)?)\)", text)
    if match:
        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()
        # Strip list-syntax brackets: finish(['a.py', 'b.py']) → finish(a.py, b.py)
        raw_args = re.sub(r"[\[\]]+", "", raw_args)
        if not raw_args:
            return tool_name, []
        args = [a.strip().strip("'\"") for a in raw_args.split(",")]
        # Strip keyword-argument prefixes (e.g. "start_line=66" → "66")
        args = [a.split("=", 1)[-1].strip().strip("'\"") if "=" in a else a for a in args]
        args = [a for a in args if a]
        # For finish(), filter out args that are clearly not file paths
        if tool_name == "finish":
            args = [a for a in args if _is_file_path(a)]
        return tool_name, args

    # Try shell-style: Action: tool "arg1" arg2 ...
    match = re.search(r"Action:\s*(\w+)\s+(.+)", text)
    if match:
        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()
        # Use shlex to split respecting quotes
        try:
            args = shlex.split(raw_args)
        except Exception:
            args = raw_args.split()
        if tool_name == "finish":
            args = [a for a in args if _is_file_path(a)]
        return tool_name, args

    return "error", []


def _normalize_candidate_path(candidate: str, repo: Path) -> str | None:
    """Normalize a candidate path and return it relative to ``repo`` if valid."""
    cleaned = candidate.strip().strip("\"'`[](){}<>,")
    if not cleaned:
        return None

    if cleaned.startswith(str(repo) + "/"):
        try:
            cleaned = Path(cleaned).relative_to(repo).as_posix()
        except ValueError:
            return None
    elif cleaned.startswith("/"):
        return None
    else:
        cleaned = cleaned.lstrip("./")

    target = repo / cleaned
    if not target.exists() or not target.is_file():
        return None
    if target.suffix not in SOURCE_FILE_SUFFIXES:
        return None
    return target.relative_to(repo).as_posix()


def _extract_paths_from_text(text: str, repo: Path) -> list[str]:
    """Extract source-file paths from free-form text, tool output, or grep results."""
    if not text:
        return []

    pattern = re.compile(
        r"(?P<path>(?:/[^\s:\"'`()<>,]+|[A-Za-z_][\w./-]*)\.(?:py|java|ts|js|cs))"
    )
    seen: set[str] = set()
    paths: list[str] = []
    for match in pattern.finditer(text):
        normalized = _normalize_candidate_path(match.group("path"), repo)
        if normalized and normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
    return paths


def _extract_paths_from_conversation(
    messages: list[dict[str, str]], repo: Path
) -> list[str]:
    """Last-resort: extract file paths mentioned in assistant messages."""
    paths: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for candidate in _extract_paths_from_text(msg.get("content", ""), repo):
            if candidate not in paths:
                paths.append(candidate)
    return paths[:20]  # Cap to avoid noise


def _add_candidate_path(
    candidate: str,
    repo: Path,
    candidates: list[str],
    seen: set[str],
) -> bool:
    normalized = _normalize_candidate_path(candidate, repo)
    if normalized is None or normalized in seen:
        return False
    seen.add(normalized)
    candidates.append(normalized)
    return True


def _collect_new_candidates(
    repo: Path,
    tool_name: str,
    args: list[str],
    observation: str,
    candidates: list[str],
    seen: set[str],
) -> int:
    before = len(candidates)

    if tool_name in {"read_file", "finish"}:
        for arg in args:
            _add_candidate_path(arg, repo, candidates, seen)

    for candidate in _extract_paths_from_text(observation, repo):
        _add_candidate_path(candidate, repo, candidates, seen)

    return len(candidates) - before


def _find_likely_tests_for_source(source_path: str, repo: Path) -> list[str]:
    """Find likely tests for a source file using common test naming conventions."""
    source = Path(source_path)
    if "tests" in source.parts or source.name.startswith("test_"):
        return []

    candidate_dirs: list[Path] = []
    current = source.parent
    while True:
        test_dir = current / "tests"
        if test_dir not in candidate_dirs:
            candidate_dirs.append(test_dir)
        if current == Path("."):
            break
        current = current.parent

    root_tests = Path("tests")
    if root_tests not in candidate_dirs:
        candidate_dirs.append(root_tests)

    candidate_names = [f"test_{source.stem}.py", f"{source.stem}_test.py"]
    matches: list[str] = []
    for directory in candidate_dirs:
        for name in candidate_names:
            candidate = repo / directory / name
            if candidate.exists() and candidate.is_file():
                matches.append(candidate.relative_to(repo).as_posix())
    return matches


def _finalize_retrieved_paths(
    raw_paths: list[str],
    repo: Path,
    *,
    candidate_paths: list[str] | None = None,
    messages: list[dict[str, str]] | None = None,
) -> list[str]:
    """Deduplicate paths, keep only valid repo files, and add likely tests."""
    ordered: list[str] = []
    seen: set[str] = set()

    def add_many(paths: list[str]) -> None:
        for path in paths:
            _add_candidate_path(path, repo, ordered, seen)

    add_many(raw_paths)
    if candidate_paths:
        add_many(candidate_paths)
    if messages:
        add_many(_extract_paths_from_conversation(messages, repo))

    finalized: list[str] = []
    finalized_seen: set[str] = set()
    for path in ordered:
        if path not in finalized_seen:
            finalized.append(path)
            finalized_seen.add(path)

        for test_path in _find_likely_tests_for_source(path, repo):
            if test_path not in finalized_seen:
                finalized.append(test_path)
                finalized_seen.add(test_path)

    return finalized[:20]


def _build_stagnation_prompt(candidate_paths: list[str]) -> str:
    candidates = "\n".join(f"- {path}" for path in candidate_paths[:8])
    if not candidates:
        candidates = "- (none found)"
    return STAGNATION_PROMPT.format(candidates=candidates)


class ReActGatherer(ContextGatherer):
    """ReAct agent that iteratively explores a repo to find relevant files."""

    name = "react_agent"

    def __init__(
        self,
        llm_config: dict | None = None,
        max_steps: int = 15,
        **kwargs: Any,
    ):
        cfg = llm_config or {}
        # Allow top-level 'llm' key to override model
        if "llm" in kwargs:
            cfg.setdefault("model", kwargs["llm"])
        self.llm = LLMClient(
            LLMConfig(**{k: v for k, v in cfg.items() if v is not None})
        )
        self.max_steps = max_steps

    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        t0 = time.perf_counter()
        total_tokens = 0
        ttft: float | None = None
        trace: list[dict[str, Any]] = []

        repo = instance.repo_snapshot
        system = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=REACT_TOOL_DESCRIPTIONS,
            max_steps=self.max_steps,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Find the files most relevant to this issue:\n\n{instance.query}",
            },
        ]

        retrieved: list[str] = []
        candidate_paths: list[str] = []
        candidate_seen: set[str] = set()
        last_action: tuple[str, tuple[str, ...]] | None = None
        repeat_count = 0
        stagnation_count = 0

        for step in range(self.max_steps):
            # Call LLM
            response = self.llm.chat(messages)
            total_tokens += response.total_tokens
            if ttft is None:
                ttft = response.latency_s  # Approximate TTFT as first call latency

            content = response.content
            # Strip <think> blocks before parsing Action
            cleaned = _strip_think_blocks(content)
            tool_name, args = _parse_action(cleaned)

            trace.append(
                {
                    "step": step + 1,
                    "thought": (
                        cleaned.split("Action:")[0].strip()
                        if "Action:" in cleaned
                        else cleaned
                    ),
                    "action": tool_name,
                    "args": args,
                    "tokens": response.total_tokens,
                }
            )

            action_key = (tool_name, tuple(args))
            if action_key == last_action:
                repeat_count += 1
            else:
                repeat_count = 1
                last_action = action_key

            # Execute tool
            if tool_name == "finish":
                retrieved = _finalize_retrieved_paths(
                    args,
                    repo,
                    candidate_paths=candidate_paths,
                    messages=messages,
                )
                break
            elif tool_name == "list_dir":
                observation = _tool_list_dir(repo, args[0] if args else ".")
            elif tool_name == "read_file":
                fpath = args[0] if args else ""
                start = int(args[1]) if len(args) > 1 else 1
                end = int(args[2]) if len(args) > 2 else start + 199
                observation = _tool_read_file(repo, fpath, start, end)
            elif tool_name == "grep":
                pattern = args[0] if args else ""
                path = args[1] if len(args) > 1 else "."
                observation = _tool_grep(repo, pattern, path)
            else:
                observation = (
                    f"Error: unknown tool '{tool_name}'. "
                    f"Available tools: list_dir, read_file, grep, finish. "
                    f"Remember: respond with ONLY Thought + Action."
                )

            new_candidates = _collect_new_candidates(
                repo,
                tool_name,
                args,
                observation,
                candidate_paths,
                candidate_seen,
            )
            if tool_name == "error" or new_candidates == 0:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if repeat_count >= MAX_REPEAT_ACTIONS:
                observation += (
                    "\nLoop warning: this exact tool call was already used. "
                    "Do not repeat it again; inspect a different file or call finish(...)."
                )

            # Build observation with step countdown
            remaining = self.max_steps - step - 1
            step_header = (
                f"[Step {step + 1}/{self.max_steps} — "
                f"{remaining} remaining]"
            )
            full_observation = f"{step_header}\n{observation}"

            # Append to conversation
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {"role": "user", "content": f"Observation:\n{full_observation}"}
            )

            trace[-1]["observation"] = observation

            should_force_wrap_up = (
                candidate_paths
                and step + 1 < self.max_steps
                and (
                    repeat_count >= MAX_REPEAT_ACTIONS
                    or stagnation_count >= MAX_STAGNATION_STEPS
                )
            )
            if should_force_wrap_up:
                logger.info("Agent stagnated — forcing early wrap-up.")
                wrap_response = self.llm.chat(
                    messages + [{"role": "user", "content": _build_stagnation_prompt(candidate_paths)}]
                )
                total_tokens += wrap_response.total_tokens
                wrap_content = wrap_response.content
                wrap_cleaned = _strip_think_blocks(wrap_content)
                wrap_tool_name, wrap_args = _parse_action(wrap_cleaned)

                trace.append(
                    {
                        "step": step + 1.5,
                        "thought": "EARLY WRAP-UP: " + (
                            wrap_cleaned.split("Action:")[0].strip()
                            if "Action:" in wrap_cleaned
                            else wrap_cleaned
                        ),
                        "action": wrap_tool_name,
                        "args": wrap_args,
                        "tokens": wrap_response.total_tokens,
                    }
                )
                messages.append({"role": "user", "content": _build_stagnation_prompt(candidate_paths)})
                messages.append({"role": "assistant", "content": wrap_content})

                if wrap_tool_name == "finish":
                    retrieved = _finalize_retrieved_paths(
                        wrap_args,
                        repo,
                        candidate_paths=candidate_paths,
                        messages=messages,
                    )
                else:
                    retrieved = _finalize_retrieved_paths(
                        [],
                        repo,
                        candidate_paths=candidate_paths,
                        messages=messages,
                    )
                break

        else:
            # Agent exhausted all steps without calling finish().
            # Give it one more chance with a wrap-up prompt.
            logger.info("Agent exhausted steps — sending wrap-up prompt.")
            messages.append(
                {"role": "user", "content": WRAP_UP_PROMPT}
            )
            response = self.llm.chat(messages)
            total_tokens += response.total_tokens
            content = response.content
            cleaned = _strip_think_blocks(content)
            tool_name, args = _parse_action(cleaned)

            trace.append(
                {
                    "step": self.max_steps + 1,
                    "thought": "WRAP-UP: " + (
                        cleaned.split("Action:")[0].strip()
                        if "Action:" in cleaned
                        else cleaned
                    ),
                    "action": tool_name,
                    "args": args,
                    "tokens": response.total_tokens,
                }
            )
            messages.append({"role": "assistant", "content": content})

            if tool_name == "finish" and args:
                retrieved = _finalize_retrieved_paths(
                    args,
                    repo,
                    candidate_paths=candidate_paths,
                    messages=messages,
                )
            else:
                # Final fallback: extract paths from conversation history
                logger.warning(
                    "Wrap-up prompt failed to produce finish(). "
                    "Falling back to path extraction."
                )
                retrieved = _finalize_retrieved_paths(
                    [],
                    repo,
                    candidate_paths=candidate_paths,
                    messages=messages,
                )

        latency = time.perf_counter() - t0

        return GatherResult(
            retrieved_contexts=retrieved,
            token_usage=total_tokens,
            latency_s=latency,
            ttft_s=ttft,
            generated_patch=None,
            trace=trace,
            conversation=list(messages),
        )
