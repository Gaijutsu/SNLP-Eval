"""ReAct-style agentic context gatherer.

Implements a Thought → Action → Observation loop where the agent uses
tools (list_dir, read_file, grep, search_codebase) to explore a repo
and locate relevant context for a given issue.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult
from harness.llm_client import LLMClient, LLMConfig

from harness.gatherers.prompts import get_react_tool_descriptions, get_react_system_prompt

logger = logging.getLogger(__name__)

TOOL_DESCRIPTIONS = get_react_tool_descriptions()
SYSTEM_PROMPT = get_react_system_prompt()

WRAP_UP_PROMPT = """\
You have used all of your investigation steps. You MUST now provide your \
final answer by calling finish() with the file paths you believe are most \
relevant to the issue. Base your answer on everything you have found so far.

Respond in EXACTLY this format:
Thought: <brief summary of relevant files found>
Action: finish(file1.py, file2.py, ...)
"""


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
            f"Use list_dir(\"{path}\") to list its contents."hf.co/unsloth
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


def _parse_action(text: str) -> tuple[str, list[str]]:
    """Parse 'Action: tool_name(arg1, arg2)' from agent output."""
    match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
    if not match:
        return "error", []
    tool_name = match.group(1).strip()
    raw_args = match.group(2).strip()
    if not raw_args:
        return tool_name, []
    args = [a.strip().strip("'\"") for a in raw_args.split(",")]
    # Strip keyword-argument prefixes (e.g. "start_line=66" → "66")
    args = [a.split("=", 1)[-1].strip().strip("'\"") if "=" in a else a for a in args]
    return tool_name, args


def _extract_paths_from_conversation(
    messages: list[dict[str, str]], repo: Path
) -> list[str]:
    """Last-resort: extract file paths mentioned in assistant messages."""
    # Collect all paths that look like source files
    path_pattern = re.compile(
        r"(?:^|[\s\"'`(,])([a-zA-Z_][\w/.-]*\.(?:py|java|ts|js|cs))\b"
    )
    seen: set[str] = set()
    paths: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for match in path_pattern.finditer(msg.get("content", "")):
            candidate = match.group(1)
            if candidate not in seen and (repo / candidate).exists():
                seen.add(candidate)
                paths.append(candidate)
    return paths[:20]  # Cap to avoid noise


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
        system = SYSTEM_PROMPT.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
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

            # Execute tool
            if tool_name == "finish":
                retrieved = args
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
                    f"Available tools: list_dir, read_file, grep, "
                    f"search_codebase, finish. "
                    f"Remember: respond with ONLY Thought + Action."
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
                retrieved = args
            else:
                # Final fallback: extract paths from conversation history
                logger.warning(
                    "Wrap-up prompt failed to produce finish(). "
                    "Falling back to path extraction."
                )
                retrieved = _extract_paths_from_conversation(messages, repo)

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
