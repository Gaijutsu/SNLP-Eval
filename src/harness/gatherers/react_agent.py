"""ReAct-style agentic context gatherer.

Implements a Thought → Action → Observation loop where the agent uses
tools (list_dir, read_file, grep, search_codebase) to explore a repo
and locate relevant context for a given issue.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult
from harness.llm_client import LLMClient, LLMConfig

# ---------------------------------------------------------------------
# Tool implementations (operate within a repo snapshot)
# ---------------------------------------------------------------------

TOOL_DESCRIPTIONS = """\
You have the following tools available:

1. list_dir(path: str) -> str
   List files and subdirectories in the given directory (relative to repo root).

2. read_file(path: str) -> str
   Read the contents of a file (relative to repo root). Returns first 200 lines.

3. grep(pattern: str, path: str = ".") -> str
   Search for a regex pattern in files under the given path. Returns matching lines.

4. search_codebase(query: str) -> str
   Semantic search over all files in the repo. Returns the top-5 most relevant file paths.
"""

SYSTEM_PROMPT = """\
You are a code investigation agent. Your job is to find the files most relevant
to a given issue/query in a code repository.

{tool_descriptions}

On each turn, respond in EXACTLY this format:
Thought: <your reasoning about what to do next>
Action: <tool_name>(arg1, arg2, ...)

When you have found all relevant files, respond:
Thought: <summary of findings>
Action: finish(file1.py, file2.py, ...)

Rules:
- Always start with a Thought.
- Call exactly ONE action per turn.
- The finish action's arguments are the relevant file paths you found.
- You have at most {max_steps} steps.
"""


def _tool_list_dir(repo: Path, path: str) -> str:
    target = repo / path.strip().strip("/")
    if not target.exists() or not target.is_dir():
        return f"Error: '{path}' is not a valid directory."
    entries = []
    for item in sorted(target.iterdir()):
        rel = item.relative_to(repo).as_posix()
        suffix = "/" if item.is_dir() else ""
        entries.append(f"  {rel}{suffix}")
    if len(entries) > 100:
        entries = entries[:100] + [f"  ... and {len(entries) - 100} more"]
    return "\n".join(entries) if entries else "(empty directory)"


def _tool_read_file(repo: Path, path: str) -> str:
    target = repo / path.strip().strip("/")
    if not target.exists() or not target.is_file():
        return f"Error: '{path}' is not a valid file."
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > 200:
            return "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
        return "\n".join(lines)
    except OSError as e:
        return f"Error reading file: {e}"


def _tool_grep(repo: Path, pattern: str, path: str = ".") -> str:
    import subprocess

    target = repo / path.strip().strip("/")
    try:
        result = subprocess.run(
            ["grep", "-rnI", "--include=*.py", "--include=*.java",
             "--include=*.ts", "--include=*.js", "--include=*.cs",
             pattern, str(target)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(repo),
        )
        output = result.stdout.strip()
        lines = output.splitlines()
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
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
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        rel = f.relative_to(repo).as_posix()
                        matches.append(f"{rel}:{i}: {line.strip()}")
                        if len(matches) >= 50:
                            break
            except OSError:
                continue
            if len(matches) >= 50:
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
    return tool_name, args


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
        self.llm = LLMClient(LLMConfig(**{k: v for k, v in cfg.items() if v is not None}))
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
            {"role": "user", "content": f"Find the files most relevant to this issue:\n\n{instance.query[:4000]}"},
        ]

        retrieved: list[str] = []

        for step in range(self.max_steps):
            # Call LLM
            response = self.llm.chat(messages)
            total_tokens += response.total_tokens
            if ttft is None:
                ttft = response.latency_s  # Approximate TTFT as first call latency

            content = response.content
            tool_name, args = _parse_action(content)

            trace.append({
                "step": step + 1,
                "thought": content.split("Action:")[0].strip() if "Action:" in content else content,
                "action": tool_name,
                "args": args,
                "tokens": response.total_tokens,
            })

            # Execute tool
            if tool_name == "finish":
                retrieved = args
                break
            elif tool_name == "list_dir":
                observation = _tool_list_dir(repo, args[0] if args else ".")
            elif tool_name == "read_file":
                observation = _tool_read_file(repo, args[0] if args else "")
            elif tool_name == "grep":
                pattern = args[0] if args else ""
                path = args[1] if len(args) > 1 else "."
                observation = _tool_grep(repo, pattern, path)
            elif tool_name == "search_codebase":
                # Use BM25 as a simple search backend
                from harness.gatherers.rag_bm25 import ChunkedIndex
                idx = ChunkedIndex(repo)
                query = args[0] if args else instance.query
                results = idx.search(query, top_k=5)
                observation = "\n".join(f"  {path} (score: {score:.4f})" for path, score in results)
            else:
                observation = f"Unknown tool: {tool_name}. Available: list_dir, read_file, grep, search_codebase, finish."

            # Append observation to conversation
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation:\n{observation[:3000]}"})

            trace[-1]["observation"] = observation[:500]

        latency = time.perf_counter() - t0

        return GatherResult(
            retrieved_contexts=retrieved,
            token_usage=total_tokens,
            latency_s=latency,
            ttft_s=ttft,
            generated_patch=None,
            trace=trace,
        )
