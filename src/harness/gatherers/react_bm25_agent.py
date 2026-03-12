"""Agentic BM25 RAG context gatherer — ReAct agent with BM25 search tool.

Combines the ReAct agent loop with BM25 retrieval as a first-class tool,
followed by an optional repair phase to generate candidate patches.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult
from harness.gatherers.rag_bm25 import ChunkedIndex
from harness.llm_client import LLMClient, LLMConfig

from harness.gatherers.prompts import (
    get_agentic_bm25_tool_descriptions,
    get_agentic_bm25_system_prompt,
    get_agentless_repair_prompt,
)

logger = logging.getLogger(__name__)

TOOL_DESCRIPTIONS = get_agentic_bm25_tool_descriptions()
SYSTEM_PROMPT = get_agentic_bm25_system_prompt()
REPAIR_PROMPT = get_agentless_repair_prompt()

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


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------


def _tool_bm25_search(
    index: ChunkedIndex, query: str, top_k: int = 10
) -> str:
    """Run a BM25 search and return formatted results."""
    results = index.search(query, top_k=top_k)
    if not results:
        return "No results found."
    lines = [f"  {path} (score: {score:.4f})" for path, score in results]
    return "\n".join(lines)


def _tool_list_dir(repo: Path, path: str) -> str:
    target = repo / path.strip().strip("/")
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if target.is_file():
        return (
            f"Error: '{path}' is a file, not a directory. "
            f'Use read_file("{path}") to read its contents.'
        )
    entries = []
    for item in sorted(target.iterdir()):
        rel = item.relative_to(repo).as_posix()
        suffix = "/" if item.is_dir() else ""
        entries.append(f"  {rel}{suffix}")
    if len(entries) > 100:
        entries = entries[:100] + [f"  ... and {len(entries) - 100} more"]
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
            f'Use list_dir("{path}") to list its contents.'
        )
    try:
        all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(all_lines)
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
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
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
    # Strip keyword-argument prefixes (e.g. "start_line=66" → "66")
    args = [a.split("=", 1)[-1].strip().strip("'\"") if "=" in a else a for a in args]
    return tool_name, args


def _extract_paths_from_conversation(
    messages: list[dict[str, str]], repo: Path
) -> list[str]:
    """Last-resort: extract file paths mentioned in assistant messages."""
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
    return paths[:20]


def _read_file_with_line_numbers(path: Path, max_lines: int = 300) -> str:
    """Read a file and prefix each line with its line number."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[:max_lines]
        numbered = [f"{i+1:>4}: {line}" for i, line in enumerate(selected)]
        result = "\n".join(numbered)
        if len(lines) > max_lines:
            result += f"\n... ({len(lines) - max_lines} more lines, total {len(lines)})"
        return result
    except OSError:
        return "(unable to read file)"


def _extract_diff(text: str) -> str | None:
    """Extract a diff block from LLM output."""
    match = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content.startswith(("---", "diff", "@@")):
            return content
    return None


# ------------------------------------------------------------------
# Main gatherer class
# ------------------------------------------------------------------


class AgenticBM25Gatherer(ContextGatherer):
    """ReAct agent with BM25 search as a first-class tool + repair phase."""

    name = "agentic_bm25"

    def __init__(
        self,
        llm_config: dict | None = None,
        max_steps: int = 15,
        bm25_top_k: int = 10,
        n_samples: int = 5,
        **kwargs: Any,
    ):
        cfg = llm_config or {}
        if "llm" in kwargs:
            cfg.setdefault("model", kwargs["llm"])
        self.llm = LLMClient(
            LLMConfig(**{k: v for k, v in cfg.items() if v is not None})
        )
        self.max_steps = max_steps
        self.bm25_top_k = bm25_top_k
        self.n_samples = n_samples

    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        t0 = time.perf_counter()
        total_tokens = 0
        ttft: float | None = None
        trace: list[dict[str, Any]] = []

        repo = instance.repo_snapshot

        # Build the BM25 index once for the entire gather call
        bm25_index = ChunkedIndex(repo)
        logger.info(
            "AgenticBM25: indexed %d files for BM25", len(bm25_index.file_paths)
        )

        system = SYSTEM_PROMPT.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
            max_steps=self.max_steps,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Find the files most relevant to this issue:\n\n"
                    f"{instance.query[:4000]}"
                ),
            },
        ]

        retrieved: list[str] = []

        # ── ReAct loop ─────────────────────────────────────────────
        for step in range(self.max_steps):
            response = self.llm.chat(messages)
            total_tokens += response.total_tokens
            if ttft is None:
                ttft = response.latency_s

            content = response.content
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
            elif tool_name == "bm25_search":
                query = args[0] if args else instance.query
                top_k = int(args[1]) if len(args) > 1 else self.bm25_top_k
                observation = _tool_bm25_search(bm25_index, query, top_k=top_k)
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
                    f"Available tools: bm25_search, list_dir, read_file, grep, "
                    f"finish. "
                    f"Remember: respond with ONLY Thought + Action."
                )

            # Build observation with step countdown
            remaining = self.max_steps - step - 1
            step_header = (
                f"[Step {step + 1}/{self.max_steps} — "
                f"{remaining} remaining]"
            )
            full_observation = f"{step_header}\n{observation[:3000]}"

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {"role": "user", "content": f"Observation:\n{full_observation}"}
            )

            trace[-1]["observation"] = observation[:500]

        else:
            # Agent exhausted all steps without calling finish()
            logger.info("AgenticBM25: agent exhausted steps — sending wrap-up prompt.")
            messages.append({"role": "user", "content": WRAP_UP_PROMPT})
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
                logger.warning(
                    "Wrap-up prompt failed to produce finish(). "
                    "Falling back to path extraction."
                )
                retrieved = _extract_paths_from_conversation(messages, repo)

        # ── Repair phase: generate candidate patches ───────────────
        patches: list[str] = []
        if retrieved and self.n_samples > 0:
            # Build code context from retrieved files
            file_contents = ""
            for fpath in retrieved[:5]:
                full_path = repo / fpath
                if full_path.exists():
                    content_text = _read_file_with_line_numbers(
                        full_path, max_lines=300
                    )
                    file_contents += (
                        f"\n### {fpath}\n```\n{content_text[:4000]}\n```\n"
                    )

            if file_contents.strip():
                for sample_idx in range(self.n_samples):
                    repair_msgs = [
                        {
                            "role": "system",
                            "content": "You are an expert software engineer.",
                        },
                        {
                            "role": "user",
                            "content": REPAIR_PROMPT.format(
                                query=instance.query[:2000],
                                code_regions=file_contents[:6000],
                            ),
                        },
                    ]
                    resp = self.llm.chat(repair_msgs, temperature=0.8)
                    total_tokens += resp.total_tokens
                    patch = _extract_diff(_strip_think_blocks(resp.content))
                    if patch:
                        patches.append(patch)

                trace.append(
                    {
                        "phase": "repair",
                        "n_patches_generated": len(patches),
                    }
                )

        best_patch = patches[0] if patches else None
        latency = time.perf_counter() - t0

        return GatherResult(
            retrieved_contexts=retrieved,
            token_usage=total_tokens,
            latency_s=latency,
            ttft_s=ttft,
            generated_patch=best_patch,
            trace=trace,
            conversation=list(messages),
        )
