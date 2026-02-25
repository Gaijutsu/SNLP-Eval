"""Agentless context gatherer — 3-phase localization → repair → validation.

Based on: "Agentless: Demystifying LLM-based Software Engineering Agents"
(Xia et al., 2024).  Uses hierarchical LLM prompting for localization
without agentic tool-use loops.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkInstance
from harness.gatherers.base import ContextGatherer, GatherResult
from harness.gatherers.rag_bm25 import ChunkedIndex, _read_file_safe
from harness.llm_client import LLMClient, LLMConfig

from prompts import (
    get_agentless_file_localization_prompt,
    get_agentless_function_locatization_prompt,
    get_agentless_repair_prompt,
)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (e.g. Qwen3 chain-of-thought)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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


FILE_LOCALIZATION_PROMPT = get_agentless_file_localization_prompt()

FUNCTION_LOCALIZATION_PROMPT = get_agentless_function_locatization_prompt()

REPAIR_PROMPT = get_agentless_repair_prompt()


class AgentlessGatherer(ContextGatherer):
    """3-phase Agentless approach: localize → repair → validate."""

    name = "agentless"

    def __init__(
        self,
        llm_config: dict | None = None,
        n_samples: int = 5,
        top_files: int = 10,
        **kwargs: Any,
    ):
        cfg = llm_config or {}
        if "llm" in kwargs:
            cfg.setdefault("model", kwargs["llm"])
        self.llm = LLMClient(
            LLMConfig(**{k: v for k, v in cfg.items() if v is not None})
        )
        self.n_samples = n_samples
        self.top_files = top_files

    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        t0 = time.perf_counter()
        total_tokens = 0
        ttft: float | None = None
        trace: list[dict[str, Any]] = []
        conversation: list[dict[str, str]] = []

        repo = instance.repo_snapshot

        # ── Phase 1: File-level localization ──────────────────────
        file_listing = self._get_file_listing(repo)

        phase1_msgs = [
            {"role": "system", "content": "You are an expert code analyst."},
            {
                "role": "user",
                "content": FILE_LOCALIZATION_PROMPT.format(
                    query=instance.query[:3000],
                    file_listing=file_listing[:6000],
                    top_n=self.top_files,
                ),
            },
        ]
        resp = self.llm.chat(phase1_msgs)
        total_tokens += resp.total_tokens
        ttft = resp.latency_s
        conversation.extend(phase1_msgs)
        conversation.append({"role": "assistant", "content": resp.content})

        candidate_files = self._parse_json_list(_strip_think_blocks(resp.content))
        # Deduplicate while preserving order (LLM may repeat paths)
        seen: set[str] = set()
        candidate_files = [
            f for f in candidate_files
            if isinstance(f, str) and f not in seen and not seen.add(f)  # type: ignore[func-returns-value]
        ]
        # Fall back to BM25 if LLM didn't return useful results
        if not candidate_files:
            idx = ChunkedIndex(repo)
            bm25_results = idx.search(instance.query, top_k=self.top_files)
            candidate_files = [path for path, _ in bm25_results]

        trace.append(
            {
                "phase": "file_localization",
                "candidate_files": candidate_files,
                "tokens": resp.total_tokens,
            }
        )

        # ── Phase 2: Function-level localization ──────────────────
        file_contents = ""
        for fpath in candidate_files[:5]:  # Limit to top 5 to fit context
            full_path = repo / fpath
            if full_path.exists():
                content = _read_file_with_line_numbers(full_path, max_lines=300)
                file_contents += f"\n### {fpath}\n```\n{content[:4000]}\n```\n"

        phase2_msgs = [
            {"role": "system", "content": "You are an expert code analyst."},
            {
                "role": "user",
                "content": FUNCTION_LOCALIZATION_PROMPT.format(
                    query=instance.query[:2000],
                    file_contents=file_contents[:8000],
                ),
            },
        ]
        resp2 = self.llm.chat(phase2_msgs)
        total_tokens += resp2.total_tokens
        conversation.extend(phase2_msgs)
        conversation.append({"role": "assistant", "content": resp2.content})

        regions = self._parse_json_list(_strip_think_blocks(resp2.content))
        trace.append(
            {
                "phase": "function_localization",
                "regions": regions,
                "tokens": resp2.total_tokens,
            }
        )

        # ── Phase 3: Repair (generate candidate patches) ─────────
        code_regions = file_contents  # Reuse the file contents
        patches: list[str] = []

        for sample_idx in range(self.n_samples):
            phase3_msgs = [
                {
                    "role": "system",
                    "content": "You are an expert software engineer.",
                },
                {
                    "role": "user",
                    "content": REPAIR_PROMPT.format(
                        query=instance.query[:2000],
                        code_regions=code_regions[:6000],
                    ),
                },
            ]
            resp3 = self.llm.chat(phase3_msgs, temperature=0.8)
            total_tokens += resp3.total_tokens
            conversation.extend(phase3_msgs)
            conversation.append({"role": "assistant", "content": resp3.content})
            patch = self._extract_diff(_strip_think_blocks(resp3.content))
            if patch:
                patches.append(patch)

        trace.append(
            {
                "phase": "repair",
                "n_patches_generated": len(patches),
                "tokens_per_sample": resp3.total_tokens if patches else 0,
            }
        )

        # Select the first valid patch (simple heuristic; full Agentless
        # would run reproduction tests here)
        best_patch = patches[0] if patches else None

        latency = time.perf_counter() - t0

        return GatherResult(
            retrieved_contexts=candidate_files,
            token_usage=total_tokens,
            latency_s=latency,
            ttft_s=ttft,
            generated_patch=best_patch,
            trace=trace,
            conversation=conversation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_file_listing(repo: Path) -> str:
        """Get a compact listing of all source files in the repo."""
        extensions = {".py", ".java", ".ts", ".js", ".cs"}
        files: list[str] = []
        for fpath in sorted(repo.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix not in extensions:
                continue
            rel = fpath.relative_to(repo).as_posix()
            if any(part.startswith(".") for part in rel.split("/")):
                continue
            files.append(rel)
        return "\n".join(files)

    @staticmethod
    def _parse_json_list(text: str) -> list:
        """Extract a JSON array from LLM output."""
        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    @staticmethod
    def _extract_diff(text: str) -> str | None:
        """Extract a diff block from LLM output."""
        match = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try without language tag
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content.startswith(("---", "diff", "@@")):
                return content
        return None
