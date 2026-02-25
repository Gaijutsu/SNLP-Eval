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

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

FILE_LOCALIZATION_PROMPT = """\
You are an expert software engineer. Given the following issue/bug report
and repository file listing, identify which files are most likely relevant
to this issue. Return ONLY a JSON array of file paths, ranked by relevance.

## Issue
{query}

## Repository Files
{file_listing}

Respond with a JSON array of up to {top_n} file paths, most relevant first.
Example: ["src/auth/login.py", "src/models/user.py"]
"""

FUNCTION_LOCALIZATION_PROMPT = """\
You are an expert software engineer. Given the following issue and file
contents, identify the specific functions/classes/code regions that need
to be modified to fix this issue. Return a JSON array of objects with
"file" and "region" keys.

## Issue
{query}

## File Contents
{file_contents}

Respond with a JSON array of objects, each having:
- "file": the file path
- "region": description of the specific function/class/code region

Example: [{{"file": "src/auth.py", "region": "def login() around line 42"}}]
"""

REPAIR_PROMPT = """\
You are an expert software engineer. Given the issue description and the
relevant code regions, generate a patch in unified diff format to fix
the issue.

## Issue
{query}

## Relevant Code
{code_regions}

Generate a minimal, correct patch in unified diff format.
Start your response with ```diff and end with ```.
"""


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

        repo = instance.repo_snapshot

        # ── Phase 1: File-level localization ──────────────────────
        file_listing = self._get_file_listing(repo)

        resp = self.llm.chat(
            [
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
        )
        total_tokens += resp.total_tokens
        ttft = resp.latency_s

        candidate_files = self._parse_json_list(resp.content)
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
                content = _read_file_safe(full_path)
                file_contents += f"\n### {fpath}\n```\n{content[:3000]}\n```\n"

        resp2 = self.llm.chat(
            [
                {"role": "system", "content": "You are an expert code analyst."},
                {
                    "role": "user",
                    "content": FUNCTION_LOCALIZATION_PROMPT.format(
                        query=instance.query[:2000],
                        file_contents=file_contents[:8000],
                    ),
                },
            ]
        )
        total_tokens += resp2.total_tokens

        regions = self._parse_json_list(resp2.content)
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
            resp3 = self.llm.chat(
                [
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
                ],
                temperature=0.8,  # Higher temp for diversity
            )
            total_tokens += resp3.total_tokens
            patch = self._extract_diff(resp3.content)
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
