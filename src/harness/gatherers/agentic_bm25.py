"""ReAct-style agentic context gatherer augmented with BM25 retrieval.

Identical to the standard ReAct agent but adds a keyword_search() tool that
lets the agent perform BM25 keyword retrieval over the entire repository
to quickly find candidate files before drilling down with grep/read_file.
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
from harness.gatherers.rag_bm25 import ChunkedIndex
from harness.llm_client import LLMClient, LLMConfig

from harness.gatherers.prompts import RERAG_TOOL_DESCRIPTIONS, RERAG_SYSTEM_PROMPT

# Re-use helper functions from the base ReAct agent
from harness.gatherers.react_agent import (
    WRAP_UP_PROMPT,
    STAGNATION_PROMPT,
    SOURCE_FILE_SUFFIXES,
    MAX_REPEAT_ACTIONS,
    _strip_think_blocks,
    _tool_list_dir,
    _tool_read_file,
    _tool_grep,
    _is_file_path,
    _parse_action,
    _normalize_candidate_path,
    _extract_paths_from_text,
    _extract_paths_from_conversation,
    _add_candidate_path,
    _collect_new_candidates,
    _find_likely_tests_for_source,
    _finalize_retrieved_paths,
    _build_stagnation_prompt,
)

logger = logging.getLogger(__name__)

# Allow more exploration steps after keyword_search before triggering stagnation
AGENTIC_MAX_STAGNATION_STEPS = 5


def _tool_keyword_search(index: ChunkedIndex, query: str, top_k: int = 10) -> tuple[str, list[str]]:
    """BM25 keyword search over all indexed source files in the repository.

    Returns (formatted_output, list_of_paths) so the caller can automatically
    register returned paths as candidates.
    """
    results = index.search(query, top_k=top_k)
    if not results:
        return "No results found.", []
    lines = []
    paths = []
    for rank, (path, score) in enumerate(results, 1):
        lines.append(f"  {rank}. {path}  (score: {score:.4f})")
        paths.append(path)
    return "BM25 search results:\n" + "\n".join(lines), paths


class ReActBM25Gatherer(ContextGatherer):
    """ReAct agent augmented with a BM25 retrieval tool."""

    name = "agentic_bm25"

    def __init__(
        self,
        llm_config: dict | None = None,
        max_steps: int = 15,
        bm25_top_k: int = 10,
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

    def gather(self, instance: BenchmarkInstance) -> GatherResult:
        t0 = time.perf_counter()
        total_tokens = 0
        ttft: float | None = None
        trace: list[dict[str, Any]] = []

        repo = instance.repo_snapshot

        # Build the BM25 index once for this instance
        bm25_index = ChunkedIndex(repo)

        system = RERAG_SYSTEM_PROMPT.format(
            tool_descriptions=RERAG_TOOL_DESCRIPTIONS,
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
        # Track files the agent explicitly explored (read_file/grep targets)
        # so they can be prioritized over BM25-auto-registered candidates
        agent_explored: list[str] = []
        agent_explored_seen: set[str] = set()
        last_action: tuple[str, tuple[str, ...]] | None = None
        repeat_count = 0
        stagnation_count = 0

        # ---- Improvement: auto-run initial keyword_search with the full issue ----
        # This mirrors what rag_bm25 does (using the full issue text), giving the
        # agent a strong starting set of candidates before it even takes a turn.
        initial_output, initial_paths = _tool_keyword_search(
            bm25_index, instance.query, self.bm25_top_k
        )
        for p in initial_paths:
            _add_candidate_path(p, repo, candidate_paths, candidate_seen)
        # Inject the results as a system-level hint so the agent sees them
        messages.append(
            {
                "role": "user",
                "content": (
                    "Here are the initial BM25 keyword search results using the "
                    "full issue text as query. Use these as a starting point, "
                    "but feel free to run additional keyword_search() calls with "
                    "more focused queries to refine results.\n\n" + initial_output
                ),
            }
        )
        trace.append(
            {
                "step": 0,
                "thought": "AUTO: initial keyword_search with full issue text",
                "action": "keyword_search",
                "args": ["<full_issue_text>"],
                "tokens": 0,
                "observation": initial_output,
            }
        )

        for step in range(self.max_steps):
            # Call LLM
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

            action_key = (tool_name, tuple(args))
            if action_key == last_action:
                repeat_count += 1
            else:
                repeat_count = 1
                last_action = action_key

            # Build prioritized candidates: agent-explored files first,
            # then BM25/auto-registered candidates
            def _prioritized_candidates() -> list[str]:
                return agent_explored + [
                    p for p in candidate_paths if p not in agent_explored_seen
                ]

            # Execute tool
            if tool_name == "finish":
                retrieved = _finalize_retrieved_paths(
                    args,
                    repo,
                    candidate_paths=_prioritized_candidates(),
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
                # Track as agent-explored for prioritization
                norm = _normalize_candidate_path(fpath, repo)
                if norm and norm not in agent_explored_seen:
                    agent_explored_seen.add(norm)
                    agent_explored.append(norm)
            elif tool_name == "grep":
                pattern = args[0] if args else ""
                path = args[1] if len(args) > 1 else "."
                observation = _tool_grep(repo, pattern, path)
                # Track grep-discovered files as agent-explored
                for gpath in _extract_paths_from_text(observation, repo):
                    if gpath not in agent_explored_seen:
                        agent_explored_seen.add(gpath)
                        agent_explored.append(gpath)
            elif tool_name == "keyword_search":
                query = args[0] if args else instance.query
                top_k = int(args[1]) if len(args) > 1 else self.bm25_top_k
                observation, ks_paths = _tool_keyword_search(bm25_index, query, top_k)
                # Auto-register keyword_search result paths as candidates
                for p in ks_paths:
                    _add_candidate_path(p, repo, candidate_paths, candidate_seen)
            else:
                observation = (
                    f"Error: unknown tool '{tool_name}'. "
                    f"Available tools: list_dir, read_file, grep, keyword_search, finish. "
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
            # keyword_search is an overview/discovery tool — don't penalise it
            # for not finding *new* candidates (it may return files already seen
            # from the initial auto-search).
            if tool_name == "keyword_search":
                stagnation_count = 0
            elif tool_name == "error" or new_candidates == 0:
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
                    or stagnation_count >= AGENTIC_MAX_STAGNATION_STEPS
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
                        candidate_paths=_prioritized_candidates(),
                        messages=messages,
                    )
                else:
                    retrieved = _finalize_retrieved_paths(
                        [],
                        repo,
                        candidate_paths=_prioritized_candidates(),
                        messages=messages,
                    )
                break

        else:
            # Agent exhausted all steps without calling finish().
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

            # Build prioritized candidates for wrap-up finalization
            prioritized = agent_explored + [
                p for p in candidate_paths if p not in agent_explored_seen
            ]

            if tool_name == "finish" and args:
                retrieved = _finalize_retrieved_paths(
                    args,
                    repo,
                    candidate_paths=prioritized,
                    messages=messages,
                )
            else:
                logger.warning(
                    "Wrap-up prompt failed to produce finish(). "
                    "Falling back to path extraction."
                )
                retrieved = _finalize_retrieved_paths(
                    [],
                    repo,
                    candidate_paths=prioritized,
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
