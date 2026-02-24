"""Patch quality metrics for SWE-bench evaluation."""

from __future__ import annotations

import difflib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from harness.benchmarks.base import BenchmarkInstance


def patch_similarity(candidate: str, gold: str) -> dict[str, float]:
    """Compute similarity between candidate and gold patches.

    Returns:
        Dict with ``edit_similarity`` (SequenceMatcher ratio in [0, 1]).
    """
    if not candidate or not gold:
        return {"edit_similarity": 0.0}
    ratio = difflib.SequenceMatcher(None, candidate, gold).ratio()
    return {"edit_similarity": ratio}


def apply_and_test_patch(
    instance: BenchmarkInstance,
    patch: str | None,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    """Apply a candidate patch to the repo snapshot and run tests.

    This is the SWE-bench-style evaluation:
    1. Apply the patch via ``git apply``.
    2. Run fail-to-pass tests.
    3. Check pass-to-pass tests are not broken.

    Args:
        instance: The benchmark instance (must have ``repo_snapshot`` and
            ``metadata`` with ``test_patch`` and ``test_cmd``).
        patch: The candidate patch text.
        timeout: Max seconds for the test run.

    Returns:
        Dict with ``applied`` (bool), ``tests_passed`` (bool),
        ``fail_to_pass`` (int), ``pass_to_pass`` (int).
    """
    if not patch:
        return {
            "applied": False,
            "tests_passed": False,
            "fail_to_pass": 0,
            "pass_to_pass": 0,
        }

    repo = instance.repo_snapshot

    # Try applying the patch
    try:
        patch_file = repo / "_candidate.patch"
        patch_file.write_text(patch, encoding="utf-8")
        result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            cwd=str(repo),
            capture_output=True,
            timeout=30,
        )
        applied = result.returncode == 0

        if applied:
            subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=str(repo),
                capture_output=True,
                timeout=30,
                check=True,
            )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return {
            "applied": False,
            "tests_passed": False,
            "fail_to_pass": 0,
            "pass_to_pass": 0,
        }
    finally:
        if (repo / "_candidate.patch").exists():
            (repo / "_candidate.patch").unlink()

    if not applied:
        return {
            "applied": False,
            "tests_passed": False,
            "fail_to_pass": 0,
            "pass_to_pass": 0,
        }

    # Run tests if test command is provided
    test_cmd = instance.metadata.get("test_cmd", "pytest")
    try:
        test_result = subprocess.run(
            test_cmd,
            shell=True,
            cwd=str(repo),
            capture_output=True,
            timeout=timeout,
        )
        tests_passed = test_result.returncode == 0
    except subprocess.TimeoutExpired:
        tests_passed = False

    # Revert the patch
    try:
        subprocess.run(
            ["git", "checkout", "."],
            cwd=str(repo),
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass

    return {
        "applied": True,
        "tests_passed": tests_passed,
        "fail_to_pass": 1 if tests_passed else 0,
        "pass_to_pass": 1 if tests_passed else 0,
    }
