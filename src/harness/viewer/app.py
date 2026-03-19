"""Results viewer — FastAPI app for exploring past experiment results."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Context Gathering Harness — Results Viewer")

# Global results directory (set via CLI)
_results_dir: Path = Path("./results")


def set_results_dir(path: Path) -> None:
    global _results_dir
    _results_dir = path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _discover_runs() -> list[dict[str, Any]]:
    """Discover all run directories under the results path.

    Supports both new versioned layout (dirs with run_meta.json)
    and legacy flat layout (results/instances/*.json).
    """
    runs: list[dict[str, Any]] = []

    if not _results_dir.exists():
        return runs

    # Check for versioned run directories (contain run_meta.json)
    for entry in sorted(_results_dir.iterdir(), reverse=True):
        if entry.is_dir():
            meta_path = entry / "run_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    runs.append({
                        "run_id": entry.name,
                        "timestamp": meta.get("timestamp", entry.name),
                        "config": meta.get("config", {}),
                        "path": str(entry),
                    })
                except (json.JSONDecodeError, OSError):
                    runs.append({
                        "run_id": entry.name,
                        "timestamp": entry.name,
                        "config": {},
                        "path": str(entry),
                    })

    # Check for legacy flat layout (results/instances/ without run_meta.json)
    legacy_instances = _results_dir / "instances"
    if legacy_instances.is_dir() and not (_results_dir / "run_meta.json").exists():
        # Only add legacy if it has instance files and wasn't already found
        instance_files = list(legacy_instances.glob("*.json"))
        if instance_files:
            runs.append({
                "run_id": "_legacy",
                "timestamp": "Legacy Run",
                "config": {},
                "path": str(_results_dir),
                "legacy": True,
            })

    return runs


def _load_run_instances(run_path: Path) -> list[dict[str, Any]]:
    """Load all per-instance JSON files from a run directory."""
    instances_dir = run_path / "instances"
    if not instances_dir.exists():
        return []

    results = []
    for fpath in sorted(instances_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _load_run_summary(run_path: Path) -> dict[str, Any] | None:
    """Load the summary.json for a run."""
    summary_path = run_path / "summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


# ------------------------------------------------------------------
# API Routes
# ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the viewer HTML."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/runs")
async def list_runs():
    """List all available runs."""
    return _discover_runs()


@app.get("/api/runs/{run_id}/summary")
async def run_summary(run_id: str):
    """Get summary statistics for a run."""
    runs = _discover_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(404, "Run not found")

    run_path = Path(run["path"])
    summary = _load_run_summary(run_path)

    # Also load run meta
    meta_path = run_path / "run_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {"summary": summary, "meta": meta}


@app.get("/api/runs/{run_id}/instances")
async def run_instances(run_id: str):
    """Get all instance results for a run."""
    runs = _discover_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(404, "Run not found")

    run_path = Path(run["path"])
    instances = _load_run_instances(run_path)

    # For the instance list view, return a condensed version (no conversation/trace)
    condensed = []
    for inst in instances:
        entry = {
            "instance_id": inst.get("instance_id", ""),
            "gatherer": inst.get("gatherer", ""),
            "model": inst.get("model"),
        }
        # Extract metrics — support both new (nested) and legacy (flat) formats
        if "metrics" in inst:
            entry.update(inst["metrics"])
        else:
            # Legacy format: metrics are at the top level
            for k, v in inst.items():
                if k not in ("instance_id", "gatherer", "model"):
                    entry[k] = v
        condensed.append(entry)

    return condensed


@app.get("/api/runs/{run_id}/instances/{instance_id}/{gatherer}")
async def instance_detail(run_id: str, instance_id: str, gatherer: str):
    """Get full detail for a specific instance+gatherer."""
    runs = _discover_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if not run:
        raise HTTPException(404, "Run not found")

    run_path = Path(run["path"])
    instances_dir = run_path / "instances"
    if not instances_dir.exists():
        raise HTTPException(404, "No instances found")

    # Find the matching file
    safe_id = instance_id.replace("/", "__").replace("\\", "__")
    fpath = instances_dir / f"{safe_id}_{gatherer}.json"

    if not fpath.exists():
        raise HTTPException(404, f"Instance {instance_id} / {gatherer} not found")

    try:
        return json.loads(fpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"Error reading instance data: {e}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main():
    """CLI entry point for the results viewer."""
    parser = argparse.ArgumentParser(
        description="Context Gathering Harness — Results Viewer",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve the viewer on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to",
    )
    args = parser.parse_args()

    import uvicorn

    set_results_dir(Path(args.results_dir))
    logger.info("📊 Results viewer at http://%s:%d", args.host, args.port)
    logger.info("📁 Scanning results in %s", args.results_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
