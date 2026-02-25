"""Experiment orchestrator — drives benchmark × gatherer evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from harness.benchmarks.base import BenchmarkAdapter
from harness.dashboard.state import DashboardState
from harness.gatherers.base import ContextGatherer
from harness.metrics.retrieval import compute_all_retrieval_metrics
from harness.metrics.efficiency import compute_efficiency_metrics
from harness.reporting import ResultStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Registry: name → class
# ------------------------------------------------------------------


def _get_adapter(cfg: dict) -> BenchmarkAdapter:
    name = cfg["name"]
    if name in ("swebench_lite", "swebench"):
        from harness.benchmarks.swebench import SWEBenchAdapter

        return SWEBenchAdapter(
            cache_dir=cfg.get("cache_dir"),
            gold_context_strategy=cfg.get("gold_context_strategy", "patch_and_tests"),
        )
    elif name in ("crosscodeeval", "crosscode"):
        from harness.benchmarks.crosscodeeval import CrossCodeEvalAdapter

        return CrossCodeEvalAdapter(
            language=cfg.get("language", "python"),
            cache_dir=cfg.get("cache_dir"),
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def _get_gatherer(cfg: dict, llm_cfg: dict | None = None) -> ContextGatherer:
    name = cfg["name"]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}

    if name == "rag_bm25":
        from harness.gatherers.rag_bm25 import BM25RAGGatherer

        return BM25RAGGatherer(**kwargs)
    elif name == "rag_dense":
        from harness.gatherers.rag_dense import DenseRAGGatherer

        return DenseRAGGatherer(**kwargs)
    elif name == "rag_hybrid":
        from harness.gatherers.rag_hybrid import HybridRAGGatherer

        return HybridRAGGatherer(**kwargs)
    elif name == "react_agent":
        from harness.gatherers.react_agent import ReActGatherer

        return ReActGatherer(llm_config=llm_cfg, **kwargs)
    elif name == "agentless":
        from harness.gatherers.agentless import AgentlessGatherer

        return AgentlessGatherer(llm_config=llm_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown gatherer: {name}")


# ------------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------------


def run_experiment(config_path: str) -> None:
    """Run the full experiment as defined by the YAML config."""
    cfg = _load_config(config_path)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load benchmark
    benchmark = _get_adapter(cfg["benchmark"])
    split = cfg["benchmark"].get("split", "test")
    limit = cfg["benchmark"].get("limit")
    instances = benchmark.load(split=split, limit=limit)
    logger.info("Loaded %d benchmark instances.", len(instances))

    gatherer_cfgs = cfg.get("gatherers", [])
    llm_cfg = cfg.get("llm", {})
    k_values = cfg.get("k_values", [1, 3, 5, 10])
    output_dir = cfg.get("output_dir", "./results")

    total_steps = len(instances) * len(gatherer_cfgs)
    dashboard = DashboardState(total=total_steps)
    store = ResultStore(output_dir)

    # Start live dashboard
    dash_cfg = cfg.get("dashboard", {})
    if dash_cfg.get("enabled", True):
        from harness.dashboard.server import start_dashboard_server

        port = dash_cfg.get("port", 8765)
        start_dashboard_server(dashboard, port=port)
        logger.info("📊 Dashboard live at http://127.0.0.1:%d", port)

    # Run
    for gcfg in gatherer_cfgs:
        gatherer = _get_gatherer(gcfg, llm_cfg)
        logger.info("▶ Running gatherer: %s", gcfg["name"])

        for inst in instances:
            logger.info("  Instance: %s", inst.id)
            result = gatherer.gather(inst)

            # Compute metrics
            retrieval_scores = compute_all_retrieval_metrics(
                result.retrieved_contexts,
                inst.gold_context,
                k_values=k_values,
            )
            patch_scores = benchmark.evaluate_patch(inst, result.generated_patch)
            efficiency = compute_efficiency_metrics(result)

            all_metrics = {**retrieval_scores, **patch_scores, **efficiency}

            # Store
            store.store(inst.id, gcfg["name"], all_metrics)

            # Update dashboard
            dashboard.record(
                gatherer_name=gcfg["name"],
                token_usage=result.token_usage,
                latency_s=result.latency_s,
                metrics=retrieval_scores,
                instance_id=inst.id,
            )

    # Generate final report
    report_dir = store.generate_report()
    logger.info("✅ Experiment complete. Results in %s", report_dir)


def _load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Context Gathering Testing Harness",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
