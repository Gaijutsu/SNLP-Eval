"""Experiment orchestrator — drives benchmark × gatherer evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from typing import Any

import yaml

from harness.benchmarks.base import BenchmarkAdapter
from harness.dashboard.state import DashboardState
from harness.gatherers.base import ContextGatherer
from harness.metrics.efficiency import compute_efficiency_metrics
from harness.metrics.retrieval import compute_all_retrieval_metrics
from harness.reporting import ResultStore

logger = logging.getLogger(__name__)


def _get_adapter(cfg: dict) -> BenchmarkAdapter:
    name = cfg["name"]
    if name in ("swebench_lite", "swebench"):
        from harness.benchmarks.swebench import SWEBenchAdapter

        return SWEBenchAdapter(cache_dir=cfg.get("cache_dir"))
    elif name in ("crosscodeeval", "crosscode"):
        from harness.benchmarks.crosscodeeval import CrossCodeEvalAdapter

        return CrossCodeEvalAdapter(
            language=cfg.get("language", "python"),
            cache_dir=cfg.get("cache_dir"),
        )
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
    raise ValueError(f"Unknown gatherer: {name}")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "datasets",
        "fsspec",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message="You are sending unauthenticated requests to the HF Hub.*",
    )


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _progress_interval(num_instances: int) -> int:
    """Choose a sparse, readable logging interval."""
    if num_instances <= 10:
        return 1
    if num_instances <= 50:
        return 10
    if num_instances <= 200:
        return 25
    return 50


def run_experiment(config_path: str) -> None:
    cfg = _load_config(config_path)
    _setup_logging()

    benchmark = _get_adapter(cfg["benchmark"])
    split = cfg["benchmark"].get("split", "test")
    limit = cfg["benchmark"].get("limit")
    instances = benchmark.load(split=split, limit=limit)

    gatherer_cfgs = cfg.get("gatherers", [])
    llm_cfg = cfg.get("llm", {})
    k_values = cfg.get("k_values", [1, 3, 5, 10])
    output_dir = cfg.get("output_dir", "./results")

    total_steps = len(instances) * len(gatherer_cfgs)
    dashboard = DashboardState(total=total_steps)
    store = ResultStore(output_dir)

    logger.info(
        "Starting run: %d instances, %d gatherer%s",
        len(instances),
        len(gatherer_cfgs),
        "" if len(gatherer_cfgs) == 1 else "s",
    )

    dash_cfg = cfg.get("dashboard", {})
    if dash_cfg.get("enabled", True):
        from harness.dashboard.server import start_dashboard_server

        port = dash_cfg.get("port", 8765)
        start_dashboard_server(dashboard, port=port)
        logger.info("Dashboard available at http://127.0.0.1:%d", port)

    global_step = 0
    per_gatherer_interval = _progress_interval(len(instances))

    for gcfg in gatherer_cfgs:
        gatherer_name = gcfg["name"]
        gatherer = _get_gatherer(gcfg, llm_cfg)

        logger.info("Running %s...", gatherer_name)

        gatherer_latencies: list[float] = []
        gatherer_tokens: list[int] = []

        for idx, inst in enumerate(instances, start=1):
            global_step += 1

            result = gatherer.gather(inst)

            retrieval_scores = compute_all_retrieval_metrics(
                result.retrieved_contexts,
                inst.gold_context,
                k_values=k_values,
            )
            patch_scores = benchmark.evaluate_patch(inst, result.generated_patch)
            efficiency = compute_efficiency_metrics(result)

            all_metrics = {**retrieval_scores, **patch_scores, **efficiency}
            store.store(inst.id, gatherer_name, all_metrics)

            dashboard.record(
                gatherer_name=gatherer_name,
                token_usage=result.token_usage,
                latency_s=result.latency_s,
                metrics=retrieval_scores,
                instance_id=inst.id,
            )

            gatherer_latencies.append(result.latency_s)
            gatherer_tokens.append(result.token_usage or 0)

            if idx % per_gatherer_interval == 0 or idx == len(instances):
                logger.info(
                    "  %s: %d/%d complete",
                    gatherer_name,
                    idx,
                    len(instances),
                )

        avg_latency = (
            sum(gatherer_latencies) / len(gatherer_latencies)
            if gatherer_latencies
            else 0.0
        )
        avg_tokens = (
            sum(gatherer_tokens) / len(gatherer_tokens)
            if gatherer_tokens
            else 0.0
        )

        logger.info(
            "Finished %s — avg latency %.2fs, avg tokens %.0f",
            gatherer_name,
            avg_latency,
            avg_tokens,
        )

    report_dir = store.generate_report()
    logger.info("Run complete. Results saved to %s", report_dir)


def main() -> None:
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