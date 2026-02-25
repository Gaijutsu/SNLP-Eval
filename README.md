# Context-Gathering Harness

> **Testing harness comparing Agentic vs RAG context-gathering strategies for LLMs on code.**

This project provides an end-to-end evaluation framework that measures how well
different context-gathering approaches (sparse retrieval, dense retrieval,
hybrid retrieval, ReAct agents, and Agentless localisation) help an LLM find
the right files — and optionally produce correct patches — on established
software-engineering benchmarks.

---

## Table of Contents

- [Key Features](#key-features)
- [Project Layout](#project-layout)
- [Module Descriptions](#module-descriptions)
  - [Runner & Orchestration](#runner--orchestration)
  - [Benchmarks](#benchmarks)
  - [Gatherers](#gatherers)
  - [Metrics](#metrics)
  - [Dashboard](#dashboard)
  - [Reporting](#reporting)
  - [LLM Client](#llm-client)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Running Tests](#running-tests)
- [Output & Results](#output--results)
- [Extending the Harness](#extending-the-harness)

---

## Key Features

| Capability | Details |
|---|---|
| **Multiple retrieval strategies** | BM25 (sparse), Dense (sentence-transformers + FAISS), Hybrid (BM25 + Dense with RRF), ReAct agent, Agentless (3-phase localisation + repair) |
| **Benchmark adapters** | SWE-bench Lite, CrossCodeEval — pluggable via a common `BenchmarkAdapter` ABC |
| **Retrieval metrics** | Precision@K, Recall@K, MRR, NDCG@K |
| **Patch metrics** | Edit similarity, git-apply + test pass rate (SWE-bench only) |
| **Efficiency metrics** | Token usage, latency, time-to-first-token |
| **Live dashboard** | FastAPI + WebSocket real-time progress & metric charts |
| **Flexible LLM backend** | OpenAI API *or* any OpenAI-compatible local server (Ollama, vLLM, llama.cpp) |
| **YAML-driven configs** | One file controls benchmark, gatherers, LLM, metrics, and output |

---

## Project Layout

```
SNLP-Eval/
├── config/                          # Experiment configuration files
│   ├── default.yaml                 #   Full experiment (all gatherers)
│   ├── test_agentless.yaml          #   Agentless-only quick run
│   ├── test_bm25.yaml              #   BM25-only quick run
│   └── test_dense.yaml             #   Dense-only quick run
├── src/
│   └── harness/                     # Main Python package
│       ├── __init__.py              #   Package marker + version
│       ├── runner.py                #   CLI entry-point & experiment orchestrator
│       ├── llm_client.py            #   Unified LLM client (OpenAI / local)
│       ├── reporting.py             #   Result storage, CSV export, summary tables
│       ├── benchmarks/              #   Benchmark adapters
│       │   ├── base.py              #     ABC + BenchmarkInstance dataclass
│       │   ├── swebench.py          #     SWE-bench Lite adapter
│       │   └── crosscodeeval.py     #     CrossCodeEval adapter
│       ├── gatherers/               #   Context-gathering strategies
│       │   ├── base.py              #     ABC + GatherResult dataclass
│       │   ├── rag_bm25.py          #     BM25 sparse retrieval
│       │   ├── rag_dense.py         #     Dense embedding retrieval (FAISS)
│       │   ├── rag_hybrid.py        #     BM25 + Dense with Reciprocal Rank Fusion
│       │   ├── react_agent.py       #     ReAct-style agentic gatherer
│       │   └── agentless.py         #     Agentless 3-phase (localise → repair → validate)
│       ├── metrics/                 #   Evaluation metrics
│       │   ├── retrieval.py         #     Precision@K, Recall@K, MRR, NDCG@K
│       │   ├── patch.py             #     Patch similarity & apply-and-test
│       │   └── efficiency.py        #     Token usage, latency, TTFT
│       └── dashboard/               #   Live experiment dashboard
│           ├── server.py            #     FastAPI + WebSocket server
│           ├── state.py             #     Thread-safe experiment state
│           └── static/
│               └── index.html       #     Dashboard frontend
├── tests/                           # Test suite
│   ├── test_metrics.py              #   Retrieval metric unit tests
│   └── test_adapters.py             #   Benchmark adapter smoke tests
├── results/                         # Experiment outputs (generated)
├── pyproject.toml                   # Build config & dependencies
└── .gitignore
```

---

## Module Descriptions

### Runner & Orchestration

**`harness/runner.py`** — The main entry-point. Parses CLI arguments, loads the
YAML config, instantiates the benchmark adapter and each gatherer, then drives
the `benchmark × gatherer` evaluation loop. After all instances are processed it
delegates to `ResultStore` for final reporting.

**CLI usage:**

```bash
harness --config config/default.yaml
```

### Benchmarks

| Module | Class | Description |
|---|---|---|
| `benchmarks/base.py` | `BenchmarkAdapter` (ABC), `BenchmarkInstance` | Defines the interface every benchmark must implement: `load()` returns instances; `evaluate_patch()` scores a candidate patch. |
| `benchmarks/swebench.py` | `SWEBenchAdapter` | Loads instances from the HuggingFace `princeton-nlp/SWE-bench_Lite` dataset. Clones each repo at the correct base commit, derives gold context from the reference patch, and supports `git apply` + test evaluation. |
| `benchmarks/crosscodeeval.py` | `CrossCodeEvalAdapter` | Loads from `microsoft/CrossCodeEval`. Evaluates cross-file code completion — retrieval metrics only (no patch evaluation). |

### Gatherers

All gatherers extend `ContextGatherer` and return a `GatherResult` containing
retrieved file paths, token usage, latency, optional patch text, and a
step-by-step trace.

| Module | Class | Strategy |
|---|---|---|
| `gatherers/rag_bm25.py` | `BM25RAGGatherer` | Indexes repo files with a custom BM25-Okapi implementation (inverted index, IDF, camelCase-aware tokeniser). Zero LLM tokens. |
| `gatherers/rag_dense.py` | `DenseRAGGatherer` | Embeds files with a SentenceTransformer model (default `all-MiniLM-L6-v2`), then ranks by cosine similarity. Supports CUDA. |
| `gatherers/rag_hybrid.py` | `HybridRAGGatherer` | Runs both BM25 and Dense retrieval, then fuses results via **Reciprocal Rank Fusion** (RRF, *k* = 60). |
| `gatherers/react_agent.py` | `ReActGatherer` | Multi-step Thought → Action → Observation loop. The LLM uses tools (`list_dir`, `read_file`, `grep`) to explore the repo and returns relevant files via a `finish(...)` action. |
| `gatherers/agentless.py` | `AgentlessGatherer` | 3-phase pipeline inspired by *Xia et al., 2024*: (1) file-level localisation, (2) function-level localisation, (3) patch generation. Generates `n_samples` candidate diffs. |

### Metrics

| Module | Functions | What it measures |
|---|---|---|
| `metrics/retrieval.py` | `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`, `compute_all_retrieval_metrics` | Standard information-retrieval quality metrics at configurable *K* values. |
| `metrics/patch.py` | `patch_similarity`, `apply_and_test_patch` | SequenceMatcher edit similarity to gold; applies candidate via `git apply`, runs tests, reports pass/fail. |
| `metrics/efficiency.py` | `compute_efficiency_metrics` | Extracts `token_usage`, `latency_s`, and `ttft_s` (time-to-first-token) from the `GatherResult`. |

### Dashboard

A **FastAPI + WebSocket** live dashboard launched in a background daemon thread
during experiments:

- **`dashboard/state.py`** — Thread-safe `DashboardState` accumulates
  per-gatherer running stats (precision, recall, NDCG, MRR, tokens, latency)
  and exposes a JSON `snapshot()` for the frontend.
- **`dashboard/server.py`** — Serves `static/index.html` at the root and pushes
  state snapshots to connected WebSocket clients every second.
- **`dashboard/static/index.html`** — Single-page frontend with progress bars,
  live metric charts, and a token-usage timeline.

By default the dashboard runs at **`http://127.0.0.1:8765`**.

### Reporting

**`harness/reporting.py`** (`ResultStore`) — Collects every per-instance result
as it arrives, writes individual JSON files to `results/instances/`, and at the
end generates:

- `results/results.csv` — flat CSV with all metrics across all gatherers
- `results/summary.json` — mean ± std per metric, grouped by gatherer
- A rich-formatted summary table printed to the console

### LLM Client

**`harness/llm_client.py`** (`LLMClient`) — Thin wrapper around the OpenAI
Python SDK. Supports:

- **Cloud** (OpenAI) — set `provider: openai`, optionally pass `api_key`.
- **Local** (Ollama, vLLM, llama.cpp, etc.) — set `provider: local` and
  `base_url` to the OpenAI-compatible endpoint.
- Streaming and non-streaming completions.
- Automatic token counting via `tiktoken` (falls back to `cl100k_base` for
  unknown models).

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| Git | any recent version (used by SWE-bench adapter to clone repos) |
| *(Optional)* Ollama or other local LLM server | for `provider: local` |
| *(Optional)* CUDA + GPU | for faster dense embeddings / FAISS |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url> SNLP-Eval
cd SNLP-Eval
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

```bash
# Standard install (CPU)
pip install -e .

# With GPU support (FAISS-GPU)
pip install -e ".[gpu]"

# With dev/test dependencies
pip install -e ".[dev]"
```

### 4. *(Optional)* Set up a local LLM

If using a local provider, start an OpenAI-compatible server.  For example with
[Ollama](https://ollama.com/):

```bash
ollama pull qwen3:0.6b          # or any model you want
ollama serve                     # exposes http://localhost:11434/v1
```

### 5. *(Optional)* Set OpenAI API key

If using the OpenAI cloud provider:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Configuration

Experiments are driven by YAML config files in `config/`. The schema:

```yaml
benchmark:
  name: swebench_lite        # swebench_lite | crosscodeeval
  split: test                # dataset split
  limit: 50                  # cap instance count (null for all)

llm:
  provider: local            # local | openai
  model: qwen3:0.6b         # model name / HF path
  base_url: http://localhost:11434/v1   # null for OpenAI cloud

gatherers:                   # list of strategies to evaluate
  - name: rag_bm25
    top_k: 10
  - name: rag_dense
    model: all-MiniLM-L6-v2
    top_k: 10
  - name: rag_hybrid
    model: all-MiniLM-L6-v2
    top_k: 10
  - name: react_agent
    max_steps: 15
  - name: agentless
    n_samples: 5

k_values: [1, 3, 5, 10]     # K cut-offs for retrieval metrics

dashboard:
  enabled: true
  port: 8765

output_dir: ./results
```

Preconfigured quick-test configs are provided:

| File | Runs |
|---|---|
| `config/default.yaml` | All gatherers (full experiment) |
| `config/test_bm25.yaml` | BM25 only |
| `config/test_dense.yaml` | Dense only |
| `config/test_agentless.yaml` | Agentless only |

---

## Running Experiments

```bash
# Run the default experiment
harness --config config/default.yaml

# Run a single-gatherer quick test
harness --config config/test_bm25.yaml

# Or invoke via Python module
python -m harness.runner --config config/test_dense.yaml
```

Once launched the harness will:

1. Load benchmark instances (cloning repos for SWE-bench on first run).
2. For each `gatherer × instance`, gather context and compute metrics.
3. Stream live progress to the dashboard at `http://127.0.0.1:8765`.
4. Write per-instance JSONs and a final `results.csv` + `summary.json`.

---

## Running Tests

```bash
# Install dev dependencies first
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_metrics.py
pytest tests/test_adapters.py
```

The test suite covers:

- **`test_metrics.py`** — Unit tests for all retrieval metrics
  (`precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`, `compute_all_retrieval_metrics`),
  including edge cases (empty inputs, `k = 0`, partial overlaps).
- **`test_adapters.py`** — Smoke tests for the benchmark adapter schema
  (`BenchmarkInstance` creation, default `evaluate_patch`, SWE-bench patch
  extraction).

---

## Output & Results

After an experiment completes, the `output_dir` (default `./results`) will
contain:

```
results/
├── results.csv              # All metrics, one row per (instance, gatherer)
├── summary.json             # Per-gatherer aggregated stats (mean, std, min, max)
└── instances/               # Individual per-instance JSON files
    ├── instance_id__gatherer.json
    └── ...
```

The console will also print a rich-formatted summary table, for example:

```
 📊 rag_bm25
┌────────────────┬────────┬────────┬───┐
│ Metric         │   Mean │    Std │ N │
├────────────────┼────────┼────────┼───┤
│ mrr            │ 0.3521 │ 0.2814 │50 │
│ precision@1    │ 0.2800 │ 0.4536 │50 │
│ recall@5       │ 0.5120 │ 0.3127 │50 │
│ ...            │    ... │    ... │...│
└────────────────┴────────┴────────┴───┘
```

---

## Extending the Harness

### Adding a new gatherer

1. Create a new file in `src/harness/gatherers/`, e.g. `my_gatherer.py`.
2. Subclass `ContextGatherer` and implement the `gather()` method:

   ```python
   from harness.gatherers.base import ContextGatherer, GatherResult
   from harness.benchmarks.base import BenchmarkInstance

   class MyGatherer(ContextGatherer):
       name = "my_gatherer"

       def gather(self, instance: BenchmarkInstance) -> GatherResult:
           # Your logic here
           return GatherResult(retrieved_contexts=[...])
   ```

3. Register it in `runner.py`'s `_get_gatherer()` function.
4. Add it to a YAML config under `gatherers:`.

### Adding a new benchmark

1. Create a new file in `src/harness/benchmarks/`.
2. Subclass `BenchmarkAdapter` and implement `load()` (and optionally
   `evaluate_patch()`).
3. Register it in `runner.py`'s `_get_adapter()` function.

---

## License

*Not yet specified.*
