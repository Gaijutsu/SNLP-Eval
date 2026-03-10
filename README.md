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
  - [Results Viewer](#results-viewer)
  - [LLM Client](#llm-client)
- [Gold Set Construction](#gold-set-construction)
  - [Strategies](#strategies)
  - [Import Graph Expansion](#import-graph-expansion)
- [Metrics Reference](#metrics-reference)
  - [Retrieval Metrics](#retrieval-metrics)
  - [Patch Metrics](#patch-metrics)
  - [Efficiency Metrics](#efficiency-metrics)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Viewing Results](#viewing-results)
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
| **Enriched results** | Per-instance JSON captures retrieved documents, gold context, LLM conversations, agent traces, and model info — enabling re-processing and deeper analysis |
| **Versioned runs** | Each experiment creates a timestamped directory; previous runs are never overwritten |
| **Results viewer** | Standalone web app to explore past runs — sortable instance tables, document comparison, conversation replay, trace timeline |
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
│       ├── reporting.py             #   Versioned result storage, CSV export, summaries
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
│       ├── dashboard/               #   Live experiment dashboard
│       │   ├── server.py            #     FastAPI + WebSocket server
│       │   ├── state.py             #     Thread-safe experiment state
│       │   └── static/index.html    #     Dashboard frontend
│       └── viewer/                  #   Post-hoc results viewer
│           ├── app.py               #     FastAPI server + REST API
│           └── static/index.html    #     Viewer frontend (SPA)
├── tests/                           # Test suite
│   ├── test_metrics.py              #   Retrieval metric unit tests
│   ├── test_adapters.py             #   Benchmark adapter smoke tests
│   └── test_reporting.py            #   ResultStore + enriched output tests
├── results/                         # Experiment outputs (versioned by timestamp)
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
| `benchmarks/swebench.py` | `SWEBenchAdapter` | Loads instances from the HuggingFace `princeton-nlp/SWE-bench_Lite` dataset. Clones each repo at the correct base commit, builds gold context using a configurable strategy (see [Gold Set Construction](#gold-set-construction)), and supports `git apply` + test evaluation. |
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
as it arrives, creates a **versioned run directory**
(`results/<YYYY-MM-DD_HHMMSS>/`), and writes:

- `run_meta.json` — config snapshot, timestamp, and run identifier
- `instances/<id>_<gatherer>.json` — **enriched** per-instance JSON with:
  - Computed metrics (precision, recall, MRR, NDCG, etc.)
  - Retrieved documents list and gold context (for re-computing metrics)
  - Full LLM conversation history (for agentic gatherers)
  - Agent trace (step-by-step thought/action/observation log)
  - Model name, token usage, latency, generated patch
- `results.csv` — flat CSV with all metrics across all gatherers
- `summary.json` — mean ± std per metric, grouped by gatherer

Previous runs are **never overwritten** — each experiment creates its own
timestamped directory.

### Results Viewer

**`harness/viewer/`** — A standalone FastAPI web application for exploring
completed experiment results. Launched via:

```bash
harness-viewer --results-dir ./results --port 8080
```

The viewer provides three navigable views:

1. **Runs list** — shows all versioned runs with config summary (benchmark,
   model, gatherers). Also detects legacy flat results.
2. **Run dashboard** — per-gatherer summary cards (mean ± std for key metrics),
   plus a **sortable, filterable instance table** where you can click any column
   header to sort by that metric and filter by gatherer or instance ID.
3. **Instance detail** — deep-dive into a single instance showing:
   - Metrics grid with colour-coded values
   - Retrieved documents vs. gold context (hits/misses highlighted)
   - LLM conversation in a chat-style view
   - Agent trace as a step-by-step timeline
   - Generated patch with diff syntax highlighting

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

## Gold Set Construction

Retrieval metrics (Precision@K, Recall@K, NDCG@K, MRR) compare each gatherer's
retrieved file list against a **gold set** of ground-truth relevant files.  The
quality and size of this gold set directly affects how informative the metrics
are.

For the SWE-bench Lite benchmark, each instance comes with a **reference patch**
(the official fix) and a **test patch** (the tests that verify the fix). The
harness constructs the gold set from these artefacts using one of three
cumulative strategies, controlled by the `gold_context_strategy` config key.

### Strategies

| Strategy | Gold set contains | Typical size | When to use |
|----------|-------------------|-------------|-------------|
| `patch_only` | Files modified in the reference patch | 1–2 files | Strict evaluation: only the exact fix location counts |
| **`patch_and_tests`** *(default)* | Patch files **+** files from the test patch | 2–5 files | Balanced: rewards finding both the code-to-fix and the relevant test files |
| `patch_tests_and_imports` | Above **+** first-party imports of the patched files | 4–15 files | Generous: credits structurally related files (interfaces, utilities, base classes) that provide useful context |

**Why not `patch_only`?** In SWE-bench Lite, ~78 % of issues touch only **one**
file, which collapses Precision@K to a binary 0 or 1/K and causes Recall@K to
saturate at 1.0 as soon as one file is found.  Including test files and imports
gives the metrics more discrimination power.

#### Example

For a Django issue whose fix modifies `django/db/models/query.py`:

| Strategy | Gold files |
|----------|------------|
| `patch_only` | `django/db/models/query.py` |
| `patch_and_tests` | `django/db/models/query.py`, `tests/queries/test_qs_combinators.py` |
| `patch_tests_and_imports` | …the above + `django/db/models/sql/query.py`, `django/db/models/manager.py`, `django/db/models/__init__.py`, etc. |

### Import Graph Expansion

The `patch_tests_and_imports` strategy extends the gold set by parsing the
`import` / `from … import` statements in each patched file and resolving them
to concrete files that exist in the repository.  Key design decisions:

- **First-party only** — only modules that map to an actual file in the repo
  are included; standard-library and third-party imports are ignored.
- **Depth-1 only** — only direct imports of patched files are added (no
  transitive closure) to keep the gold set focused.
- **AST-based parsing** — uses Python's `ast` module for reliable extraction.
  Falls back to a regex pattern when a file cannot be parsed (e.g. Python 2
  syntax).
- **Module → path mapping** — dotted names (e.g. `django.db.models`) are
  converted to candidate paths (`django/db/models.py` and
  `django/db/models/__init__.py`) and also tried relative to the source
  file's directory.

---

## Metrics Reference

The harness captures three families of metrics for every `(instance, gatherer)` pair.
Retrieval metrics are computed for all gatherers; patch metrics only when a candidate
patch is produced (agentic gatherers); efficiency metrics are always recorded.

### Retrieval Metrics

Computed in `metrics/retrieval.py`.  All retrieval metrics compare the gatherer's
ranked list of **retrieved files** against the **gold set** of ground-truth relevant
files (see [Gold Set Construction](#gold-set-construction)).  *K* values are
configurable via the `k_values` config key (default: 1, 3, 5, 10).

| Metric | Key | Range | Definition |
|--------|-----|-------|------------|
| **Precision@K** | `precision@{k}` | [0, 1] | Fraction of the top-*K* retrieved files that are in the gold set.  Answers: *"Of the files I returned, how many were actually relevant?"* |
| **Recall@K** | `recall@{k}` | [0, 1] | Fraction of gold-set files found in the top-*K* results.  Answers: *"Of all the relevant files, how many did I find?"*  Returns 0 when the gold set is empty. |
| **MRR** | `mrr` | (0, 1] or 0 | Mean Reciprocal Rank — `1 / rank` of the **first** relevant result in the list.  Higher is better; 1.0 means the first result is relevant.  Returns 0 if no relevant file appears at all. |
| **NDCG@K** | `ndcg@{k}` | [0, 1] | Normalized Discounted Cumulative Gain at *K*.  Uses **binary relevance** (1 if in gold set, else 0).  Rewards placing relevant files earlier (discounted by `1 / log₂(rank + 1)`), normalized against the ideal ranking. |

> [!NOTE]
> All retrieval functions deduplicate the retrieved list before scoring, so
> duplicate file paths do not inflate results.

### Patch Metrics

Computed in `metrics/patch.py`.  These evaluate the quality of a generated
patch (unified diff) against the reference patch from the benchmark.

| Metric | Key | Range | Definition |
|--------|-----|-------|------------|
| **Edit Similarity** | `edit_similarity` | [0, 1] | `SequenceMatcher` ratio between the candidate and gold patches as raw text. 1.0 = identical patches; 0.0 = completely different (or either patch is missing). |
| **Applied** | `applied` | bool | Whether the candidate patch applies cleanly via `git apply --check` against the repo snapshot. |
| **Tests Passed** | `tests_passed` | bool | Whether the repository's test suite passes after applying the candidate patch. Only meaningful when `applied` is true. |
| **Fail-to-Pass** | `fail_to_pass` | 0 or 1 | 1 if previously-failing tests now pass after the patch; 0 otherwise. |
| **Pass-to-Pass** | `pass_to_pass` | 0 or 1 | 1 if previously-passing tests still pass after the patch; 0 otherwise. |

> [!IMPORTANT]
> Patch metrics require a repo snapshot on disk and are currently only
> supported for SWE-bench instances.  The `apply_and_test_patch` function
> automatically reverts the patch after testing.

### Efficiency Metrics

Computed in `metrics/efficiency.py`.  Extracted directly from the `GatherResult`
returned by each gatherer.

| Metric | Key | Unit | Definition |
|--------|-----|------|------------|
| **Token Usage** | `token_usage` | tokens | Total number of LLM tokens consumed (prompt + completion) during context gathering.  Zero for non-LLM gatherers (e.g. BM25). |
| **Latency** | `latency_s` | seconds | Wall-clock time from the start to the end of the `gather()` call. |
| **Time-to-First-Token** | `ttft_s` | seconds | Time from the request being issued to receiving the first token back from the LLM.  `null` for non-streaming or non-LLM gatherers. |

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

> [!TIP]
> **Disabling `<think>` blocks for Qwen3 models.**  Qwen3 models produce
> `<think>...</think>` chain-of-thought blocks by default.  While the harness
> strips these before parsing, they consume significant tokens (~30–40% of
> output).  To disable thinking in Ollama, pass `/nothink` at the start of
> your prompt, or create a custom Modelfile:
>
> ```
> FROM qwen3:8b
> PARAMETER num_ctx 16384
> SYSTEM "You are a helpful assistant. /nothink"
> ```
>
> Then `ollama create qwen3-nothink -f Modelfile` and use `qwen3-nothink`
> as your model name in the config.

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
  # How to build the ground-truth file set for retrieval metrics.
  # Options: patch_only | patch_and_tests | patch_tests_and_imports
  gold_context_strategy: patch_and_tests

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
2. Create a versioned run directory (e.g. `results/2026-02-26_130000/`).
3. Save a config snapshot as `run_meta.json`.
4. For each `gatherer × instance`, gather context and compute metrics.
5. Write enriched per-instance JSONs (documents, conversations, traces).
6. Stream live progress to the dashboard at `http://127.0.0.1:8765`.
7. Generate `results.csv` + `summary.json` in the run directory.

---

## Viewing Results

After one or more experiments have completed, launch the results viewer:

```bash
# Point the viewer at your results directory
harness-viewer --results-dir ./results --port 8080

# Or invoke via Python module
python -m harness.viewer.app --results-dir ./results
```

Open **`http://127.0.0.1:8080`** to browse:

- **Runs list** — select which experiment run to explore.
- **Run dashboard** — compare gatherers side-by-side and drill into the
  sortable instance table.
- **Instance detail** — inspect retrieved documents vs. gold context,
  replay the LLM conversation, follow the agent trace, and view patches.

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
- **`test_reporting.py`** — Tests for the `ResultStore` including versioned
  directories, enriched per-instance JSON output, run metadata, report
  generation, and backwards compatibility with the legacy format.

---

## Output & Results

After an experiment completes, the `output_dir` (default `./results`) will
contain a **versioned run directory**:

```
results/
├── 2026-02-26_130000/              # One directory per run (never overwritten)
│   ├── run_meta.json               # Config snapshot + timestamp
│   ├── results.csv                 # All metrics, one row per (instance, gatherer)
│   ├── summary.json                # Per-gatherer aggregated stats
│   └── instances/                  # Enriched per-instance JSON files
│       ├── django__django-11001_rag_bm25.json
│       ├── django__django-11001_react_agent.json
│       └── ...
└── 2026-02-27_091500/              # Next run — separate directory
    └── ...
```

### Enriched Per-Instance JSON

Each instance file now contains full detail for re-processing and analysis:

```json
{
  "instance_id": "django__django-11001",
  "gatherer": "react_agent",
  "model": "Qwen3-8B",
  "metrics": {
    "precision@1": 1.0,
    "recall@5": 0.25,
    "mrr": 1.0,
    "latency_s": 27.6,
    "token_usage": 9438
  },
  "retrieved_documents": ["django/db/models/query.py", "..." ],
  "gold_context": ["django/db/models/query.py", "tests/queries/test_qs.py"],
  "conversation": [
    {"role": "system", "content": "You are a code investigation agent..."},
    {"role": "user", "content": "Find the files..."},
    {"role": "assistant", "content": "Thought: I'll search for..."}
  ],
  "trace": [
    {"step": 1, "action": "grep", "args": ["QuerySet"], "tokens": 500}
  ],
  "generated_patch": "--- a/django/db/models/query.py\n+++ ..."
}
```

The console also prints a rich-formatted summary table:

```
 📊 rag_bm25
┌────────────────┬────────┬────────┬───┐
│ Metric         │   Mean │    Std │ N │
├────────────────┼────────┼────────┼───┤
│ mrr            │ 0.3521 │ 0.2814 │50 │
│ precision@1    │ 0.2800 │ 0.4536 │50 │
│ recall@5       │ 0.5120 │ 0.3127 │50 │
│ ...            │    ... │    ...http://localhost:11434/v1 │...│
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
