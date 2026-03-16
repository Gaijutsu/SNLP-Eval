RAG Subteam Plan (Week Focus: Build the RAG
Baseline) — Scaffold SNLP-Eval Version
0. Purpose
This document defines what the RAG subteam will deliver this week and how our work plugs into the wider project
(Agentic exploration vs RAG for repository-scale code tasks). The priority is to ship a minimal, reproducible RAG
baseline and a fault-localisation evaluation (Recall@K) that runs inside the shared scaffold repository, so we have
concrete results for the next progress report and TA meeting.
1. Current Scope (this week)
Must-have (Week focus):
●
●
●
A working RAG retrieval pipeline (file-level retrieval first) integrated into the scaffold repo
A working evaluation run that reports Recall@K over retrieved files
Baseline results on a fixed small subset (e.g., 30 tasks)
Optional (only if time remains after must-haves):
●
●
●
Add a simple reranker
Extend retrieval units beyond files (functions/snippets)
Downstream patching (% resolved/tests-pass) later (not owned by the RAG subteam this week)
2. Shared Definitions (locked)
●
●
●
●
●
Retrieval unit (Week 1): file paths (repo-relative)
Query text: natural-language task description (issue/task statement)
Ground truth (Week 1): set of files modified in the gold patch (or equivalent oracle file set per dataset)
Primary metric: Recall@K for K ∈ {5, 10, 20}
Run protocol (Week 1): fixed subset of N=30 tasks for iteration; scale to N=100 when stable
3. Scaffold Repo Integration Rules
We will use the scaffold as the base. To avoid spending time on harness complexity, we agree:
1. No folder renames / restructuring. We keep the scaffold folder names exactly as they are.
2.
“Proof-of-life” run by Day 2: one command must run end-to-end on ~5 tasks and write a results/
artifact (CSV/JSON) containing Recall@K.
3. Golden config + command: RAG subteam will add/maintain a single baseline config for Week 1 (BM25).
4. Fallback policy: if the harness blocks progress for >1 day, we add a minimal standalone runner under
scripts/ inside the same repo that outputs Recall@K, and integrate into the harness later.
5. Dashboard/reporting is non-blocking: dashboards/extra reporting must not be required for proof-of-life.
4. Where RAG Work Lives in the Scaffold (explicit mapping)
We do not change these folders; we work within them:
●
●
●
●
●
●
Configs: config/ (RAG subteam adds/maintains a “golden” config)
RAG gatherers / retrieval implementations: src/harness/gatherers/
○
e.g., rag_bm25.py, rag_dense.py, rag_hybrid.py (names as in scaffold)
Metrics (Recall@K): src/harness/metrics/
○
e.g., retrieval.py (Recall@K implementation / reporting)
Benchmark adapters + ground-truth extraction: src/harness/benchmarks/
○
dataset adapter(s) define query_text, task subset, and gold_files
Run entrypoint / orchestration: src/harness/runner.py (or scaffold’s runner module)
Outputs: results/ (must contain machine-readable results such as CSV/JSON)
Ownership marker: add a short comment header in the RAG gatherers indicating “Owned by RAG subteam
(Mert/Sara/Jonas)” for clarity.
5. Ownership and Responsibilities (3-person split)
Mert: RAG Baseline + Evaluation Lead
Deliverables:
1. RAG baseline spec + integration
○
Define the baseline retrieval behaviour (BM25 first; dense/hybrid as ablations)
○
Ensure gatherer produces ranked file paths compatible with evaluation
2. Evaluation output
○
Recall@K for K=5/10/20 over files
○
Produce the first results table (and optional plot)
3. Two cheap ablations (only after baseline works)
○
BM25 vs dense OR BM25 vs hybrid
○
K sensitivity (already included)
4. Write-up contribution
○
methods paragraph: retrieval + Recall@K definition
○
early results paragraph + 3–5 failure mode notes
Definition of done (end of week):
●
A reproducible run inside the scaffold repo that outputs Recall@K results for ≥2 retrieval variants on N=30
tasks.
Sara: Dataset Plumbing + Ground Truth Extraction
Deliverables:
1. Stable task subset file (N=30) committed in the repo (location consistent with scaffold)
2. Extract and store per task:
○
task_id
○
repo_id / repo path
○
query_text
○
gold_files (files modified in gold patch / oracle set)
3. Sanity stats: avg #gold files; distribution; any parsing caveats
Definition of done:
●
A single clean dataset artifact (JSON/CSV) used by the benchmark adapter, plus sanity stats committed.
Jonas: Indexing + Retrieval Engine (BM25 / Dense / Optional Rerank)
Deliverables:
1. BM25 index over file contents per repo (baseline)
2. Retrieval function used by gatherer:
○
retrieve(repo_id, query_text, k) -> ranked list of file paths
3. Engineering basics:
○
caching
○
runtime logs
○
deterministic behaviour where relevant
Definition of done:
●
Working BM25 retrieval integrated into the scaffold gatherer, tested end-to-end on ≥5 tasks via
proof-of-life run.
6. Proof-of-life Run (Day 2 target)
Goal: confirm the scaffold harness can run and output Recall@K.
Requirement: one command runs on ~:w5 tasks and produces a results/ file.
Example (adjust to the scaffold’s actual invocation):
●
python -m harness.runner --config config/rag_baseline_week1.yaml
Expected output:
●
results/recall_at_k.csv (or a single results/results.json containing Recall@K values)
7. Timeline (this week)
Day 1–2
●
●
●
Sara finalises task subset + gold
_
files extraction (N=30)
Jonas stands up BM25 retrieval + gatherer integration
Mert plugs evaluation + produces proof-of-life output on 5 tasks
Day 3–4
●
●
●
Run baseline results on N=30
Debug issues (path normalisation, repo mapping)
Start 1 ablation (BM25 vs dense OR BM25 vs hybrid) if stable
Day 5
●
●
Final baseline table: Recall@5/10/20 (N=30; scale to N=100 if stable)
Short failure mode notes + next steps for the full group meeting
8. Risk Controls
●
●
●
●
Scope control: file-level retrieval only until stable.
Avoid overengineering: dashboards and extended metrics are non-blocking.
If dense retrieval slows us down: BM25 baseline ships first; dense becomes next-week ablation.
Harness risk: if blocked >1 day, add scripts/ fallback runner inside repo.
9. What We Will Report This Week
●
●
●
●
Retrieval setup (BM25; optionally dense/hybrid)
Recall@K results (K=5/10/20) on N=30 subset
3–5 failure mode observations
Next-week plan: reranking and/or scale N; align with agent team comparison
10. Open Questions (parked; not blocking Week 1)
●
●
●
Dataset choice finalisation (SWE-bench vs alternative) is a team-level decision
Whether and when to run % resolved/tests-pass (later, not Week 1 RAG deliverable)
Whether to include semantic entropy/perplexity analysis (secondary only)
Additional notes: using best agentic llm model that evaluated the above plan
⚠ Things to Watch / Tighten Up
1. The CLI invocation in Section 6 is slightly off
The plan says:
bash
python -m src.harness.runner --config config/rag_
baseline
_
week1.yaml
The actual module path (per pyproject.toml) is:
bash
python -m harness.runner --config config/rag_
baseline
_
week1.yaml
Or via the installed entry point:
bash
harness --config config/rag_
baseline
_
week1.yaml
This matters because src/ is the package root in pyproject.toml (where = ["src"]), so src.harness won't resolve.
Small thing, but will bite someone on Day 1.
2. Gold context format: paths need care
In benchmarks/base.py, BenchmarkInstance.gold
_
context is list[str] — the SWE-bench adapter extracts these from
the gold patch. Your Member 2 should verify early that these paths are repo-relative and normalised (no leading /,
no ./ prefix), because the retrieval metrics in metrics/retrieval.py do exact string matching between retrieved and
gold contexts. A mismatch like src/foo.py vs ./src/foo.py will silently tank Recall@K.
3. retrieval.py already has Recall@K — confirm it does what you expect
The scaffold's metrics/retrieval.py already implements recall
at
_
_
k, precision
at
_
_
k, mrr, and ndcg_
at
_
writing anything new, Member 1 should read through the existing implementation and confirm:
k. Before
It operates on file-path strings (it does)
Edge cases (empty gold set, k=0) are handled (they are, per the test suite)
This means Member 1's evaluation work is largely integration, not implementation.
4. Config naming: add rag_
baseline
_
week1.yaml explicitly
The scaffold has test
_
bm25.yaml already, but it uses limit: 5 and is a quick-test config. You'll want a dedicated
config/rag_
baseline
_
week1.yaml with limit: 30 and only BM25 enabled. Make sure this file is committed early —
it's your "golden config" and everyone should run against it.
5. Member 2's dataset artifact: clarify where it lives
The plan says "committed in the repo (location consistent with scaffold)" but doesn't specify a path. I'd suggest
either:
data/subset
_
30.json (new data/ dir)
Or just a parameter in the YAML config, since the SWE-bench adapter already supports limit: 30 and could accept a
task
_
ids: [...] filter
Decide and document this on Day 1 so Member 3 doesn't block.
6. Hybrid gatherer (rag_
hybrid.py) is not in default.yaml
You noted hybrid as an optional ablation, which is fine — but be aware it's already implemented in the scaffold
(rag_
hybrid.py, 3.4 KB) and simply fuses BM25 + Dense via RRF. It's almost free to add as an ablation once dense
works.
💡 Suggestions
Suggestion Rationale
Run pip install -e . on Day 1 and do harness --config config/test
_
bm25.yaml to see what breaks Surfaces harness
issues early (before the fallback decision point)
Add a --dry-run flag or smoke test that loads 1 instance without cloning repos SWE-bench repo cloning is slow;
you don't want Day 2 proof-of-life blocked by git clones
Pin the task subset by instance
_
id list, not just limit: 30 limit: 30 might give different subsets if dataset
ordering changes; a pinned list ensures reproducibility
Verdict
The plan is ready to execute. The scope is tight, the ownership split is clear, and the fallback policy is sensible. The
main risks are the path normalisation issue (silent metric bugs) and harness friction with SWE-bench repo cloning.
Both are manageable if surfaced early on Day 1.