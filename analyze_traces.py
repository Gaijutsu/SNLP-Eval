#!/usr/bin/env python3
"""Quick analysis of underperforming agentic_bm25 instances."""
import json
import os

base = "results/2026-03-24_112424/instances"
cases = ["astropy__astropy-14995", "astropy__astropy-6938", "astropy__astropy-14182", "django__django-10924"]

for case in cases:
    print(f"\n{'='*80}")
    print(f"INSTANCE: {case}")
    print(f"{'='*80}")

    for method in ["rag_bm25", "agentic_bm25", "react_agent"]:
        fpath = os.path.join(base, f"{case}_{method}.json")
        with open(fpath) as f:
            data = json.load(f)
        retrieved = data.get("retrieved_contexts", [])
        gold = data.get("gold_contexts", [])
        mrr = data.get("metrics", {}).get("mrr", "?")
        print(f"\n  [{method}] MRR={mrr}")
        print(f"    gold:      {gold}")
        print(f"    retrieved: {retrieved[:10]}")

        if method == "agentic_bm25":
            trace = data.get("trace", [])
            print(f"    --- Trace ({len(trace)} steps) ---")
            for step in trace:
                action = step.get("action", "?")
                args = step.get("args", [])
                obs = step.get("observation", "")
                obs_preview = obs[:200].replace("\n", " | ") if obs else "(none)"
                thought = step.get("thought", "")[:150].replace("\n", " | ")
                print(f"      Step {step.get('step')}: {action}({', '.join(str(a) for a in args)})")
                print(f"        Thought: {thought}")
                print(f"        Obs: {obs_preview}")
