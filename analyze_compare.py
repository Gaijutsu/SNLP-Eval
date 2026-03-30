#!/usr/bin/env python3
"""Compare retrieved files across methods for key instances."""
import json, os

base = "results/2026-03-24_112424/instances"
cases = ["astropy__astropy-14995", "astropy__astropy-6938", "astropy__astropy-14182", "django__django-10924"]

for case in cases:
    print(f"\n{'='*80}")
    print(f"INSTANCE: {case}")
    print(f"{'='*80}")
    for method in ["rag_bm25", "agentic_bm25", "react_agent"]:
        with open(os.path.join(base, f"{case}_{method}.json")) as f:
            data = json.load(f)
        gold = data.get("gold_contexts", [])
        retrieved = data.get("retrieved_contexts", [])
        mrr = data.get("metrics", {}).get("mrr", "?")
        recall10 = data.get("metrics", {}).get("recall@10", "?")
        print(f"\n  [{method}] MRR={mrr}, recall@10={recall10}")
        print(f"    gold:      {gold}")
        print(f"    retrieved: {retrieved}")

        # For react_agent, show trace briefly
        if method == "react_agent":
            trace = data.get("trace", [])
            for step in trace:
                action = step.get("action", "?")
                args = step.get("args", [])
                print(f"      Step {step.get('step')}: {action}({', '.join(str(a) for a in args[:3])})")
