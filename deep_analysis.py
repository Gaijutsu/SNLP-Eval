import json, os, csv

base = 'results/2026-03-24_112424/instances'

# Read results.csv to get all per-instance metrics
with open('results/2026-03-24_112424/results.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Count wins/ties/losses for agentic_bm25 vs each other gatherer
wins_vs_rag = 0
losses_vs_rag = 0
ties_vs_rag = 0  
wins_vs_react = 0
losses_vs_react = 0
ties_vs_react = 0

instances = set(r['instance_id'] for r in rows)
for inst in sorted(instances):
    mrr = {}
    for r in rows:
        if r['instance_id'] == inst:
            mrr[r['gatherer']] = float(r['mrr'])
    
    a = mrr.get('agentic_bm25', 0)
    b = mrr.get('rag_bm25', 0)
    r = mrr.get('react_agent', 0)
    
    if a > b: wins_vs_rag += 1
    elif a < b: losses_vs_rag += 1
    else: ties_vs_rag += 1
    
    if a > r: wins_vs_react += 1
    elif a < r: losses_vs_react += 1
    else: ties_vs_react += 1

print("agentic_bm25 vs rag_bm25:")
print(f"  Wins: {wins_vs_rag}, Losses: {losses_vs_rag}, Ties: {ties_vs_rag}")
print(f"agentic_bm25 vs react_agent:")
print(f"  Wins: {wins_vs_react}, Losses: {losses_vs_react}, Ties: {ties_vs_react}")

# Also check: how often does agentic_bm25 use keyword_search first vs other tools?
print("\n--- Checking agentic_bm25 first-tool usage patterns ---")
keyword_first = 0
other_first = 0
total_keyword_calls = 0
total_steps = 0

for inst in sorted(instances):
    f = os.path.join(base, f'{inst}_agentic_bm25.json')
    with open(f) as fh:
        data = json.load(fh)
    conv = data.get('conversation', [])
    first_action_found = False
    for msg in conv:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if 'keyword_search' in content:
                total_keyword_calls += 1
                if not first_action_found:
                    keyword_first += 1
                    first_action_found = True
            elif 'Action:' in content and not first_action_found:
                other_first += 1
                first_action_found = True
        if msg.get('role') == 'assistant' and 'Action:' in msg.get('content',''):
            total_steps += 1

print(f"  keyword_search used as first action: {keyword_first}/{keyword_first + other_first}")
print(f"  Total keyword_search calls: {total_keyword_calls}")
print(f"  Total agent steps: {total_steps}")

# Check stagnation detection triggers
stagnation_count = 0
for inst in sorted(instances):
    f = os.path.join(base, f'{inst}_agentic_bm25.json')
    with open(f) as fh:
        data = json.load(fh)
    conv = data.get('conversation', [])
    for msg in conv:
        if msg.get('role') == 'user' and 'repeating actions' in msg.get('content', '').lower():
            stagnation_count += 1
            break

print(f"\n  Stagnation detected (forced finish): {stagnation_count}/{len(instances)}")

# Token usage comparison
print("\n--- Average token usage ---")
for g in ['rag_bm25', 'react_agent', 'agentic_bm25']:
    tokens = [float(r['token_usage']) for r in rows if r['gatherer'] == g]
    latency = [float(r['latency_s']) for r in rows if r['gatherer'] == g]
    print(f"  {g}: avg_tokens={sum(tokens)/len(tokens):.0f}, avg_latency={sum(latency)/len(latency):.1f}s")
