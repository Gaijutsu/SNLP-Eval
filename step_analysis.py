import json, os

base = 'results/2026-03-24_112424/instances'
instances = set()
for f in os.listdir(base):
    if f.endswith('.json'):
        inst_id = '_'.join(f.split('_')[:-1])
        # Actually need to be smarter about splitting since instance_id contains underscores
        pass

# Just read all agentic_bm25 files
for f in sorted(os.listdir(base)):
    if not f.endswith('_agentic_bm25.json'):
        continue
    inst_id = f.replace('_agentic_bm25.json', '')
    
    with open(os.path.join(base, f)) as fh:
        ab = json.load(fh)
    
    react_f = f.replace('_agentic_bm25.json', '_react_agent.json')
    with open(os.path.join(base, react_f)) as fh:
        ra = json.load(fh)
    
    ab_steps = len(ab.get('trace', []))
    ra_steps = len(ra.get('trace', []))
    ab_mrr = ab['metrics']['mrr']
    ra_mrr = ra['metrics']['mrr']
    
    # Check if keyword_search results matched gold
    ab_conv = ab.get('conversation', [])
    ks_query = ""
    ks_results = []
    for msg in ab_conv:
        c = msg.get('content', '')
        if 'keyword_search(' in c and msg.get('role') == 'assistant':
            # Extract query
            import re
            m = re.search(r'keyword_search\("([^"]*)"', c)
            if m:
                ks_query = m.group(1)
        if 'BM25 search results:' in c:
            # Extract file paths from results
            for line in c.split('\n'):
                m = re.search(r'\d+\.\s+(\S+)\s+\(score:', line)
                if m:
                    ks_results.append(m.group(1))
    
    marker = "!!" if ab_mrr < ra_mrr else "  "
    print(f"{marker} {inst_id}")
    print(f"   agentic_bm25: MRR={ab_mrr:.3f}, steps={ab_steps}")
    print(f"   react_agent:  MRR={ra_mrr:.3f}, steps={ra_steps}")
    if ks_query:
        print(f"   keyword_search query: \"{ks_query}\"")
    if ks_results:
        print(f"   keyword_search top3: {ks_results[:3]}")
    print()
