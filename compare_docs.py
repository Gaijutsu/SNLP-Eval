import json, os

base = 'results/2026-03-24_112424/instances'
cases = ['django__django-10924', 'astropy__astropy-14182', 'astropy__astropy-14995', 'astropy__astropy-6938']

for case in cases:
    print(f"=== {case} ===")
    for gatherer in ['rag_bm25', 'react_agent', 'agentic_bm25']:
        f = os.path.join(base, f'{case}_{gatherer}.json')
        with open(f) as fh:
            data = json.load(fh)
        docs = data.get('retrieved_documents', [])[:5]
        mrr = data['metrics']['mrr']
        print(f'  {gatherer}: MRR={mrr:.3f}, top5={docs}')
    print()
