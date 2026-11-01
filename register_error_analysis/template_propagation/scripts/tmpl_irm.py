import json, sys, time, urllib.request, pandas as pd
from concurrent.futures import ThreadPoolExecutor
S=pd.read_pickle('/tmp/tmpl/selected.pkl'); ids=sorted(S.pdb.unique())
Q='query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id pdbx_initial_refinement_model{accession_code source_name type details}}}'
def fetch(b):
    for k in range(6):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            return json.load(urllib.request.urlopen(r,timeout=300))['data']['entries']
        except Exception as e: time.sleep(2**k)
    return []
out={}
with ThreadPoolExecutor(4) as ex:
    for res in ex.map(fetch,[ids[i:i+200] for i in range(0,len(ids),200)]):
        for e in res:
            if e: out[e['rcsb_id'].lower()]=[{'acc':(m.get('accession_code') or '').lower(),'src':m.get('source_name'),'type':m.get('type'),'details':m.get('details')} for m in (e['pdbx_initial_refinement_model'] or [])]
json.dump(out,open('/tmp/tmpl/irm_selected.json','w')); print(len(out),'entries; with any IRM:',sum(1 for v in out.values() if v))
