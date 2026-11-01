import json, sys, re, time, urllib.request, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
T='/tmp/tmpl'
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx').drop(columns=['Unnamed: 0'],errors='ignore')
t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
ids=sorted(t.pdb.unique())
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id pdbx_initial_refinement_model{accession_code source_name type}
polymer_entities{entity_poly{pdbx_seq_one_letter_code_can type} polymer_entity_instances{rcsb_polymer_entity_instance_container_identifiers{auth_asym_id auth_to_entity_poly_seq_mapping}}}}}'''
def fetch(b):
    for k in range(6):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            return json.load(urllib.request.urlopen(r,timeout=300))['data']['entries']
        except Exception as e: print('retry',e,file=sys.stderr); time.sleep(2**k)
    return []
maps={}; seqs={}; irm={}
with ThreadPoolExecutor(4) as ex:
    for res in ex.map(fetch,[ids[i:i+40] for i in range(0,len(ids),40)]):
        for e in res:
            p=e['rcsb_id'].lower(); irm[p]=[(m['accession_code'] or '').lower() for m in (e['pdbx_initial_refinement_model'] or []) if m.get('accession_code')]
            for pe in e['polymer_entities'] or []:
                s=(pe['entity_poly']['pdbx_seq_one_letter_code_can'] or '').replace('\n','')
                for inst in pe['polymer_entity_instances'] or []:
                    c=inst['rcsb_polymer_entity_instance_container_identifiers']
                    maps[(p,c['auth_asym_id'])]=c['auth_to_entity_poly_seq_mapping']; seqs[(p,c['auth_asym_id'])]=s
json.dump(irm,open(f'{T}/irm_flagged.json','w'))
def to_seq(r):
    m=maps.get((r.pdb,r.chain))
    if not m: return pd.Series({'s0':np.nan,'s1':np.nan})
    idx=[i for i,lab in enumerate(m) if lab and re.match(r'-?\d+',lab) and r.rs<=int(re.match(r'-?\d+',lab).group())<=r.re]
    return pd.Series({'s0':min(idx) if idx else np.nan,'s1':max(idx) if idx else np.nan})
t[['s0','s1']]=t.apply(to_seq,axis=1)
t['seq']=[seqs.get(k,'') for k in zip(t.pdb,t.chain)]
print('mapped',t.s0.notna().mean())
t=t.dropna(subset=['s0'])
# families: same exact sequence, overlapping seq ranges
t=t.sort_values(['seq','s0']); fam=[]; fid=-1; last=(None,-1)
for r in t.itertuples():
    if r.seq!=last[0] or r.s0>last[1]: fid+=1; last=(r.seq,r.s1)
    else: last=(r.seq,max(last[1],r.s1))
    fam.append(fid)
t['family']=fam
t.to_pickle(f'{T}/errors_seqmapped.pkl')
g=t.groupby('family').agg(entries=('pdb','nunique'),rows=('pdb','size'))
print('families',len(g),' with >=2 flagged entries',(g.entries>=2).sum())
