"""Map-based check independent of AlphaFold: per-residue RSCC (X-ray) / Q-score (EM) of the flagged
segment relative to the rest of the same chain, for every entry with a register call."""
import json, sys, numpy as np, pandas as pd, subprocess
T='/tmp/tmpl'; O=f'{T}/out'
R=pd.read_pickle(f'{O}/register_calls.pkl'); F=pd.read_pickle(f'{T}/families.pkl').set_index('family')
C=pd.read_pickle(f'{T}/candidates.pkl')[['family','pdb','offset']]
R=R.merge(C,on=['family','pdb'],how='left')
R=R[R.called]
ids=sorted(R.pdb.unique()); open(f'{T}/mapfit_ids.txt','w').write('\n'.join(ids))
subprocess.run(['python3','-I','/tmp/scripts/gql_local.py',f'{T}/mapfit_ids.txt',f'{T}/local_called.jsonl','8'],check=True)
loc={}
for l in open(f'{T}/local_called.jsonl'):
    r=json.loads(l); loc[(r['pdb'],r['chain'])]=r['feats']
rows=[]
for r in R.itertuples():
    f=loc.get((r.pdb,r.chain))
    if not f: continue
    key='RSCC' if 'RSCC' in f and f['RSCC'] else ('Q_SCORE' if 'Q_SCORE' in f and f['Q_SCORE'] else None)
    if not key: continue
    v={int(k):x for k,x in f[key].items()}
    fam=F.loc[r.family]; seg=[i+1+int(r.offset) for i in range(int(fam.s0),int(fam.s1)+1)]
    sv=[v[s] for s in seg if s in v]
    rest=[x for s,x in v.items() if s not in seg]
    if len(sv)<3 or len(rest)<20: continue
    rows.append(dict(family=r.family,pdb=r.pdb,S=r.S,res=r.res,method=r.method,metric=key,seg=np.mean(sv),chain_med=np.median(rest),
                     seg_z=(np.mean(sv)-np.mean(rest))/np.std(rest),seg_pct=(np.array(rest)<np.mean(sv)).mean()*100))
M=pd.DataFrame(rows); M.to_pickle(f'{O}/mapfit.pkl'); print(len(M),'entries with map-fit values')
