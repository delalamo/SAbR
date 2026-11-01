import json, sys, os, subprocess, numpy as np, pandas as pd
T='/tmp/tmpl'; T2='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/tmpl2'; O2=f'{T2}/out'
Rc=pd.read_pickle(f'{O2}/register_calls_chain.pkl'); R=pd.read_pickle(f'{O2}/register_calls.pkl')
F=pd.read_pickle(f'{T}/families.pkl').set_index('family')
st=dict(zip(zip(R.family,R.pdb),R.call))
Rc['entry_call']=[st.get(k) for k in zip(Rc.family,Rc.pdb)]
use=Rc[(Rc.call==Rc.entry_call)&Rc.call.isin(['same as flagged','other register'])]
have=set()
for l in open(f'{T}/local_called.jsonl'): have.add(json.loads(l)['pdb'])
need=sorted(set(use.pdb)-have)
open(f'{T2}/mapfit_need.txt','w').write('\n'.join(need)); print('entries needing map features:',len(need),flush=True)
if need: subprocess.run(['python3','-I','/tmp/scripts/gql_local.py',f'{T2}/mapfit_need.txt',f'{T2}/local_extra.jsonl','8'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
loc={}
for fn in [f'{T}/local_called.jsonl',f'{T2}/local_extra.jsonl']:
    if os.path.exists(fn):
        for l in open(fn):
            r=json.loads(l); loc[(r['pdb'],r['chain'])]=r['feats']
rows=[]
for r in use.itertuples():
    f=loc.get((r.pdb,r.chain))
    if not f: continue
    key='RSCC' if f.get('RSCC') else ('Q_SCORE' if f.get('Q_SCORE') else None)
    if not key: continue
    v={int(k):x for k,x in f[key].items()}
    fam=F.loc[r.family]; seg=[i+1+int(r.offset) for i in range(int(fam.s0),int(fam.s1)+1)]
    sv=[v[s] for s in seg if s in v]; rest=[x for s,x in v.items() if s not in seg]
    if len(sv)<3 or len(rest)<20: continue
    rows.append(dict(family=r.family,pdb=r.pdb,chain=r.chain,S=r.call=='same as flagged',res=r.res,method=r.method,metric=key,
                     seg_pct=(np.array(rest)<np.mean(sv)).mean()*100,seg_z=(np.mean(sv)-np.mean(rest))/(np.std(rest) or np.nan)))
Mc=pd.DataFrame(rows); Mc.to_pickle(f'{O2}/mapfit_chain.pkl')
M=Mc.groupby(['family','pdb']).agg(S=('S','first'),res=('res','first'),method=('method','first'),metric=('metric','first'),seg_pct=('seg_pct','mean'),seg_z=('seg_z','mean')).reset_index()
M.to_pickle(f'{O2}/mapfit.pkl'); print(len(M),'entries with map-fit values')
