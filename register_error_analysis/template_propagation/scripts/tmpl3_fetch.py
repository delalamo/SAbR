import sys, time, pandas as pd, numpy as np
sys.path.insert(0,'/tmp/scripts'); from ca_fetch import get_ca
from concurrent.futures import ThreadPoolExecutor
T='/tmp/tmpl'
F=pd.read_pickle(f'{T}/families.pkl'); C=pd.read_pickle(f'{T}/candidates.pkl')
F=F[F.papers>=2]
rng=np.random.default_rng(0); sel=[]
for fid,g in C[C.family.isin(F.family)].groupby('family'):
    g=g.sort_values('date'); picked=[]; papers=set()
    def take(rows,n,distinct=True):
        k=0
        for r in rows.itertuples():
            if k>=n or len(picked)>=12: break
            if r.pdb in [p.pdb for p in picked]: continue
            if distinct and r.paper in papers: continue
            picked.append(r); papers.add(r.paper); k+=1
    fl=g[g.flagged]; take(fl.head(1),1,False); take(fl.iloc[1:],3)
    others=g[~g.pdb.isin([p.pdb for p in picked])]
    take(others,3); take(others.sort_values('res'),2); take(others.sort_values('date',ascending=False),2)
    take(others.sample(frac=1,random_state=int(fid)),12)
    for i,r in enumerate(picked): sel.append(dict(r._asdict(),is_ref=(i==0)))
S=pd.DataFrame(sel).drop(columns=['Index']); S.to_pickle(f'{T}/selected.pkl')
pairs=sorted(set(zip(S.pdb,S.chain))); print('families',S.family.nunique(),'selected rows',len(S),'unique chains',len(pairs),flush=True)
t0=time.time(); done=0
with ThreadPoolExecutor(32) as ex:
    for i,_ in enumerate(ex.map(lambda pc: get_ca(*pc),pairs)):
        if i%2000==0: print(i,len(pairs),'%.0fs'%(time.time()-t0),flush=True)
print('done %.0fs'%(time.time()-t0),flush=True)
