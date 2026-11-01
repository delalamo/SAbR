import sys, pandas as pd, numpy as np
sys.path.insert(0,'/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/scripts'); from ca2 import get_ca2
from concurrent.futures import ThreadPoolExecutor
T='/tmp/tmpl'; T2='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/tmpl2'
S=pd.read_pickle(f'{T}/selected.pkl')[['family','pdb','is_ref','flagged','date','res','method','last_author','paper']]
CA=pd.read_pickle(f'{T2}/candidates_allchains.pkl')
t=pd.read_pickle(f'{T}/errors_seqmapped.pkl')
F=pd.read_pickle(f'{T}/families.pkl').set_index('family')
fl=t.groupby(['family','pdb']).chain.apply(lambda x: sorted(set(x))).to_dict()
famseq=t.groupby('family').seq.first().to_dict()
chains=CA.groupby(['family','pdb']).apply(lambda g: list(zip(g.chain,g.offset))).to_dict()
rows=[]; missing_ref=0
for r in S.itertuples():
    avail=dict(chains.get((r.family,r.pdb),[]))
    flc=fl.get((r.family,r.pdb),[]) if r.flagged else []
    # flagged chains must be included even if absent from pdb_seqres matching
    for c in flc:
        if c not in avail:
            s=t[(t.family==r.family)&(t.pdb==r.pdb)&(t.chain==c)].seq.iloc[0]
            f=F.loc[r.family]; win=f.win; a=famseq[r.family].find(win); b=s.find(win)
            if a>=0 and b>=0: avail[c]=b-a
    order=[c for c in flc if c in avail]+[c for c in avail if c not in flc]
    order=order[:max(4,len([c for c in flc if c in avail]))]
    if r.is_ref and not [c for c in flc if c in avail]: missing_ref+=1
    for i,c in enumerate(order):
        rows.append(dict(family=r.family,pdb=r.pdb,chain=c,offset=int(avail[c]),chain_flagged=c in flc,
            is_ref_entry=r.is_ref,is_ref=bool(r.is_ref and c in flc and i==0),flagged=r.flagged,date=r.date,res=r.res,method=r.method,last_author=r.last_author,paper=r.paper))
X=pd.DataFrame(rows)
print('chain rows',len(X),'entries',X.groupby(['family','pdb']).ngroups,'families with a flagged-chain reference',X[X.is_ref].family.nunique(),'of',S.family.nunique(),'| ref entries lacking flagged chain:',missing_ref)
X.to_pickle(f'{T2}/selected_chains.pkl')
pairs=sorted(set(zip(X.pdb,X.chain)))
with ThreadPoolExecutor(16) as ex: res=list(ex.map(lambda pc: get_ca2(*pc),pairs))
print('chains',len(pairs),'fetched/cached ok',sum(r is not None for r in res),'failed',sum(r is None for r in res))
