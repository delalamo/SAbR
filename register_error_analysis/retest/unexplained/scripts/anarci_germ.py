import json, sys, pandas as pd
from anarci import run_anarci
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
M=pd.read_pickle(f'{SP}/retest/master2.pkl'); M=M[M.single]
seqs=sorted(set(M.seq)); out={}
for k in range(0,len(seqs),1000):
    chunk=[(str(i),s) for i,s in enumerate(seqs[k:k+1000])]
    _,_,det,_=run_anarci(chunk,scheme='imgt',ncpu=4,assign_germline=True)
    for (n,s),d in zip(chunk,det):
        if d:
            g=d[0].get('germlines',{})
            v=g.get('v_gene',[[None,None],None]); j=g.get('j_gene',[[None,None],None])
            out[s]=dict(species=v[0][0] if v and v[0] else None,v_gene=v[0][1] if v and v[0] else None,v_id=v[1] if v else None,j_gene=j[0][1] if j and j[0] else None)
    print(k,len(seqs),flush=True)
json.dump(out,open(f'{SP}/unexpl/germline.json','w')); print('done')
