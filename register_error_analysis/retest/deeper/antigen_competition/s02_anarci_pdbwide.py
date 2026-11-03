# ANARCI (IMGT) on all prefiltered PDB-wide candidate sequences; cache domain calls per unique sequence
import json, sys, os
from anarci import anarci
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/antigen_competition'
cand=json.load(open(f'{C}/cand_seqs.json'))
out=f'{C}/anarci_pdbwide.json'
res=json.load(open(out)) if os.path.exists(out) else {}
todo=[s for s in cand if s not in res]
B=1000
for i in range(0,len(todo),B):
    chunk=todo[i:i+B]
    num,det,_=anarci([(str(j),s) for j,s in enumerate(chunk)],scheme='imgt',ncpu=2,assign_germline=False,output=False)
    for s,d in zip(chunk,det):
        res[s]=[dict(chain_type=x['chain_type'],start=x['query_start'],end=x['query_end'],evalue=float(x['evalue'])) for x in d] if d else []
    json.dump(res,open(out,'w'))
    print(i+len(chunk),len(todo),flush=True)
print('done',len(res),sum(1 for v in res.values() if v))
