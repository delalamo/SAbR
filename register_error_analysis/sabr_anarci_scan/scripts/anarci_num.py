"""ANARCI IMGT numbering per sequence position, cached."""
import json, os, sys
from anarci import run_anarci
def number_seqs(seqs, cache_fn, ncpu=4):
    cache=json.load(open(cache_fn)) if os.path.exists(cache_fn) else {}
    todo=sorted({s for s in seqs if s and s not in cache})
    for k in range(0,len(todo),1000):
        chunk=[(str(i),s) for i,s in enumerate(todo[k:k+1000])]
        _,numbering,details,_=run_anarci(chunk,scheme='imgt',ncpu=ncpu,assign_germline=False)
        for (n,s),num,det in zip(chunk,numbering,details):
            doms=[]
            if num:
                for dom,info in zip(num,det):
                    nb,st,en=dom; idx=st; pos=[]
                    for (p,ins),aa in nb:
                        if aa!='-': pos.append([idx,p,ins.strip()]); idx+=1
                    doms.append({'chain_type':info['chain_type'],'start':st,'end':en,'pos':pos})
            cache[s]=doms
        json.dump(cache,open(cache_fn,'w'))
    return cache
