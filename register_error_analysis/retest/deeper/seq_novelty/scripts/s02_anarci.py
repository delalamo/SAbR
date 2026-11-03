"""IMGT-number all prefiltered PDB candidate sequences with ANARCI (numbering only; germlines assigned later, vectorised).
Output: cache/seq_novelty/anarci_num.pkl : {seq: [ (chain_type, start, end, evalue, [(pos,ins,aa),...]) , ...] or []}"""
import json, pickle, os, sys, time
from anarci import run_anarci
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/seq_novelty'
cand=json.load(open(f'{C}/cand_seqs.json'))
out_f=f'{C}/anarci_num.pkl'
res=pickle.load(open(out_f,'rb')) if os.path.exists(out_f) else {}
todo=[s for s in cand if s not in res]
print('todo',len(todo),flush=True)
CH=2000; t0=time.time()
for i in range(0,len(todo),CH):
    chunk=todo[i:i+CH]
    _,num,ali,_=run_anarci([(str(j),s) for j,s in enumerate(chunk)],scheme='imgt',ncpu=int(sys.argv[1]) if len(sys.argv)>1 else 2,assign_germline=False)
    for s,n,a in zip(chunk,num,ali):
        doms=[]
        if n:
            for (numb,st,en),d in zip(n,a):
                doms.append((d['chain_type'],st,en,d.get('evalue'),[(p,ins,aa) for (p,ins),aa in numb if aa!='-']))
        res[s]=doms
    pickle.dump(res,open(out_f+'.tmp','wb')); os.replace(out_f+'.tmp',out_f)
    print(i+len(chunk), round(time.time()-t0), flush=True)
print('done')
