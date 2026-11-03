"""Empirical Kabat->IMGT position map (modal) for framework/canonical/Vernier/interface positions, per chain type."""
import pickle, json, collections, random
from anarci import run_anarci
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/seq_novelty'
db=pickle.load(open(f'{C}/abdb.pkl','rb')); D=db['D']
random.seed(1)
out={}
for ct in 'HKL':
    seqs=D[D.ctype==ct].vdom.sample(400,random_state=1).tolist()
    _,ni,_,_=run_anarci([(str(i),s) for i,s in enumerate(seqs)],scheme='imgt',ncpu=2)
    _,nk,_,_=run_anarci([(str(i),s) for i,s in enumerate(seqs)],scheme='kabat',ncpu=2)
    cnt=collections.defaultdict(collections.Counter)
    for a,b,s in zip(ni,nk,seqs):
        if not a or not b: continue
        ia={};ka={}
        for nb,st,en in a[:1]:
            j=st
            for (p,ins),aa in nb:
                if aa!='-': ia[j]=(p,ins.strip()); j+=1
        for nb,st,en in b[:1]:
            j=st
            for (p,ins),aa in nb:
                if aa!='-': ka[j]=(p,ins.strip()); j+=1
        for j,k in ka.items():
            if k[1]=='' and j in ia: cnt[k[0]][ia[j]]+=1
    out[ct]={k:[list(v.most_common(1)[0][0]),v.most_common(1)[0][1]/sum(v.values())] for k,v in cnt.items()}
json.dump(out,open(f'{C}/kabat2imgt.json','w'))
for ct in out:
    print(ct, {k:(v[0][0],round(v[1],2)) for k,v in sorted(out[ct].items()) if k in (2,4,24,25,26,27,28,29,30,33,34,35,36,37,38,39,44,45,46,47,48,49,51,54,55,64,66,67,68,69,71,73,78,87,89,90,91,93,94,95,96,97,98,103)})
