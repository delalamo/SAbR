import json, sys, collections, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
sys.path.insert(0,f'{SP}/scripts'); from sabr_eval import region, segments
pd.set_option('display.width',250)
R=pd.read_pickle(f'{SP}/unexpl/runs.pkl'); M=pd.read_pickle(f'{SP}/retest/master2.pkl'); M=M[M.single]
recs={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r=json.loads(l)
        if r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r['residues']
loc={}
for l in open(f'{SP}/retest/local_all.jsonl'):
    r=json.loads(l); f=r['feats']; key='RSCC' if f.get('RSCC') else ('Q_SCORE' if f.get('Q_SCORE') else None)
    if key: loc[(r['pdb'],r['chain'])]={int(k):v for k,v in f[key].items()}
def classify(r):
    v=[x for x in recs[(r.pdb,r.chain)] if x[4] is not None]
    seg=v[r.i0:r.i1+1]
    a118=[x[2] for x in v if x[4]==118 and not x[5]]; s118=[x[2] for x in v if x[6]==118 and not x[7]]
    skip120 = any(x[4]==120 and x[6]==121 for x in seg) and a118==s118
    if r.imgt0>=105 or r.imgt1>=105:
        if skip120 and r.imgt0>=118: return 'FR4: SAbR skips IMGT 120 after W/F118-G119'
        if a118!=s118: return 'CDR3→FR4: conserved W/F118 displaced'
        return 'CDR3 internal (anchors agree)'
    if r.imgt0>=66: return 'FR3'
    if 56<=r.imgt0<=65 or 56<=r.imgt1<=65: return 'CDR2 / CDR2-FR3'
    if 27<=r.imgt0<=38 or 27<=r.imgt1<=38: return 'CDR1 / CDR1-FR2'
    return 'other'
U=R[R.cat=='unexplained'].copy(); U['pattern']=U.apply(classify,axis=1)
def pct_for(p,c,idxset):
    vals=loc.get((p,c)); res=recs.get((p,c))
    if not vals or not res: return np.nan
    v=[x for x in res if x[4] is not None and x[3] is not None]; vidx={x[3]+1 for x in v}
    sv=[vals[i] for i in idxset if i in vals]; rest=[vals[i] for i in vidx-idxset if i in vals]
    return 100*np.mean(np.array(rest)<np.mean(sv)) if len(sv)>=3 and len(rest)>=30 else np.nan
U['pct']=[pct_for(r.pdb,r.chain,{x[3]+1 for x in [x for x in recs[(r.pdb,r.chain)] if x[4] is not None][r.i0:r.i1+1] if x[3] is not None}) for r in U.itertuples()]
clean=M[M.flag==0]; rng=np.random.default_rng(4); ctrl=[]
for r in U.itertuples():
    pool=clean[(clean.ctype==r.ctype)&(clean.method==r.method)]
    for q in pool.sample(min(10,len(pool)),random_state=int(rng.integers(1e9))).itertuples():
        res=recs.get((q.pdb,q.chain))
        if not res: continue
        idx={x[3]+1 for x in res if x[4] is not None and x[3] is not None and r.imgt0<=x[4]<=r.imgt1}
        ctrl.append(dict(pattern=r.pattern,method=r.method,pct=pct_for(q.pdb,q.chain,idx)))
C=pd.DataFrame(ctrl)
out=U.groupby('pattern').agg(runs=('pdb','size'),antibodies=('seqkey','nunique'),papers=('paper','nunique'),EM=('method',lambda x:round(100*(x=='EM').mean())),
    median_res=('res','median'),map_pct=('pct','median')).join(C.groupby('pattern').pct.median().rename('control_pct'))
print(out.sort_values('runs',ascending=False).round(1).to_string())
print('\nby method:'); print(U.groupby(['pattern','method']).pct.median().unstack().round(1).join(C.groupby(['pattern','method']).pct.median().unstack().round(1),rsuffix=' ctrl').to_string())
# does the 'skip 120' pattern ever occur in high-resolution templates, or in the clean set at all?
tp=[json.loads(l) for l in open(f'{SP}/retest/templates.jsonl')]
n120=0; ntp=0
for r in tp:
    if r.get('status')!='ok': continue
    ntp+=1; n120+=any(x[4]==120 and x[6]==121 for x in r['residues'] if x[4] is not None)
print(f'\nhigh-res (<3 Å) templates where SAbR numbers ANARCI-120 as 121: {n120}/{ntp}')
allc=[(k,any(x[4]==120 and x[6]==121 for x in v if x[4] is not None)) for k,v in recs.items()]
print('all scanned 3–5 Å chains with that pattern: %d/%d'%(sum(b for _,b in allc),len(allc)))
U.to_pickle(f'{SP}/unexpl/unexplained_patterns.pkl')
