import json, sys, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
sys.path.insert(0,f'{SP}/scripts'); from sabr_eval import region
O=f'{SP}/sabr/out'
CH=pd.read_pickle(f'{O}/chains_flags.pkl')
recs={}
for f in ['validation.jsonl','snapshot_rest.jsonl','new.jsonl']:
    for l in open(f'{O}/{f}'):
        r=json.loads(l)
        if r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r['residues']
loc={}
for l in open(f'{SP}/sabr/mapfit/local.jsonl'):
    r=json.loads(l); f=r['feats']
    key='RSCC' if f.get('RSCC') else ('Q_SCORE' if f.get('Q_SCORE') else None)
    if key: loc[(r['pdb'],r['chain'])]={int(k):v for k,v in f[key].items()}
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx'); t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
conk=t.groupby(['pdb','chain']).apply(lambda g: set().union(*[set(range(a,b+1)) for a,b in zip(g.rs,g.re)])).to_dict()
def runs(res):
    out=[]; cur=[]
    for x in [x for x in res if x[4] is not None]+[None]:
        if x is not None and (x[4],x[5] or '')!=(x[6],x[7] or ''): cur.append(x)
        else:
            if len(cur)>=3: out.append(cur)
            cur=[]
    return out
def pct(vals,seg_idx,v_idx):
    s=[vals[i] for i in seg_idx if i in vals]; rest=[vals[i] for i in v_idx if i in vals and i not in seg_idx]
    if len(s)<3 or len(rest)<30: return np.nan
    return 100*np.mean(np.array(rest)<np.mean(s))
rows=[]
meta=CH.set_index(['pdb','chain'])
for (p,c),row in meta[meta.flag==1].iterrows():
    vals=loc.get((p,c)); res=recs.get((p,c))
    if not vals or not res: continue
    v_idx={x[3]+1 for x in res if x[4] is not None and x[3] is not None}
    for s in runs(res):
        si={x[3]+1 for x in s if x[3] is not None}
        d=[x[6]-x[4] for x in s if region(x[4])=='FR' and not (x[5] or x[7])]
        a0=s[0][4]
        kind='no framework shift measurable' if not d else ('shift in FR3 80–86 (DE loop)' if 80<=a0<=86 else 'framework shift')
        ck=conk.get((p,c),set()); ov=bool(ck & {x[0] for x in s})
        rows.append(dict(set='detected',pdb=p,chain=c,era=row.era,method=row.method,kind=kind,imgt0=a0,imgt1=s[-1][4],conkit_overlap=ov,pct=pct(vals,si,v_idx)))
D=pd.DataFrame(rows)
# controls: same IMGT window in undetected chains
ctrl=pd.read_csv(f'{SP}/sabr/mapfit/control_chains.csv'); rng=np.random.default_rng(5); crow=[]
wins=D[['imgt0','imgt1']].values
for r in ctrl.itertuples():
    vals=loc.get((r.pdb,r.chain)); res=recs.get((r.pdb,r.chain))
    if not vals or not res: continue
    a,b=wins[rng.integers(len(wins))]
    v_idx={x[3]+1 for x in res if x[4] is not None and x[3] is not None}
    si={x[3]+1 for x in res if x[4] is not None and a<=x[4]<=b and x[3] is not None}
    crow.append(dict(set='control (no disagreement)',pdb=r.pdb,chain=r.chain,era=r.era,method=r.method,kind='control',pct=pct(vals,si,v_idx)))
A=pd.concat([D,pd.DataFrame(crow)]).dropna(subset=['pct'])
pd.set_option('display.width',250)
print('== Fit of the segment to the map, as percentile within its own V domain (50 = typical; lower = worse)')
print(A.groupby(['era','method','set']).pct.agg(['size','median',lambda x:(x<25).mean()*100]).rename(columns={'<lambda_0>':'% below 25th'}).round(1).to_string())
print('\n== Detected runs by kind')
print(D.dropna(subset=['pct']).groupby(['era','kind']).pct.agg(['size','median']).round(1).to_string())
print('\n== Snapshot detections: overlapping a conkit flag vs not')
print(D[D.era.str.startswith('snap')].dropna(subset=['pct']).groupby('conkit_overlap').pct.agg(['size','median']).round(1).to_string())
from scipy.stats import mannwhitneyu
for era in A.era.unique():
    for m in ['EM','X-ray']:
        x=A[(A.era==era)&(A.method==m)&(A.set=='detected')&(A.kind=='framework shift')].pct; y=A[(A.era==era)&(A.method==m)&(A.set!='detected')].pct
        if len(x)>5 and len(y)>5: print(f'{era} {m}: framework-shift detections median {x.median():.0f} vs control {y.median():.0f}; Mann-Whitney p={mannwhitneyu(x,y).pvalue:.2g}')
A.to_csv(f'{O}/mapfit_detections.csv',index=False)
