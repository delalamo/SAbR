"""Spatial proximity of each kappa error run to the Pro8 kink (IMGT 6-12, the A/A' strand switch) measured in
correctly built reference kappa chains (cis-Pro8, clean, EM controls from 05_omega). Tests the claim that the
association holds 'even when the error is far from FR1'."""
import json, numpy as np, pandas as pd
from Bio.PDB import MMCIFParser
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O=pd.read_csv('omega_check.csv'); ref=O[(~O.err_first)&(O.omega_class=='cis')].head(12)
K=pd.read_csv('kappa_frame.csv.gz'); K=K[~K.amyloid]
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
er=K[(K.method=='EM')&K.pro8&K.err_first]
R=runs.drop(columns=['paper']).merge(er[['pdb','chain','cis8','unit','paper']],on=['pdb','chain'])
keys=set(zip(ref.pdb,ref.chain)); amap={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k in keys and k not in amap:
            amap[k]={x[4]:(x[0],x[1] or ' ') for x in r['residues'] if x[4] is not None and not x[5]}
P=MMCIFParser(QUIET=True,auth_residues=True)
import os
def path(p):
    a=f'{SP}/sabr/dl/{p}.cif'; return a if os.path.exists(a) else f'{SP}/deeper/cache/verify_cis8/{p}.cif'
coords=[]
for _,r in ref.iterrows():
    st=P.get_structure(r.pdb,path(r.pdb))[0][r.chain]; m=amap[(r.pdb,r.chain)]; c={}
    for i,(n,ic) in m.items():
        try: c[i]=st[(' ',int(n),ic)]['CA'].coord
        except Exception: pass
    coords.append(c)
def dist(a0,a1):
    ds=[]
    for c in coords:
        A=[c[i] for i in range(6,13) if i in c]; B=[c[i] for i in range(int(a0),int(a1)+1) if i in c]
        if A and B: ds.append(min(np.linalg.norm(x-y) for x in A for y in B))
    return np.median(ds) if ds else np.nan
R['min_CA_dist_to_IMGT6_12']=[dist(a,b) for a,b in zip(R.imgt0,R.imgt1)]
# chain-level: nearest run of each chain
C=R.groupby(['pdb','chain']).agg(dmin=('min_CA_dist_to_IMGT6_12','min'),cis8=('cis8','first'),unit=('unit','first'),paper=('paper','first'),modes=('mode',lambda s:';'.join(sorted(set(s))))).reset_index()
C['near']=pd.cut(C.dmin,[-1,6,10,100],labels=['<=6 A (contact)','6-10 A','>10 A'])
print(R.groupby('mode').min_CA_dist_to_IMGT6_12.describe()[['count','min','50%','max']].round(1).to_string())
print(C.groupby('near').agg(chains=('cis8','size'),units=('unit','nunique'),papers=('paper','nunique'),cis8=('cis8','sum')).to_string())
U=C.groupby('unit').agg(dmin=('dmin','min'),cis8=('cis8','max'))
U['far']=U.dmin>10
print(U.groupby('far').cis8.agg(['sum','size','mean']))
C.to_csv('error_chain_proximity.csv',index=False); R.to_csv('error_run_proximity.csv',index=False)
