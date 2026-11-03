"""Independent check of the wwPDB cis annotation: compute the omega dihedral CA(7)-C(7)-N(8)-CA(8) at kappa Pro8 from
coordinates for all EM error kappa chains and a random sample of EM clean kappa chains (same years)."""
import os, sys, json, gzip, urllib.request, numpy as np, pandas as pd, random
from Bio.PDB import MMCIFParser
from Bio.PDB.vectors import calc_dihedral
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_cis8'; os.makedirs(CACHE,exist_ok=True)
K=pd.read_csv('kappa_frame.csv.gz'); K=K[~K.amyloid]
EM=K[(K.method=='EM')&(K.pro8)&((K.err_first)|(~K.err_any))].copy()
err=EM[EM.err_first]
random.seed(1)
ctrl=EM[(~EM.err_first)&(EM.year>=2018)].sample(160,random_state=1)
S=pd.concat([err,ctrl])
# author numbers of IMGT 7 and 8
amap={}
keys=set(zip(S.pdb,S.chain))
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k not in keys or k in amap: continue
        d={}
        for x in r.get('residues') or []:
            if x[4] in (7,8) and not x[5]: d[x[4]]=(x[0],x[1] or ' ',x[2])
        amap[k]=d
def get(pdb):
    p=f'{SP}/sabr/dl/{pdb}.cif'
    if os.path.exists(p): return p
    q=f'{CACHE}/{pdb}.cif'
    if not os.path.exists(q):
        try:
            data=urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb}.cif.gz',timeout=120).read()
            open(q,'wb').write(gzip.decompress(data))
        except Exception as ex: print('dl fail',pdb,ex,file=sys.stderr); return None
    return q
P=MMCIFParser(QUIET=True,auth_residues=True)
rows=[]
for pdb,g in S.groupby('pdb'):
    f=get(pdb)
    if f is None: continue
    try: st=P.get_structure(pdb,f)[0]
    except Exception as ex: print('parse',pdb,ex,file=sys.stderr); continue
    for _,c in g.iterrows():
        d=amap.get((pdb,c.chain),{})
        om=np.nan
        if 7 in d and 8 in d and c.chain in st:
            chn=st[c.chain]
            try:
                r7=chn[(' ',int(d[7][0]),d[7][1])]; r8=chn[(' ',int(d[8][0]),d[8][1])]
                om=np.degrees(calc_dihedral(r7['CA'].get_vector(),r7['C'].get_vector(),r8['N'].get_vector(),r8['CA'].get_vector()))
            except Exception as ex: pass
        rows.append({'pdb':pdb,'chain':c.chain,'err_first':c.err_first,'cis8_annot':c.cis8,'omega8':om,'year':c.year,'modes':c.modes,'paper':c.paper})
O=pd.DataFrame(rows); O['cis8_coord']=O.omega8.abs()<30
O['omega_class']=np.select([O.omega8.abs()<30,O.omega8.abs()>150,O.omega8.notna()],['cis','trans','distorted'],'na')
O.to_csv('omega_check.csv',index=False)
print(pd.crosstab([O.err_first,O.cis8_annot],O.omega_class))
print(O[O.err_first].omega8.describe())
