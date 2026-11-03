"""Pro115 checks: (a) position-based definition of 'error elsewhere' (no run overlapping IMGT 104-128) vs the
investigator's mode-regex filter; (b) omega at Pro115 from coordinates in the error chains; (c) 3D distance from each
error run to IMGT 115 measured in correctly built clean kappa chains (reference geometry)."""
import os, sys, json, gzip, math, urllib.request, collections, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_integrity')
from statlib import fit_feature
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_cisC_integrity'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_integrity'
D=pd.read_csv(f'{OUT}/frame_em.csv.gz'); D['c115']=D.cis115.astype(float)
R=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
modes=D.modes.fillna('')
D['mode_D_inv']=modes.str.contains('D|E_FR4')
ov=R[(R.imgt1>=104)].groupby(['pdb','chain']).size()
D['run_in_cdr3fr4']=[(p,c) in ov.index for p,c in zip(D.pdb,D.chain)]
K=D[(D.ctype=='K')&D.pro115].copy()
ZHOU='10.1038/s41422-021-00487-9'
res=[]
def rec(lbl,S):
    r=fit_feature(S,'c115','bin',['C(ybin)','res']); E=S[S.y==1]
    P=S.groupby(['cluster','y']).c115.mean().reset_index(); P['vb']=(P.c115>=0.5).astype(int)
    a=P[P.y==1].vb; b=P[P.y==0].vb; pf=stats.fisher_exact([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]])[1]
    d={'test':lbl,'err_val':r.get('err_val'),'ctrl_val':r.get('ctrl_val'),'OR':r.get('OR'),'p':r.get('p'),'n_err_papers':E.cluster.nunique(),'n_err_units':E.unit.nunique(),
       'paper_level':f'{a.sum()}/{len(a)} vs {b.mean()*100:.1f}%','paper_level_p':pf}
    res.append(d); print(d)
rec('investigator filter (regex D|E_FR4 on modes)',K[~(K.mode_D_inv&(K.y==1))])
rec('position filter: no run overlapping IMGT>=104',K[~(K.run_in_cdr3fr4&(K.y==1))])
rec('position filter, drop Zhou',K[~(K.run_in_cdr3fr4&(K.y==1))&(K.cluster!=ZHOU)])
rec('position filter, FR1-mode (A) chains only as errors',K[(K.y==0)|((K.y==1)&~K.run_in_cdr3fr4&modes.loc[K.index].str.contains('A_FR1'))])
rec('position filter, non-A chains only as errors',K[(K.y==0)|((K.y==1)&~K.run_in_cdr3fr4&~modes.loc[K.index].str.contains('A_FR1'))])
pd.DataFrame(res).to_csv(f'{OUT}/05_pro115_tests.csv',index=False)
# ---------- coordinates
from importlib import import_module
def get(p):
    for fn in [f'{SP}/sabr/dl/{p}.cif',f'{CACHE}/{p}.cif.gz']:
        if os.path.exists(fn): return fn
    fn=f'{CACHE}/{p}.cif.gz'
    try: urllib.request.urlretrieve(f'https://files.rcsb.org/download/{p}.cif.gz',fn); return fn
    except Exception as ex: print('fail',p,ex); return None
def atoms(fn,chains):
    op=gzip.open if fn.endswith('.gz') else open
    cols=[]; out=collections.defaultdict(dict); inloop=False
    with op(fn,'rt') as f:
        for line in f:
            if line.startswith('_atom_site.'): cols.append(line.strip().split('.',1)[1]); inloop=True; continue
            if inloop and (line.startswith('ATOM') or line.startswith('HETATM')):
                p=line.split()
                if len(p)!=len(cols): continue
                r=dict(zip(cols,p))
                if r.get('pdbx_PDB_model_num','1')!='1' or r['auth_asym_id'] not in chains or r['label_seq_id']=='.': continue
                if r['label_atom_id'] not in ('N','CA','C') or r.get('label_alt_id','.') not in ('.','A'): continue
                out[(r['auth_asym_id'],int(r['label_seq_id']))][r['label_atom_id']]=np.array([float(r['Cartn_x']),float(r['Cartn_y']),float(r['Cartn_z'])])
            elif inloop and line.startswith('#') and out: break
    return out
def dih(p0,p1,p2,p3):
    b0=p0-p1; b1=p2-p1; b2=p3-p2; b1=b1/np.linalg.norm(b1)
    v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
    return math.degrees(math.atan2(np.dot(np.cross(b1,v),w),np.dot(v,w)))
imgt={}
want=set(zip(K.pdb,K.chain))
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k in want and k not in imgt: imgt[k]={(x[4],x[5]):x[3] for x in (r.get('residues') or []) if x[3] is not None and x[4] is not None}
E=K[(K.y==1)&~K.run_in_cdr3fr4]
om=[]
for p,g in E.groupby('pdb'):
    fn=get(p); A=atoms(fn,set(g.chain))
    for c in g.chain:
        i=imgt[(p,c)].get((115,'')); 
        a=A.get((c,i+1)); b=A.get((c,i))
        w=dih(b['CA'],b['C'],a['N'],a['CA']) if a and b and 'C' in b and 'N' in a else None
        om.append({'pdb':p,'chain':c,'ann_cis115':bool(K[(K.pdb==p)&(K.chain==c)].cis115.iloc[0]),'omega114_115':w})
O=pd.DataFrame(om); O['omega_cis']=O.omega114_115.abs()<30
print('Pro115 omega vs annotation (error chains):'); print(pd.crosstab(O.ann_cis115,O.omega_cis))
O.to_csv(f'{OUT}/05_pro115_omega.csv',index=False)
# reference geometry: clean kappa chains (cis115 True, res<=3.5) with cached coordinates
ref=K[(K.y==0)&(K.cis115)&(K.res<=3.5)].sample(25,random_state=3)
refc=[]
for r in ref.itertuples():
    fn=get(r.pdb)
    if fn is None: continue
    A=atoms(fn,{r.chain}); m=imgt[(r.pdb,r.chain)]
    ca={pos:A[(r.chain,si+1)]['CA'] for pos,si in m.items() if (r.chain,si+1) in A and 'CA' in A[(r.chain,si+1)]}
    if (115,'') in ca: refc.append(ca)
print('reference kappa chains',len(refc))
RE=R.merge(E[['pdb','chain']],on=['pdb','chain'])
dist=[]
for r in RE.itertuples():
    ds=[]
    for ca in refc:
        pts=[v for (pos,ins),v in ca.items() if r.imgt0<=pos<=r.imgt1]
        if pts: ds.append(min(np.linalg.norm(v-ca[(115,'')]) for v in pts))
    dist.append({'pdb':r.pdb,'chain':r.chain,'mode':r.mode,'imgt0':r.imgt0,'imgt1':r.imgt1,'min_CA_dist_to_115_median':np.median(ds) if ds else np.nan})
DS=pd.DataFrame(dist); DS.to_csv(f'{OUT}/05_run_to_pro115_distance.csv',index=False)
print(DS.to_string())
