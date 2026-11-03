"""Coordinate check of the cis annotation: recompute omega for every peptide of selected antibody chains
(all EM error chains with a modelled constant domain, their entry partners, and a random control sample),
and compare with the RCSB CIS-PEPTIDE feature at canonical constant-domain sites and over the whole constant domain."""
import os, gzip, math, json, urllib.request, sys, collections, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_cisC_integrity'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_integrity'
D=pd.read_csv(f'{OUT}/frame_em.csv.gz')
err=D[(D.y==1)&D.const_ok]
ents=set(err.pdb)
rng=np.random.default_rng(7)
ctl=D[(D.y==0)&D.const_ok&(~D.pdb.isin(ents))]
pick=pd.concat([ctl[ctl.ccis==1].sample(40,random_state=1),ctl[ctl.ccis==0].sample(40,random_state=2)])
ents|=set(pick.pdb)
sel=set(zip(err.pdb,err.chain))|set(zip(pick.pdb,pick.chain))|set(zip(D[D.pdb.isin(set(err.pdb))&D.const_ok].pdb,D[D.pdb.isin(set(err.pdb))&D.const_ok].chain))
def get(p):
    for fn in [f'{SP}/sabr/dl/{p}.cif',f'{CACHE}/{p}.cif.gz']:
        if os.path.exists(fn): return fn
    fn=f'{CACHE}/{p}.cif.gz'
    for a in range(3):
        try:
            urllib.request.urlretrieve(f'https://files.rcsb.org/download/{p}.cif.gz',fn); return fn
        except Exception as ex: print('retry',p,ex,file=sys.stderr)
    return None
def atoms(fn,chains):
    op=gzip.open if fn.endswith('.gz') else open
    cols=[]; res=collections.defaultdict(dict); inloop=False
    with op(fn,'rt') as f:
        for line in f:
            if line.startswith('_atom_site.'): cols.append(line.strip().split('.',1)[1]); inloop=True; continue
            if inloop and cols and (line.startswith('ATOM') or line.startswith('HETATM')):
                p=line.split()
                if len(p)!=len(cols): continue
                r=dict(zip(cols,p))
                if r.get('pdbx_PDB_model_num','1')!='1': continue
                a=r['auth_asym_id']
                if a not in chains: continue
                if r['label_atom_id'] not in ('N','CA','C'): continue
                if r.get('label_alt_id','.') not in ('.','A'): continue
                if r['label_seq_id']=='.': continue
                res[(a,int(r['label_seq_id']))][r['label_atom_id']]=np.array([float(r['Cartn_x']),float(r['Cartn_y']),float(r['Cartn_z'])])
            elif inloop and cols and line.startswith('#'):
                if res: break
    return res
def dih(p0,p1,p2,p3):
    b0=p0-p1; b1=p2-p1; b2=p3-p2; b1/=np.linalg.norm(b1)
    v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
    return math.degrees(math.atan2(np.dot(np.cross(b1,v),w),np.dot(v,w)))
rows=[]
for p in sorted(ents):
    fn=get(p)
    if fn is None: continue
    chs={c for (q,c) in sel if q==p}
    A=atoms(fn,chs)
    for c in chs:
        r=D[(D.pdb==p)&(D.chain==c)].iloc[0]
        ve=int(r.ve); sites=[int(s) for s in str(r.site_idx).split(';') if s not in ('nan','')]
        om={}
        for (a,sid),at in A.items():
            if a!=c: continue
            prev=A.get((a,sid-1))
            if prev and all(k in prev for k in ('CA','C')) and all(k in at for k in ('N','CA')):
                om[sid-1]=dih(prev['CA'],prev['C'],at['N'],at['CA'])   # key = 0-based index of residue i (second residue)
        ccis=[i for i,w in om.items() if i>ve+3 and abs(w)<30]
        site_w=[om.get(s) for s in sites]
        rows.append(dict(pdb=p,chain=c,y=r.y,ctype=r.ctype,paper=r.paper,ann_const_cis=r.n_cis_const,omega_const_cis=len(ccis),
                         ann_chain_cis=r.n_cis_chain,omega_chain_cis=sum(abs(w)<30 for w in om.values()),
                         n_omega=len(om),site_omega=';'.join('nan' if w is None else f'{w:.0f}' for w in site_w),
                         ann_site_cis=r.site_cis_n,omega_site_cis=sum(w is not None and abs(w)<30 for w in site_w),
                         nonplanar_const=sum(1 for i,w in om.items() if i>ve+3 and 30<=abs(w)<150)))
O=pd.DataFrame(rows); O.to_csv(f'{OUT}/03_omega.csv',index=False)
O['grp']=np.where(O.y==1,'error',np.where(O.pdb.isin(set(err.pdb)),'partner','control_sample'))
print(O.groupby('grp').apply(lambda g: pd.Series({'n':len(g),'agree_const_count':(g.ann_const_cis==g.omega_const_cis).mean(),
   'agree_chain_count':(g.ann_chain_cis==g.omega_chain_cis).mean(),'agree_site':(g.ann_site_cis==g.omega_site_cis).mean(),
   'omega_const_cis_any':(g.omega_const_cis>0).mean(),'nonplanar_const_mean':g.nonplanar_const.mean()})).to_string())
print(O[(O.ann_const_cis!=O.omega_const_cis)|(O.ann_site_cis!=O.omega_site_cis)].to_string())
