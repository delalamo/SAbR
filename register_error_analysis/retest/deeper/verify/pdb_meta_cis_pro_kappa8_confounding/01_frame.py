"""Build the kappa Pro8 analysis frame (EM + X-ray), reproducing pdb_meta 08/09 definitions, plus extra covariates:
local Q/RSCC at Pro8 and residues IMGT 1-15, N-terminal completeness (IMGT 7 built), modes, lab, unit, sars2, species."""
import json, sys, collections, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
PM='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_confounding'
D=pd.read_csv(f'{PM}/chain_cis_sheet.csv.gz')
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz')
cols=['pdb','chain','year','last_author','unit','species','v_gene','v_id','sars2','role','aux_reagent','irm','fit','vhh','lq','lq_z','err_copied_or_artifact_only','title','sw_ISOLDE','sw_Rosetta','sw_Coot','sw_Phenix','sw_ModelAngelo','sw_Namdinator','sw_REFMAC','sw_BUSTER']
D=D.merge(ch[cols],on=['pdb','chain'],how='left')
D=D.merge(pd.read_csv(f'{PM}/chain_novel_v.csv.gz')[['pdb','chain','novel_v']],on=['pdb','chain'],how='left')
D=D.merge(pd.read_csv(f'{PM}/chain_features_rcsb.csv.gz',usecols=['pdb','chain','chain_q','chain_unobs_cov','chain_cis']),on=['pdb','chain'],how='left')
K=D[D.ctype=='K'].copy()
keys=set(zip(K.pdb,K.chain))
# per-residue map: seq_index -> (imgt, ins, resname, author_num, icode)
imgt={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k not in keys or k in imgt: continue
        imgt[k]=[(x[3],x[4],x[5],x[2],x[0],x[1]) for x in r.get('residues') or [] if x[3] is not None]
# local fit: Q (EM) or RSCC (X-ray) keyed by author number
loc={}
for l in open(f'{SP}/retest/local_all.jsonl'):
    r=json.loads(l); k=(r['pdb'],r['chain'])
    if k not in keys: continue
    am=r['auth_map']; f=r.get('feats') or {}
    fk=[x for x in f if x in ('Q_SCORE','RSCC')]
    if not fk: continue
    vals=f[fk[0]]
    loc[k]={str(am[int(kk)-1]):v for kk,v in vals.items() if int(kk)-1<len(am)}
rows=[]
for _,c in K.iterrows():
    k=(c.pdb,c.chain); res=imgt.get(k,[]); L=loc.get(k)
    byimgt={(a,ins or ''):(si,rn,an,ic) for si,a,ins,rn,an,ic in res if a is not None}
    p8=byimgt.get((8,'')); p7=byimgt.get((7,''))
    q8=q_n=np.nan
    if L is not None:
        if p8: q8=L.get(str(p8[2]),np.nan)
        qs=[L.get(str(v[2])) for (a,i),v in byimgt.items() if 1<=a<=15]
        qs=[x for x in qs if x is not None]
        q_n=np.mean(qs) if qs else np.nan
        qv=[L.get(str(v[2])) for (a,i),v in byimgt.items() if 1<=a<=128]
        qv=[x for x in qv if x is not None]
        qV=np.mean(qv) if qv else np.nan
    else: qV=np.nan
    # first built IMGT position
    first=min([a for (a,i) in byimgt],default=np.nan)
    rows.append({'pdb':c.pdb,'chain':c.chain,'q_pro8':q8,'q_nterm':q_n,'q_V':qV,'imgt7_built':p7 is not None,
                 'first_imgt':first,'res8':(p8[1] if p8 else None)})
X=pd.DataFrame(rows)
K=K.merge(X,on=['pdb','chain'],how='left')
pro=K.v_pro_pos.fillna('').str.split(';'); cisp=K.v_cis_pos.fillna('').str.split(';')
K['pro8']=[('8' in a) for a in pro]; K['cis8']=[('8PRO' in a) for a in cisp]
K['pro115']=[('115' in a) for a in pro]; K['cis115']=[('115PRO' in a) for a in cisp]
K['v_cis_other']=[sum(1 for p in a if p and p!='8PRO') for a in cisp]
K['amyloid']=K.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)
K.to_csv(f'{OUT}/kappa_frame.csv.gz',index=False)
print(K.shape, K.pro8.sum(), K.q_pro8.notna().sum())
