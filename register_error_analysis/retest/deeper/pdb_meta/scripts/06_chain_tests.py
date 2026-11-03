"""Chain-level tests within cryo-EM: V chains with a first-occurrence error vs V chains with no run at all.
(a) across entries: logistic, adjusted for deposition-year bin, resolution, chain type; cluster by paper.
(b) within error entries: error chains vs clean V chains of the same entry (paired Wilcoxon)."""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature, bh, ybin
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
ch=pd.read_csv(f'{SP}/chains.csv.gz'); C=pd.read_csv(f'{OUT}/chain_features_rcsb.csv.gz')
F=pd.read_csv(f'{OUT}/entry_features_rcsb.csv.gz',usecols=['pdb','vem_Q_score','non_q','non_inc','ab_q'])
d=ch.merge(C,on=['pdb','chain'],how='left').merge(F,on='pdb',how='left')
d=d[d.method=='EM'].copy()
d=d[~d.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)]
d=d[(d.err_first)|(~d.err_any)].copy(); d['y']=d.err_first.astype(int)
d['cluster']=d.paper.fillna(d.pdb); d['ybin']=ybin(pd.to_datetime(d.deposit).dt.year.values)
d['chain_q_rel_entry']=d.chain_q-d.vem_Q_score; d['chain_q_rel_nonab']=d.chain_q-d.non_q
d['chain_inc_rel_nonab']=d.chain_inc-d.non_inc
d['frac_modeled']=d.chain_modeled/d.chain_len
d['fv_only']=(d.chain_modeled<135).astype(int)
d['angle_outl_per_100']=100*d.chain_angle_outl/d.chain_modeled
d['cis_any']=(d.chain_cis>0).astype(int)
d['clash_per_res']=d.chain_clash_per_res
FEATS=[('chain_q','cont','map fit','partly OVERLAPS error'),('chain_q_rel_entry','cont','map fit','vs entry mean'),('chain_q_rel_nonab','cont','map fit','vs non-antibody chains'),
 ('chain_inc','cont','map fit',''),('chain_inc_rel_nonab','cont','map fit',''),
 ('chain_unobs_cov','cont','modelled fraction','OVERLAPS gap mechanism'),('frac_modeled','cont','modelled fraction','OVERLAPS'),('fv_only','bin','modelled fraction','Fv-only model'),
 ('clash_per_res','cont','validation','may OVERLAP'),('angle_outl_per_100','cont','validation',''),('chain_rmsz_ang','cont','validation',''),
 ('chain_sheet_cov','cont','secondary structure','DSSP-like sheet assignment'),('chain_unassigned_ss_cov','cont','secondary structure',''),
 ('cis_any','bin','modelling','any cis peptide in chain'),]
res=[]
for ct in ['all','H','K','L']:
    dd=d if ct=='all' else d[d.ctype==ct]
    cov=['C(ybin)','res']+(['C(ctype)'] if ct=='all' else [])
    for f,k,g,n in FEATS:
        x=dd.copy()
        if k=='bin': x[f]=x[f].astype(float)
        r=fit_feature(x,f,k,cov); r.update(group=g,note2=n,ctype=ct); res.append(r)
R=bh(pd.DataFrame(res))
R=R[['ctype','group','feature','note2','n_err','n_ctrl','err_val','ctrl_val','unit','OR','lo','hi','p','q','p_unadj']]
R.to_csv(f'{OUT}/chain_tests_em.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',40)
print(R[R.ctype=='all'].to_string(index=False,float_format=lambda x:f'{x:.3g}'))
print(R[(R.ctype!='all')&(R.q<0.1)].to_string(index=False,float_format=lambda x:f'{x:.3g}'))
# within-entry paired comparison (error entries with both error and clean V chains of the same chain type)
rows=[]
for (pdb,ct),g in d.groupby(['pdb','ctype']):
    if g.y.sum()==0 or (g.y==0).sum()==0: continue
    r={'pdb':pdb,'ctype':ct}
    for f in ['chain_q','chain_inc','chain_unobs_cov','clash_per_res','chain_sheet_cov','chain_cis','frac_modeled']:
        r[f+'_err']=g[g.y==1][f].mean(); r[f+'_ok']=g[g.y==0][f].mean()
    rows.append(r)
P=pd.DataFrame(rows); P.to_csv(f'{OUT}/within_entry_pairs.csv',index=False)
print('within-entry pairs (same entry, same chain type):',len(P))
for f in ['chain_q','chain_inc','chain_unobs_cov','clash_per_res','chain_sheet_cov','chain_cis','frac_modeled']:
    a=P[f+'_err']; b=P[f+'_ok']; ok=a.notna()&b.notna()
    if ok.sum()>5: print(f, f'err median {a[ok].median():.3f} vs clean {b[ok].median():.3f}; err<clean in {(a[ok]<b[ok]).mean()*100:.0f}% ; n={ok.sum()}; Wilcoxon p={stats.wilcoxon(a[ok],b[ok]).pvalue:.3g}')
