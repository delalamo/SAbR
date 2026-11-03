"""Reproduce headline numbers from the independent frame; compare to the investigator's chain table chain by chain."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature, ybin
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_integrity'
T='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
D=pd.read_csv(f'{OUT}/frame.csv.gz')
D=D[(D.method=='EM')&(~D.amyloid)&((D.err_first)|(~D.err_any))].copy()
D['y']=D.err_first.astype(int); D['cluster']=D.paper.fillna(D.pdb); D['ybin']=ybin(pd.to_datetime(D.deposit).dt.year.values)
D['const_ok']=(D.n_const_mod>60)&(D.const_pro_mod>0)
D['ccis']=(D.n_cis_const>0).astype(float).where(D.const_ok)
I=pd.read_csv(f'{T}/chain_cis_sheet.csv.gz')
I['i_ok']=(I.n_const_modeled>60)&(I.const_pro_n>0); I['i_ccis']=(I.const_cis_n>0).astype(float).where(I.i_ok)
M=D.merge(I[['pdb','chain','i_ok','i_ccis']],on=['pdb','chain'],how='left')
print('const_ok agreement',pd.crosstab(M.const_ok,M.i_ok)); print('ccis agreement among both ok',pd.crosstab(M.ccis,M.i_ccis))
modes=D.modes.fillna('')
D['mode_D_inv']=modes.str.contains('D|E_FR4'); D['mode_D_true']=modes.str.contains('Dplus|Dminus|E_FR4')
rows=[]
def t(sub,feat,label,cov=['C(ybin)','res']):
    r=fit_feature(sub.assign(**{feat:sub[feat].astype(float)}),feat,'bin',cov); r['test']=label; rows.append(r)
    print(label,{k:r.get(k) for k in ['err_val','ctrl_val','OR','p','note']})
for ct in 'KH': t(D[D.ctype==ct],'ccis',f'{ct} any const cis (investigator def, rebuilt)')
t(D,'ccis','pooled any const cis',['C(ybin)','res','C(ctype)'])
K=D[(D.ctype=='K')&D.pro115]
K=K.assign(c115=K.cis115.astype(float))
t(K[~(K.mode_D_inv&(K.y==1))],'c115','Pro115 cis, investigator mode filter (regex D|E_FR4)')
t(K[~(K.mode_D_true&(K.y==1))],'c115','Pro115 cis, corrected filter (Dplus|Dminus|E_FR4 only)')
print('modes of the 24 investigator Pro115 error chains:'); e=K[(K.y==1)&~K.mode_D_inv]; print(e.modes.value_counts().to_string())
pd.DataFrame(rows).to_csv(f'{OUT}/02_reproduce.csv',index=False)
D.to_csv(f'{OUT}/frame_em.csv.gz',index=False)
