"""Combine all test families; global Benjamini-Hochberg on the primary p (M2 for model-based tests; matched expectation test for windows)."""
import pandas as pd, numpy as np
from statsmodels.stats.multitest import multipletests
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
A=pd.read_csv(f'{OUT}/tests_em_units.csv')
A=A.assign(family='unit features (EM, error vs clean)',p_primary=A.p_adj2,OR_primary=A.OR_adj2,lo_primary=A.lo2,hi_primary=A.hi2)
T=pd.read_csv(f'{OUT}/tests_typicality.csv')
T=T.assign(family='sequence typicality (PDB profile log-likelihood)',p_primary=T.p_M2,OR_primary=T.OR_M2,OR_adj=T.OR_M1,p_adj=T.p_M1,OR_adj3=T.OR_M3,p_adj3=T.p_M3,n_err3=T.n_err_M3,per='per SD')
W=pd.read_csv(f'{OUT}/window_rare_summary.csv')
W=W.rename(columns={'group':'mode','n_units':'n_err'}).assign(family='exact run window vs matched clean units (same chain type, year bin)',p_primary=W.p,
    err_mean=W.obs_frac,ctrl_mean=W.exp_frac,OR_primary=W.ratio,per='observed/expected')
J=pd.read_csv(f'{OUT}/D_jsplit.csv'); J=J[J.model=='M2'].assign(family='D modes: J-germline identity split',mode='D_all',p_primary=J.p,OR_primary=J.OR_per_SD,per='per SD')
cols=['family','mode','feature','n_err','n_ctrl','err_mean','ctrl_mean','per','OR_adj','p_adj','OR_primary','lo_primary','hi_primary','p_primary','n_err3','OR_adj3','p_adj3']
X=pd.concat([d.reindex(columns=cols) for d in (A,T,W,J)],ignore_index=True)
ok=X.p_primary.notna()
X.loc[ok,'q_global']=multipletests(X.loc[ok,'p_primary'],method='fdr_bh')[1]
ok3=X.p_adj3.notna(); X.loc[ok3,'q3_global']=multipletests(X.loc[ok3,'p_adj3'],method='fdr_bh')[1]
X=X.rename(columns={'OR_adj':'OR_M1','p_adj':'p_M1','OR_primary':'OR_M2','lo_primary':'lo_M2','hi_primary':'hi_M2','p_primary':'p_M2','n_err3':'n_err_M3','OR_adj3':'OR_M3','p_adj3':'p_M3','q_global':'q_M2_global','q3_global':'q_M3_global'})
X.to_csv(f'{OUT}/all_tests_bh.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
print(len(X), (X.q_M2_global<0.05).sum())
print(X[X.q_M2_global<0.05].sort_values('p_M2')[['mode','feature','n_err','err_mean','ctrl_mean','OR_M1','OR_M2','lo_M2','hi_M2','p_M2','q_M2_global','n_err_M3','OR_M3','q_M3_global']].round(4).to_string(index=False))
