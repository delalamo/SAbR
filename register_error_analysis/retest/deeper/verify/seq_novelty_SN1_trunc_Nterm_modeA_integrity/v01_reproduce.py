"""Reproduce the SN1 counts from the investigator's units table and check the case/control definitions."""
import pandas as pd, numpy as np
from scipy import stats
T='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
U=pd.read_csv(f'{T}/units.csv.gz',low_memory=False)
EM=U.method_u=='EM'
A=EM&U.err_first_u&U.modes_u.fillna('').str.contains('A_FR1Nterm')
cts=U[A].ctype.unique()
C=EM&U.clean&U.ctype.isin(cts)
print('cases',A.sum(),'ctypes',U[A].ctype.value_counts().to_dict())
print('controls',C.sum(), U[C].ctype.value_counts().to_dict())
print('trunc cases',U[A].seq_nt_trunc.sum(), U[A].seq_nt_trunc.mean().round(3))
print('trunc ctrl',U[C].seq_nt_trunc.sum(), U[C].seq_nt_trunc.mean().round(3))
print(pd.crosstab(U[A|C].ctype,[U[A|C].seq_nt_trunc,A[A|C]]))
print('first imgt cases',U[A].seq_first_imgt.value_counts().sort_index().to_dict())
print('first imgt ctrl',U[C].seq_first_imgt.value_counts().sort_index().to_dict())
t=pd.crosstab(A[A|C],U[A|C].seq_nt_trunc); print(t, stats.fisher_exact(t.values))
# any EM first error
AE=EM&U.err_first_u; CE=EM&U.clean
print('all EM errors trunc',U[AE].seq_nt_trunc.mean().round(3),AE.sum(),'clean',U[CE].seq_nt_trunc.mean().round(3),CE.sum())
# check the error chain vs representative chain
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
print('dbrow NaN or <0 in cases', (U[A].dbrow<0).sum(), U[A].dbrow.isna().sum())
cs=U[A][['unit','pdb','chain','paper','last_author','ctype','seq_first_imgt','dep_year','n_chains']]
ra=runs[runs['mode']=='A_FR1Nterm']
cs['rep_chain_has_A_run']=[((ra.pdb==p)&(ra.chain==c)).any() for p,c in zip(cs.pdb,cs.chain)]
print('rep chain itself has mode A run:',cs.rep_chain_has_A_run.sum(),'/',len(cs))
cs.to_csv('v01_modeA_cases.csv',index=False)
