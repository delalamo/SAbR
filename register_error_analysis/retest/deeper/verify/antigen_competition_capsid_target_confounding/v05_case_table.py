# Case table of the capsid error entries (EM) with the confounder flags, and the within-capsid lq_z check against the general pattern.
import numpy as np, pandas as pd
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_confounding'
E=pd.read_csv(f'{OUT}/em_entries_verify.csv.gz')
org=pd.read_csv(f'{SP}/deeper/cache/verify_capsid_confounding/chain_ab_org.csv')
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','err_first','L3','v_gene','v_id'])
x=ch[ch.err_first].merge(org,on=['pdb','chain'])
g=x.groupby('pdb').agg(err_chain_org=('ab_org',lambda s:';'.join(sorted(set(map(str,s))))),err_L3=('L3','max')).reset_index()
C=E[(E.cap==1)&(E.y==1)].merge(g,on='pdb',how='left')
C['flag']=np.where(C.y_art.isna(),'confirmed numbering artifact',np.where(C.y_canon.isna(),'CDR-internal disagreement only (no register shift)','register shift'))
C=C.sort_values(['lab','pdb'])
cols=['pdb','lab','dyear','res','modes','err_chain_org','err_L3','flag']
print(C[cols].to_string(index=False)); C[cols].to_csv(f'{OUT}/capsid_error_entries_flags.csv',index=False)
print('\nby network:'); print(C.groupby('lab').agg(entries=('pdb','size'),papers=('paper','nunique'),artifact=('flag',lambda s:(s=='confirmed numbering artifact').sum()),cdr_internal=('flag',lambda s:(s.str.startswith('CDR')).sum())).to_string())
print('\nlq_z (worst chain) error vs clean, within capsid and within other EM entries:')
for cap in [1,0]:
    d=E[E.cap==cap].dropna(subset=['lqz_min']); a=d[d.y==1].lqz_min; b=d[d.y==0].lqz_min
    print(f'  cap={cap}: error {a.mean():.2f} (n={len(a)}) vs clean {b.mean():.2f} (n={len(b)}); diff {a.mean()-b.mean():.2f}; MW p={stats.mannwhitneyu(a,b).pvalue:.2g}')
