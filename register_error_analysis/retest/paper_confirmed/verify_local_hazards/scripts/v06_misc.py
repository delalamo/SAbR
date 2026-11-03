"""D+ CDR3 length within chain type in the full set (rest) for comparison; EM-only starting-model counts at chain/unit/paper
level; X-ray paper set composition by year. Output: tables/v06.log"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_local_hazards/tables'
LH='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/local_hazards/tables'
log=open(f'{OUT}/v06.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); log.write(s+'\n')
U=pd.read_pickle(f'{LH}/analysis_units.pkl.gz')
def units(d): return d.sort_values(['pdb','chain']).drop_duplicates(['mode','paper','vseq'])
for meth,sel in [('EM',lambda d:d[d.EM]),('all',lambda d:d)]:
    for cs,ct in [('rest','ctl_full'),('paper_strict','ctl_paper')]:
        c=units(sel(U[(U.cset==cs)&(U['mode']=='D+')])); k=units(sel(U[(U.cset==ct)&(U['mode']=='D+')]))
        a,b=c.L3.dropna(),k.L3.dropna()
        P(f'{meth} {cs}: pooled types {a.median()} vs {b.median()} p={mannwhitneyu(a,b).pvalue:.3g}; heavy share {c.ctype.eq("H").mean():.2f} vs {k.ctype.eq("H").mean():.2f}')
        for h in [True,False]:
            a,b=c[c.ctype.eq('H')==h].L3.dropna(),k[k.ctype.eq('H')==h].L3.dropna()
            if len(a)>1: P(f'    {"heavy" if h else "light"}: n={len(a)} {a.median()} vs {b.median()} MW p={mannwhitneyu(a,b).pvalue:.3g}')
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False).merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz'),on=['pdb','chain'])
W=C[C.in_paper_window&(C.paper_case_strict|C.paper_control)&C.method.eq('EM')].copy()
W['grp']=np.where(W.irm.eq('none recorded'),'none',np.where(W.irm.eq('experimental (PDB) model'),'pdb','other'))
P('\nEM, window, strict cases + paper controls: starting model none vs PDB')
for lvl,dd in [('chain',W),('unit',W.sort_values('paper_case_strict',ascending=False).drop_duplicates(['paper','vseq'])),('paper',W.sort_values('paper_case_strict',ascending=False).drop_duplicates(['paper']))]:
    t=dd[dd.grp.isin(['none','pdb'])].groupby('grp').paper_case_strict.agg(['sum','count'])
    OR,p=fisher_exact([[t.loc['none','sum'],t.loc['none','count']-t.loc['none','sum']],[t.loc['pdb','sum'],t.loc['pdb','count']-t.loc['pdb','sum']]])
    P(f'  {lvl}: none {t.loc["none","sum"]}/{t.loc["none","count"]} vs pdb {t.loc["pdb","sum"]}/{t.loc["pdb","count"]}: OR {OR:.2f} p={p:.3g}')
x=C[C.paper_case_strict&C.method.ne('EM')]
P('\nX-ray strict chains by year:',x.groupby('year').size().to_dict(),'| papers',x.paper.nunique())
log.close()
