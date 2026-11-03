"""Mode A: is 'first built IMGT residue' (min SAbR IMGT label, investigator/deepdive) circular for N-terminal register
errors? Compare with the ANARCI IMGT position of the first built SEQRES residue (alignment, v02) for paper_strict,
rest and controls; recompute the Mann-Whitney and >3 tests both ways. Also: exact +1 vs other positive shifts on the
C-flank, and EM-only starting-model counts at chain/unit/paper level. Output: tables/v05.log"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_local_hazards/tables'
LH='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/local_hazards/tables'
log=open(f'{OUT}/v05.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); log.write(s+'\n')
pd.set_option('display.width',250)
U=pd.read_pickle(f'{LH}/analysis_units.pkl.gz')
G=pd.read_pickle(f'{OUT}/v02_gap_frame.pkl.gz')[['pdb','chain','first_built_v']]
U=U.merge(G,on=['pdb','chain'],how='left')
def units(d): return d.sort_values(['pdb','chain']).drop_duplicates(['mode','paper','vseq'])
A=U[(U['mode']=='A')&U.EM]
for cs,ct in [('paper_strict','ctl_paper'),('paper_incl','ctl_paper'),('rest','ctl_full')]:
    c=units(A[A.cset==cs]); k=units(A[A.cset==ct]); k=k[k.light.isin(set(c.light))]
    if cs=='paper_strict': P(c[['pdb','chain','ctype','imgt0','imgt1','delta','first_imgt','first_built_v']].to_string(index=False))
    for col in ['first_imgt','first_built_v']:
        a,b=c[col].dropna(),k[col].dropna()
        P(f'{cs:12s} {col:14s} n={len(a)} median {a.median()} vs {b.median()} MW p={mannwhitneyu(a,b).pvalue:.3g} | >3: {(a>3).sum()}/{len(a)} ({100*(a>3).mean():.0f}%) vs {100*(b>3).mean():.1f}% Fisher p={fisher_exact([[(a>3).sum(),(a<=3).sum()],[(b>3).sum(),(b<=3).sum()]])[1]:.3g} | >1: {100*(a>1).mean():.0f}% vs {100*(b>1).mean():.1f}%')
    d=c.first_imgt-c.first_built_v; P(f'   {cs}: SAbR label minus ANARCI position of first built residue: {d.value_counts().to_dict()}')
# C-flank exact +1
R=pd.read_csv(f'{SP}/deeper/runs_all.csv')
P('\nexact delta distribution of C-flank (118-128) runs, all first-occurrence non-1a/4/3 runs, deduplicated events:')
s=R[(~R.copied_or_artifact.astype(bool))&~R.cls.astype(str).isin(['1a','4','3'])&(R.imgt0<=128)&(R.imgt1>=118)].drop_duplicates(['paper','seqkey','imgt0','delta'])
P(s.delta.value_counts().sort_index().to_dict())
log.close()
