"""Paper-level version of the investigator's conkit-selection contrasts (paper-confirmed vs conkit-missed in window), and Zhou share in T8b."""
import numpy as np, pandas as pd
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'; DEEP='/home/user/SAbR/register_error_analysis/retest/deeper'
MF='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_model_fingerprint/tables'
ZHOU='10.1038/s41422-021-00487-9'; L=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); L.append(s)
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',dtype={'chain':str}).drop(columns=['seq'],errors='ignore')
C=C.merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz',dtype={'chain':str}),on=['pdb','chain'])
C=C[~C.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)].copy(); C['clu']=C.paper.fillna(C.pdb)
C['missed']=C.err_first.astype(bool)&C.in_paper_window&~C.paper_case_strict&~C.paper_case_incl_copies
F=pd.read_csv(f'{MF}/tables/fw_computed.csv',dtype={'chain':str})[['pdb','chain','rep_rmsd']]
VD=f'{DEEP}/verify/case_files_DE_DE2_framework_distortion_integrity'
F0=pd.concat([pd.read_csv(f'{VD}/fw_recomputed.csv',dtype={'chain':str}),pd.read_csv(f'{VD}/fw_recomputed_extra.csv',dtype={'chain':str})])[['pdb','chain','rep_rmsd']]
X=C.merge(pd.concat([F,F0]).drop_duplicates(['pdb','chain']).dropna(),on=['pdb','chain'])
T=pd.concat([pd.read_pickle(f'{SP}/deeper/cache/verify_su_gap/tallies.pkl'),pd.read_pickle(f'{SP}/paperfp/tallies_xray.pkl')]); T['chain']=T.chain.astype(str)
V=T[(T.cat=='V')&T.dbin.isin(['d4_10','d11p','clean'])].groupby(['pdb','chain'])[['n_rama','allowed']].sum(); V=V[V.n_rama>=30]; V['pA']=100*V.allowed/V.n_rama
R=C.merge(V.reset_index(),on=['pdb','chain'])
for lab,df,v in [('fw RMSD',X,'rep_rmsd'),('rama beyond site',R,'pA')]:
    d=df[df.method=='EM']; a=d[d.paper_case_strict].groupby('clu')[v].mean(); b=d[d.missed].groupby('clu')[v].mean()
    az=a.drop(ZHOU,errors='ignore')
    P(f'EM {lab}: paper-confirmed papers mean {a.mean():.3g} (n={len(a)}) vs conkit-missed papers {b.mean():.3g} (n={len(b)}): Welch p={stats.ttest_ind(a,b,equal_var=False).pvalue:.2g}, MW p={stats.mannwhitneyu(a,b).pvalue:.2g}; without Zhou paper mean {az.mean():.3g}, Welch p={stats.ttest_ind(az,b,equal_var=False).pvalue:.2g}')
    ca=d[d.paper_case_strict]; P(f'   chain level: paper {ca[v].mean():.3g} (Zhou chains {ca[ca.clu==ZHOU][v].mean():.3g}, n={int((ca.clu==ZHOU).sum())}) vs missed {d[d.missed][v].mean():.3g}')
ct=pd.read_csv(f'{MF}/tables/canonical_tests.csv'); P('canonical_tests columns:',list(ct.columns)[:12])
open(f'{OUT}/v04_selection_paperlevel.txt','w').write('\n'.join(L)+'\n')
