"""DE2 checks with the extra samples: fresh matched controls, correct chains in the same error entries,
and copies of the SAME antibody (identical V sequence) in the same entry that carry no run."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, wilcoxon
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_integrity'
A=pd.read_csv(f'{OUT}/analysis_table.csv')
G=pd.read_csv(f'{OUT}/sample_groups_extra.csv').merge(pd.read_csv(f'{OUT}/fw_recomputed_extra.csv'),on=['pdb','chain'],how='left')
G['yb']=pd.cut(G.year,[0,2019,2020,2021,2023,2030],labels=['<=2019','2020','2021','2022-23','>=2024']).astype(str)
G['ct']=np.where(G.vhh==1,'VHH',G.ctype)
CH=pd.read_csv(f'{SP}/deeper/chains.csv.gz')[['pdb','chain','vseq']]
e=A[(A.group=='error')&(A.confirm!='not confirmed')]
fm=G[G.group=='fresh_matched'].drop_duplicates('unit')
rows=[]
def logit(e_,c_,f):
    d=pd.concat([e_.assign(err=1),c_.assign(err=0)]).dropna(subset=[f]); d['x']=(d[f]-d[f].mean())/d[f].std()
    m=smf.logit('err ~ x + C(yb) + res + C(ct)',data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
    return np.exp(m.params['x']),m.pvalues['x'],int(d.err.sum()),int((d.err==0).sum())
for f in ['rep_rmsd','fr12_rmsd','panel_min']:
    OR,p,ne,nc=logit(e,fm,f)
    rows.append(dict(test='confirmed D/E vs FRESH matched clean EM',feature=f,n_err=ne,n_ctl=nc,err_median=e[f].median(),ctl_median=fm[f].median(),
                     err_mean=e[f].mean(),ctl_mean=fm[f].mean(),OR=OR,p=p))
# same-entry correct chains (no run of any kind), one per (pdb, unit) vs the error chain of that entry
se=G[G.group=='same_entry_correct']
ea=A[(A.group=='error')]  # one chain per unit; use all D/E error chains to pair by entry
F1=pd.read_csv(f'{OUT}/fw_recomputed.csv'); Gall=pd.read_csv(f'{OUT}/sample_groups.csv').merge(F1,on=['pdb','chain'])
errch=Gall[Gall.group=='error'].merge(pd.read_pickle(f'{SP}/deeper/cache/case_files_DE/err_struct.pkl')[['pdb','chain','confirm']],on=['pdb','chain'],how='left')
errch=errch[errch.confirm!='not confirmed']
pe=errch.groupby('pdb').rep_rmsd.mean(); ps=se.groupby('pdb').rep_rmsd.mean()
both=pd.concat([pe.rename('err'),ps.rename('ok')],axis=1).dropna()
rows.append(dict(test='same entry: D/E error chains vs correct chains (entry-paired Wilcoxon)',feature='rep_rmsd',n_strata=len(both),
                 err_mean=both.err.mean(),ctl_mean=both.ok.mean(),err_median=both.err.median(),ctl_median=both.ok.median(),p=wilcoxon(both.err-both.ok).pvalue,
                 note=f'entries err>ok {(both.err>both.ok).sum()}/{len(both)}'))
rows.append(dict(test='same-entry correct chains vs matched clean (MWU, unit level)',feature='rep_rmsd',n_err=se.drop_duplicates('unit').shape[0],
                 n_ctl=(A.group=='matched_clean').sum(),err_median=se.drop_duplicates('unit').rep_rmsd.median(),ctl_median=A[A.group=='matched_clean'].rep_rmsd.median(),
                 p=mannwhitneyu(se.drop_duplicates('unit').rep_rmsd.dropna(),A[A.group=='matched_clean'].rep_rmsd).pvalue))
# same antibody (identical vseq) in the same entry: error copy vs run-free copy
se2=se.merge(CH,on=['pdb','chain']); ec=errch.merge(CH,on=['pdb','chain'])
pairs=ec.merge(se2[['pdb','vseq','chain','rep_rmsd','fr12_rmsd']],on=['pdb','vseq'],suffixes=('_err','_ok'))
pp=pairs.groupby('pdb')[['rep_rmsd_err','rep_rmsd_ok','fr12_rmsd_err','fr12_rmsd_ok']].mean()
pp.to_csv(f'{OUT}/same_antibody_same_entry_pairs.csv')
if len(pp)>=3:
    rows.append(dict(test='same antibody, same entry: error copy vs run-free copy (Wilcoxon over entries)',feature='rep_rmsd',n_strata=len(pp),
                     err_mean=pp.rep_rmsd_err.mean(),ctl_mean=pp.rep_rmsd_ok.mean(),p=wilcoxon(pp.rep_rmsd_err-pp.rep_rmsd_ok).pvalue,
                     note=f'entries err>ok {(pp.rep_rmsd_err>pp.rep_rmsd_ok).sum()}/{len(pp)}; '+';'.join(f'{i}:{a:.2f}/{b:.2f}' for i,a,b in zip(pp.index,pp.rep_rmsd_err,pp.rep_rmsd_ok))))
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/verify_extra_tests.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',200); print(T.round(4).to_string())
print(both.round(2).to_string())
