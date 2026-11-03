"""DE2: is the framework deviation specific to the erroneous chain, or shared by run-free chains of the same entry?
Entry-paired comparisons, overall, within chain type, and for identical-V-sequence copies; three metrics."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_integrity'
CH=pd.read_csv(f'{SP}/deeper/chains.csv.gz')[['pdb','chain','vseq','ctype','vhh']]
F1=pd.read_csv(f'{OUT}/fw_recomputed.csv'); G1=pd.read_csv(f'{OUT}/sample_groups.csv')
err=G1[G1.group=='error'].merge(F1,on=['pdb','chain']).merge(pd.read_pickle(f'{SP}/deeper/cache/case_files_DE/err_struct.pkl')[['pdb','chain','confirm']],on=['pdb','chain'],how='left')
err=err[err.confirm!='not confirmed']
ok=pd.read_csv(f'{OUT}/sample_groups_extra.csv').merge(pd.read_csv(f'{OUT}/fw_recomputed_extra.csv'),on=['pdb','chain'])
ok=ok[ok.group=='same_entry_correct']
err=err.merge(CH[['pdb','chain','vseq']],on=['pdb','chain']); ok=ok.merge(CH[['pdb','chain','vseq']],on=['pdb','chain'])
for d in (err,ok): d['ct']=np.where(d.vhh==1,'VHH',d.ctype)
rows=[]
for f in ['rep_rmsd','fr12_rmsd','panel_min']:
    for lab,keys in [('entry',['pdb']),('entry x chain type',['pdb','ct']),('entry x identical V sequence',['pdb','vseq'])]:
        a=err.groupby(keys)[f].mean().rename('err'); b=ok.groupby(keys)[f].mean().rename('ok')
        m=pd.concat([a,b],axis=1).dropna()
        if len(m)<3: rows.append(dict(pairing=lab,feature=f,n_pairs=len(m))); continue
        rows.append(dict(pairing=lab,feature=f,n_pairs=len(m),n_entries=m.reset_index().pdb.nunique(),err_mean=m.err.mean(),ok_mean=m.ok.mean(),
                         err_median=m.err.median(),ok_median=m.ok.median(),median_diff=(m.err-m.ok).median(),n_err_gt_ok=int((m.err>m.ok).sum()),
                         p_wilcoxon=wilcoxon(m.err-m.ok).pvalue))
        if lab=='entry x identical V sequence' and f=='rep_rmsd': m.round(3).to_csv(f'{OUT}/same_antibody_same_entry_pairs.csv')
T=pd.DataFrame(rows); k=T.p_wilcoxon.notna(); T.loc[k,'q_BH']=multipletests(T.loc[k,'p_wilcoxon'],method='fdr_bh')[1]
T.to_csv(f'{OUT}/verify_same_entry.csv',index=False)
pd.set_option('display.width',250); print(T.round(4).to_string())
print(pd.read_csv(f'{OUT}/same_antibody_same_entry_pairs.csv').to_string())
