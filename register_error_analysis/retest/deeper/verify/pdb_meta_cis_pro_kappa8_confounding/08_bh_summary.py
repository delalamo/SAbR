"""Collect all tests, apply Benjamini-Hochberg across them, and run the entry-marker test without the 2 largest papers."""
import numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',100)
A=pd.read_csv('models_main.csv'); B=pd.read_csv('models_more.csv'); W=pd.read_csv('within_paper_entry.csv')
W=W.rename(columns={'condlogit_OR':'OR','cl_lo':'lo','cl_hi':'hi','cl_p':'p'})
W['test']='W chain level, conditional logit '+W.test
T=pd.concat([A,B,W[['test','err_cis','ctrl_cis','OR','lo','hi','p']]],ignore_index=True)
ok=T.p.notna()&(T.p>0)&(T.OR>1e-6)  # drop separated fits (OR ~ 0, p meaningless)
T['q']=np.nan; T.loc[ok,'q']=multipletests(T.loc[ok,'p'],method='fdr_bh')[1]
T.loc[~ok&T.p.notna(),'note']='complete separation (0 error chains cis); OR not estimable'
T.to_csv('all_tests_bh.csv',index=False)
print(T[['test','n_err','n_ctrl','err_cis','ctrl_cis','OR','lo','hi','p','q','note']].to_string(index=False,float_format=lambda v:f'{v:.3g}'))
