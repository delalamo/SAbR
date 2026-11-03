"""(a) Among mode-A cases whose N-terminus is not modelled: does the shifted model put the first residue at IMGT<=1
(threaded from the first visible strand position) equally often whether the missing residues are absent from SEQRES or present
but unmodelled? (b) BH across this verification's tests."""
import pandas as pd, numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
D=pd.read_csv('v02_modeA_case_detail.csv')
import ast
D['a0']=D.anarci_first.map(lambda s: ast.literal_eval(s)[0]); D['s0']=D.sabr_first.map(lambda s: ast.literal_eval(s)[0])
D=D[D.a0>1]
D['to_start']=D.s0<=1
t=pd.crosstab(D.trunc,D.to_start); print(t); print(stats.fisher_exact(t.values))
print(D.groupby('trunc').apply(lambda g: pd.Series({'n':len(g),'med_shift':(g.s0-g.a0).median(),'frac_s0_le1':g.to_start.mean()})))
R=pd.read_csv('v04_model_results.csv')
tests=R[R.p.notna()][['test','term','OR','lo','hi','p']].copy()
extra=pd.DataFrame([dict(test='M2 germline-alignment truncation flag (ANARCI-free)',term='gl_trunc',OR=5.47,lo=2.97,hi=10.07,p=4.9e-8),
                    dict(test='mode-A cases with unbuilt N-term: first residue threaded to IMGT<=1, trunc vs present-unmodelled',term='fisher',OR=np.nan,p=stats.fisher_exact(t.values)[1])])
tests=pd.concat([tests,extra],ignore_index=True)
tests['q_BH']=multipletests(tests.p,method='fdr_bh')[1]
tests.to_csv('v07_all_tests_bh.csv',index=False)
pd.set_option('display.width',220); print(tests.round(4).to_string())
