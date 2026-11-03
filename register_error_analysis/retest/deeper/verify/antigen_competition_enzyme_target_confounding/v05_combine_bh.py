# One Benjamini-Hochberg correction over all tests in this verification (v02, v03, v04).
import pandas as pd
from statsmodels.stats.multitest import multipletests
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_confounding'
fs=['tests_enzyme_confounding.csv','tests_definition_role_gap.csv','tests_enzyme_definition.csv']
R=pd.concat([pd.read_csv(f'{OUT}/{f}').assign(source=f) for f in fs],ignore_index=True)
R=R[R.p.notna()].copy(); R['q_BH_all']=multipletests(R.p,method='fdr_bh')[1]
cols=['test','rate_in','rate_out','OR','lo','hi','p','q_BH_all','n','n_err','source']
R[cols].to_csv(f'{OUT}/all_tests_bh.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',80)
print(R[cols].round(4).to_string())
