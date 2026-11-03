# One Benjamini-Hochberg correction over every test in this verification (v02, v03, v04)
import pandas as pd
from statsmodels.stats.multitest import multipletests
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_integrity'
fr=[]
for f in ['verify_tests.csv','consortium_tests.csv','extra_tests.csv']:
    d=pd.read_csv(f'{OUT}/{f}'); d['source']=f; fr.append(d)
T=pd.concat(fr,ignore_index=True); T=T[T.p.notna()].drop(columns=[c for c in T.columns if c.startswith('q_')])
T['q_BH_all']=multipletests(T.p,method='fdr_bh')[1]
T.to_csv(f'{OUT}/all_tests_bh.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',80)
print(len(T),'tests'); print(T[['test','capsid','other','OR','p','q_BH_all']].round(4).to_string())
