"""Collect all tests from v03/v06/v07, apply Benjamini-Hochberg, write summary table."""
import pandas as pd, numpy as np, re
from statsmodels.stats.multitest import multipletests
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
a=pd.read_csv(f'{OUT}/v03_models.csv'); a['source']='v03'
b=pd.read_csv(f'{OUT}/v06_models.csv'); b['source']='v06'; b['feature']='seq_trunc'
rows=[]
for line in open(f'{OUT}/v07_finer.log'):
    m=re.match(r'(.*): OR ([\d.]+) \[([\d.]+)-([\d.]+)\] p=([\de.-]+)',line.strip())
    if m and 'Mantel' not in line: rows.append(dict(model=m.group(1),feature='seq_trunc',OR=float(m.group(2)),lo=float(m.group(3)),hi=float(m.group(4)),p=float(m.group(5)),source='v07'))
    m=re.match(r'Mantel-Haenszel OR .*: ([\d.]+) \[([\d.]+)-([\d.]+)\] p=([\de.-]+);',line.strip())
    if m: rows.append(dict(model='Mantel-Haenszel across model-start strata (start>1)',feature='seq_trunc',OR=float(m.group(1)),lo=float(m.group(2)),hi=float(m.group(3)),p=float(m.group(4)),source='v07'))
T=pd.concat([a,b,pd.DataFrame(rows)],ignore_index=True)
T=T[T.p.notna()]
T['q_BH']=multipletests(T.p,method='fdr_bh')[1]
T.to_csv(f'{OUT}/v09_all_tests_bh.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',80)
print(T[['source','model','feature','n_err','n_ctrl','err_rate','ctrl_rate','OR','lo','hi','p','q_BH']].round(4).to_string())
