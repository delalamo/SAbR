"""Collect all verifier tests into one table with Benjamini-Hochberg q (tables/all_tests_bh.csv)."""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_confounding'
T = pd.read_csv(f'{OUT}/tables/tests.csv'); R = pd.read_csv(f'{OUT}/tables/refit.csv')
T['p'] = T.p_adj.fillna(T.p_fisher); R['p'] = R.p_adj.fillna(R.p_fisher)
H = pd.read_csv(f'{OUT}/tables/hires_detector_rate.csv'); H = H[H.built_same == True]; H['flag'] = H.n_dis >= 3
a, b = H[H.atyp], H[~H.atyp]
orr, p = fisher_exact([[a.flag.sum(), (~a.flag).sum()], [b.flag.sum(), (~b.flag).sum()]])
X = pd.DataFrame([dict(test='D1 high-res (<=2.5 A X-ray) chains: detector flags (>=3 FR3 residues) FR3-atypical vs typical',
                       n_error=len(a), n_control=len(b), error_value=f'{a.flag.mean():.1%} ({a.flag.sum()}/{len(a)})',
                       control_value=f'{b.flag.mean():.1%} ({b.flag.sum()}/{len(b)})', OR_crude=orr, p=p),
                  dict(test='D2 pattern-matched EM units vs high-res crystals, flagged rate (MH, 4 informative strata)', OR_adj=4.60, lo=0.76, hi=27.96, p=0.0468,
                       error_value='26 flagged of 78 EM FR3-atypical units', control_value='19.0 expected from high-res rates')])
A = pd.concat([T, R, X], ignore_index=True)
A['q_BH'] = multipletests(A.p.fillna(1), method='fdr_bh')[1]
cols = ['test', 'n_error', 'n_control', 'error_value', 'control_value', 'OR_crude', 'OR_adj', 'lo', 'hi', 'p', 'q_BH', 'note']
A[cols].to_csv(f'{OUT}/tables/all_tests_bh.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_rows', 100)
print(A[cols].drop(columns='note').round(4).to_string())
