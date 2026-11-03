# Auxiliary-antibody comparison + BH over all verification tests
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_integrity'
rows = []
for lab, t in [('aux (text-rule role) units: host enzyme 9/51 vs transporter/GPCR 14/379', [[9, 42], [14, 365]]),
               ('host-enzyme units: auxiliary 9/51 vs antibody of interest 1/29', [[9, 42], [1, 28]]),
               ('first-occurrence entries: origin-3 run (chain end/unbuilt loop), host enzyme 11/15 vs other 65/155', [[11, 4], [65, 90]]),
               ('first-occurrence entries: B_CDR2FR3 mode, host enzyme 2/15 vs other 23/155', [[2, 13], [23, 132]]),
               ('first-occurrence entries: A_FR1Nterm mode, host enzyme 8/15 vs other 50/155', [[8, 7], [50, 105]])]:
    o, p = stats.fisher_exact(t); rows.append(dict(test=lab, stratum='EM', OR=o, p=p)); print(f'{lab}: OR {o:.2f} p={p:.2g}')
T = pd.concat([pd.read_csv(f'{OUT}/verify_tests.csv'), pd.read_csv(f'{OUT}/verify_tests_v02.csv'), pd.DataFrame(rows)], ignore_index=True)
T = T[T.p.notna()].copy(); T['q'] = multipletests(T.p, method='fdr_bh')[1]
T.drop(columns=[c for c in T.columns if c == 'q_within_verify']).to_csv(f'{OUT}/all_verify_tests_bh.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(T[['test', 'stratum', 'cases_in', 'cases_out', 'OR', 'p', 'q']].round(4).to_string())
