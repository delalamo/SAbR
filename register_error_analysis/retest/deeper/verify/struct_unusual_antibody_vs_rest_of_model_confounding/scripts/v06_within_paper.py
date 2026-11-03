"""Further confounding checks on the population table: within-paper (paper fixed effects) comparison, last-author
clustering, spline in deposition date, small-rest exclusion, and where the Q-score gap disagreement comes from."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_confounding/tables'
E = pd.read_csv(f'{OUT}/population_entries.csv')
P = E[E.grp.isin(['error', 'clean'])].copy(); P['case'] = P.err_first
rows = []


def fit(name, y, df, rhs, group='paper'):
    vars_ = [v for v in set([y, 'case', group] + [t for t in ['ybin', 'res', 'tdep', 'paper', 'sars2'] if t in rhs or t == group])]
    x = df[vars_].replace([np.inf, -np.inf], np.nan).dropna()
    m = smf.ols(f'{y} ~ case + {rhs}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[group])[0]})
    rows.append(dict(analysis=name, y=y, rhs=rhs, cluster=group, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                     case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(), effect=m.params['case'],
                     ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case']))


for y in ['gap_V_A', 'gap_V_Cl', 'gap_Vex_A', 'O_Cl', 'gap_V_Q']:
    fit('cluster by last author', y, P, 'C(ybin) + res', 'last_author')
    fit('spline in deposition date + res + SARS-CoV-2', y, P, 'bs(tdep, df=5) + res + sars2')
    fit('rest of model >= 200 residues', y, P[P.rest_n >= 200], 'C(ybin) + res')
    mixed = P[P.paper.isin(P.groupby('paper').case.agg(lambda c: c.nunique() == 2).pipe(lambda s: s[s].index))]
    fit('within paper: paper fixed effects (papers with both error and clean entries)', y, mixed, 'C(paper) + res')
    fit('same papers, no paper FE', y, mixed, 'C(ybin) + res')
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/population_within_paper.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(R.round(4).to_string())
print('papers with both:', mixed.paper.nunique(), 'entries', len(mixed), mixed.case.sum())
# Q gap by group
E['g2'] = np.where(E.role_su == 'case', 'error: struct_unusual case entry', np.where(E.grp == 'error', 'error: later entry, same unit',
          np.where(E.role_su == 'control', 'clean: struct_unusual control', E.grp)))
print(E.groupby('g2')[['gap_V_A', 'gap_V_Cl', 'gap_V_Q', 'V_Q', 'O_Q', 'O_A', 'O_Cl', 'res', 'year']].agg(['mean', 'median']).round(3).T.to_string())
print(E.groupby('g2').size())
