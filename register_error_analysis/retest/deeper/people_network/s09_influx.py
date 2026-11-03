#!/usr/bin/env python3
"""Did an influx of teams new to antibody cryo-EM drive the 2020-21 peak?  Share of new teams by deposition year,
error rate by year x new-team flag, and a direct standardisation of the period error rates to a fixed team mix."""
import sys
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
from common import OUT, CACHE
S = pd.read_csv(f'{CACHE}/study_experience.csv.gz')
S['yr'] = pd.cut(S.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100], labels=['<=2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)
t = S.groupby('yr').agg(n=('y', 'size'), err_rate=('y', 'mean'), share_new_team=('first_abem_team_na', 'mean'))
r = S.groupby(['yr', 'first_abem_team_na']).y.agg(['size', 'mean']).unstack()
r.columns = [f'{a}_newteam{b}' for a, b in r.columns]
t = t.join(r)
w = S.first_abem_team_na.mean()   # overall share of new teams
t['err_rate_std_to_overall_mix'] = t['mean_newteam1'] * w + t['mean_newteam0'] * (1 - w)
t.round(3).to_csv(f'{OUT}/tables/influx_by_year.csv')
print(t.round(3).to_string())

# does the new-team penalty disappear from 2024?  (exploratory split suggested by the table above)
import statsmodels.formula.api as smf, statsmodels.api as sm
S['late'] = (S.year >= 2024).astype(int)
rows = []
for nm, d in [('<=2023', S[S.late == 0]), ('>=2024', S[S.late == 1])]:
    m = smf.glm('y ~ first_abem_team_na + C(ycat) + res + l2_nv', d, family=sm.families.Binomial()).fit()
    ci = np.exp(m.conf_int().loc['first_abem_team_na']).values
    rows.append(dict(stratum=nm, n=len(d), n_err=int(d.y.sum()), OR=np.exp(m.params['first_abem_team_na']), lo=ci[0], hi=ci[1], p=m.pvalues['first_abem_team_na']))
m = smf.glm('y ~ first_abem_team_na * late + C(ycat) + res + l2_nv', S, family=sm.families.Binomial()).fit()
rows.append(dict(stratum='interaction new_team x (>=2024)', n=len(S), n_err=int(S.y.sum()), OR=np.exp(m.params['first_abem_team_na:late']),
                 lo=np.exp(m.conf_int().loc['first_abem_team_na:late'][0]), hi=np.exp(m.conf_int().loc['first_abem_team_na:late'][1]),
                 p=m.pvalues['first_abem_team_na:late']))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/tables/new_team_by_era.csv', index=False)
print(R.round(4).to_string())

# which error modes do new teams make?  (error studies only; a study counts once per mode)
from common import SP
from scipy.stats import fisher_exact
E = pd.read_csv(f'{CACHE}/entries_people.csv.gz')
R = pd.read_csv(f'{SP}/runs_first_occurrence.csv', usecols=['pdb', 'mode', 'method'])
R = R[R.method == 'EM']
R['study'] = R.pdb.map(dict(zip(E.pdb, E.study)))
M = R.dropna(subset=['study']).drop_duplicates(['study', 'mode'])
M = M.merge(S[S.y == 1][['study', 'first_abem_team_na']], on='study')
n_new = S[(S.y == 1)].first_abem_team_na.sum(); n_old = (S.y == 1).sum() - n_new
rows = []
for md, g in M.groupby('mode'):
    a = int(g.first_abem_team_na.sum()); b = len(g) - a
    orr, p = fisher_exact([[a, n_new - a], [b, n_old - b]])
    rows.append(dict(mode=md, new_team_err_studies=f'{a}/{n_new}', experienced_err_studies=f'{b}/{n_old}', OR=orr, p=p))
MM = pd.DataFrame(rows); MM.to_csv(f'{OUT}/tables/modes_by_new_team.csv', index=False)
print(MM.round(3).to_string())
