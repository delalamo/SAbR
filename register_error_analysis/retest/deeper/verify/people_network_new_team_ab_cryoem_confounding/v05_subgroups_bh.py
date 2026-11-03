#!/usr/bin/env python3
"""Fully adjusted subgroup models (era x nanobody), one-study-per-lab within <=2023, per-year ORs, and BH across all
verification tests (v02-v05)."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/people_network_new_team_ab_cryoem_confounding/tables'
S = pd.read_csv(f'{SP}/cache/verify_people_newteam_conf/study_frame.csv.gz')
S['lq_z_med'] = S.lq_z_med.fillna(S.lq_z_med.median()); S['q_med'] = S.q_med.fillna(S.q_med.median())
S['tier'] = S.tier.fillna('Unpublished')
rng = np.random.default_rng(5)
FULL = ('dep_dec + res_med + l2_nv + l2_ne + any_K + any_L + vhh_any + sars2 + mouse_any + atyp_any + min_vid + novel_any + aux_any + '
        'capsid + enzyme + first_fam + lq_z_med + q_med + unobs_max + af_any + exp_start_any + irm_none + rigid_any + flex_any + '
        'l_n_auth + C(ctry3) + C(tier)')
rows = []
for lab, d in [('<=2023, full adjustment', S[S.dep_year <= 2023]), ('>=2024, full adjustment', S[S.dep_year >= 2024]),
               ('<=2023 non-nanobody, full adjustment', S[(S.dep_year <= 2023) & (S.vhh_any == 0)]),
               ('any nanobody, full adjustment', S[S.vhh_any == 1]),
               ('<=2023, non-China, all-key flag, full adjustment', S[(S.dep_year <= 2023) & (S.china == 0)])]:
    term = 'new_all' if 'all-key' in lab else 'new'
    f = f'y ~ {term} + {FULL}'.replace(' + vhh_any', '' if 'nanobody' in lab else ' + vhh_any')
    if 'non-nanobody' in lab:   # complete separation with the full set (no errors in some strata): drop tier, country, capsid, enzyme
        f = f.replace(' + C(tier)', '').replace(' + C(ctry3)', ' + china').replace(' + capsid + enzyme', '')
    m = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.lab_key)[0]})
    ci = np.exp(m.conf_int().loc[term]).values
    rows.append(dict(test=lab, n=len(d), n_err=int(d.y.sum()), rate_new=d[d[term] == 1].y.mean(), rate_old=d[d[term] == 0].y.mean(),
                     OR=np.exp(m.params[term]), lo=ci[0], hi=ci[1], p=m.pvalues[term]))
# one study per lab within <=2023
d0 = S[S.dep_year <= 2023]
ors, ps = [], []
for _ in range(500):
    d = d0.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('lab_key')
    m = smf.glm('y ~ new + C(ycat) + res_med + l2_nv', d, family=sm.families.Binomial()).fit()
    ors.append(np.exp(m.params['new'])); ps.append(m.pvalues['new'])
rows.append(dict(test='<=2023, one study per lab (500 draws; median OR, median p)', n=d0.lab_key.nunique(), OR=np.median(ors),
                 lo=np.percentile(ors, 2.5), hi=np.percentile(ors, 97.5), p=np.median(ps), frac_p_lt_05=np.mean(np.array(ps) < 0.05)))
# per-year ORs (crude, exact)
from scipy.stats import fisher_exact
for yc, d in S.groupby('ycat'):
    t = [[((d.new == 1) & (d.y == 1)).sum(), ((d.new == 1) & (d.y == 0)).sum()], [((d.new == 0) & (d.y == 1)).sum(), ((d.new == 0) & (d.y == 0)).sum()]]
    o, p = fisher_exact(t)
    rows.append(dict(test=f'year {yc} crude (Fisher)', n=len(d), n_err=int(d.y.sum()), rate_new=d[d.new == 1].y.mean(), rate_old=d[d.new == 0].y.mean(), OR=o, p=p))
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/v05_subgroups.csv', index=False)
pd.set_option('display.width', 250)
print(T.round(4).to_string())

# ---- BH across verification tests (primary rows only: each model/test once)
parts = []
for f, col in [('v02_adjusted_models.csv', 'test'), ('v02_mantel_haenszel.csv', 'strata'), ('v03_heterogeneity.csv', 'test'),
               ('v03_lab_resampling_withinlab_ipw.csv', 'test'), ('v04_lab_structure_units.csv', 'test'), ('v05_subgroups.csv', 'test')]:
    d = pd.read_csv(f'{OUT}/{f}')
    pc = 'p' if 'p' in d else None
    if 'p_median' in d:
        d['p'] = d.p.fillna(d.p_median)
    d = d.rename(columns={col: 'test'})
    d['source'] = f
    keep = ['source', 'test'] + [c for c in ['n', 'n_err', 'OR', 'MH_OR', 'OR_median', 'lo', 'hi', 'p'] if c in d]
    parts.append(d[keep])
B = pd.concat(parts, ignore_index=True)
B['OR'] = B.OR.fillna(B.get('MH_OR')).fillna(B.get('OR_median'))
B = B.drop(columns=[c for c in ['MH_OR', 'OR_median'] if c in B])
ok = B.p.notna()
B.loc[ok, 'q_BH'] = multipletests(B.loc[ok, 'p'], method='fdr_bh')[1]
B.to_csv(f'{OUT}/v05_all_tests_bh.csv', index=False)
print('\nBH over', ok.sum(), 'tests'); print(B.round(4).to_string())
