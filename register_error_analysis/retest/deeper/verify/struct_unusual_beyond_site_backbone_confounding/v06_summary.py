"""Collect the key verification tests from v01-v05 into one table with Benjamini-Hochberg q. Output tables/v06_key_tests.csv"""
import pandas as pd, numpy as np
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_confounding/tables'
rows = []
v1 = pd.read_csv(f'{OUT}/v01_models.csv')
for _, r in v1.iterrows():
    if r.model in ('base, cluster=last author', 'kitchen sink (all of the above)', 'paper-level means (1 row per paper x role)',
                   'last-author-level means (1 row per lab x role)', '+ refinement software flags',
                   '+ same-entry non-V geometry (entry modelling quality)', 'non-SARS-CoV-2 only', '2020-2022 only'):
        rows.append(dict(source='v01', test=f'{r.metric} | {r.model}', n1=r.n_case, n0=r.n_ctrl, mean1=r.case_mean,
                         mean0=r.ctrl_mean, effect=r.effect, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p))
v2 = pd.read_csv(f'{OUT}/v02_strict_models.csv')
for _, r in v2[v2.subset == 'all'].iterrows():
    if r.metric.endswith(('_E3', '_E4', '_E5')):
        rows.append(dict(source='v02', test=f'{r.metric} (strict outside region)', n1=r.n_case, n0=r.n_ctrl,
                         mean1=r.case_mean, mean0=r.ctrl_mean, effect=r.effect, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p))
v2d = pd.read_csv(f'{OUT}/v02_distance_profile.csv')
for _, r in v2d.iterrows():
    rows.append(dict(source='v02', test=f'non-favoured Ramachandran, distance bin {r.bin}', n1=r.n_case_units,
                     n0=r.n_ctrl_units, mean1=r.case_pooled_pct, mean0=r.ctrl_pooled_pct, effect=r.effect_pp,
                     ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p))
v3 = pd.read_csv(f'{OUT}/v03_models.csv')
for _, r in v3[v3.model.isin(['base', 'non-SARS-CoV-2'])].iterrows():
    if r.metric.endswith('_E3'):
        rows.append(dict(source='v03', test=f'{r.arm} independent controls: {r.metric} | {r.model}', n1=r.n_case,
                         n0=r.n_ctrl, mean1=r.case_mean, mean0=r.ctrl_mean, effect=r.effect, ci_lo=r.ci_lo,
                         ci_hi=r.ci_hi, p=r.p))
v4 = pd.read_csv(f'{OUT}/v04_within_entry_models.csv')
for _, r in v4[v4.metric.isin(['allowed', 'clash', 'rota'])].iterrows():
    rows.append(dict(source='v04', test=f'within-entry (entry FE): error chain vs {r.comparator}: {r.metric}',
                     n1=r.n_err_chains, n0=r.n_comp_chains, mean1=r.err_mean, mean0=r.comp_mean,
                     effect=r.effect_entryFE, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p))
v5 = pd.read_csv(f'{OUT}/v05_decomposition.csv')
for _, r in v5.iterrows():
    if 'interaction' in r.model or r.model.startswith('all cases'):
        continue
    rows.append(dict(source='v05', test=f'{r.metric} | {r.model}', n1=r.n_1, n0=r.n_0, mean1=r.mean_1, mean0=r.mean_0,
                     effect=r.effect, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p))
T = pd.DataFrame(rows)
T['q_bh'] = multipletests(T.p, method='fdr_bh')[1]
T.to_csv(f'{OUT}/v06_key_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 95)
print(T.round(4).to_string())
