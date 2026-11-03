"""Exact reproduction of the investigator's p-value from their own entry_vrpt.csv; identify the zero-filled entries;
test whether the V-minus-rest Q gap (density) explains the Ramachandran gap."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_integrity'
s = pd.read_csv(f'{INV}/sample.csv', dtype={'chain': str})
em = s.groupby('pdb').agg(res=('res', 'first'), ybin=('ybin', 'first'), role_cc=('role_cc', 'first'))
I = pd.read_csv(f'{INV}/entry_vrpt.csv').join(em, on='pdb')
I['case'] = (I.role_cc == 'case').astype(int)
for y in ['gap_rama_allowed', 'gap_clash', 'rest_clash', 'rest_rama_allowed']:
    x = I.dropna(subset=[y])
    m = smf.ols(f'{y} ~ case + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
    print(f'investigator data {y:18s} n={x.case.sum()}/{(1-x.case).sum()} effect {m.params.case:.3f} p {m.pvalues.case:.3g}')
print('zero-filled entries (v_n==0 or rest_n==0):')
print(I[(I.v_n == 0) | (I.rest_n == 0)][['pdb', 'role_cc', 'v_n', 'rest_n', 'gap_rama_allowed', 'gap_clash']].to_string())
# fixed: NaN where empty
J = I.copy()
for k in ['rama_allowed', 'clash', 'q']:
    J.loc[J.rest_n == 0, f'rest_{k}'] = np.nan; J.loc[J.v_n == 0, f'v_{k}'] = np.nan
    J[f'gap_{k}'] = J[f'v_{k}'] - J[f'rest_{k}']
for y in ['gap_rama_allowed', 'gap_clash', 'rest_clash', 'rest_rama_allowed']:
    x = J.dropna(subset=[y])
    m = smf.ols(f'{y} ~ case + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
    print(f'fixed (NaN when empty) {y:18s} n={x.case.sum()}/{(1-x.case).sum()} effect {m.params.case:.3f} ({m.conf_int().loc["case",0]:.2f}-{m.conf_int().loc["case",1]:.2f}) p {m.pvalues.case:.3g}')
# density adjustment, on my verified table without amyloid
E = pd.read_csv(f'{OUT}/entry_metrics_verify2.csv', index_col=0)
G = E[~E.amyloid & (E.rest_n > 0)].copy()
for y in ['gap_V_all_allowed', 'gap_V_m10_allowed', 'gap_V_errfree_chains_allowed', 'V_all_allowed']:
    for adj in ['', ' + gap_V_all_q', ' + V_all_q + rest_q']:
        x = G.dropna(subset=[y, 'gap_V_all_q', 'V_all_q', 'rest_q'])
        m = smf.ols(f'{y} ~ case + C(ybin) + res{adj}', x).fit(cov_type='HC1')
        print(f'{y:30s} adj[{adj:20s}] n={int(x.case.sum())}/{int((1-x.case).sum())} effect {m.params.case:.3f} p {m.pvalues.case:.3g}')
# correlation of Rama gap with Q gap
x = G.dropna(subset=['gap_V_all_allowed', 'gap_V_all_q'])
from scipy import stats
print('spearman gap_allowed vs gap_q all', stats.spearmanr(x.gap_V_all_allowed, x.gap_V_all_q))
