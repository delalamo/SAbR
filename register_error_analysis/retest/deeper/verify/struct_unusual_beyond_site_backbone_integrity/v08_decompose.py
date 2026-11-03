"""Decompose the case-chain excess: case chains in entries that also contain an error-free V chain vs case chains
in entries without one; and the error-free chains of those entries, each vs all control chains (adjusted)."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
D = pd.read_csv(f'{OUT}/v_entry_level_chains.csv', dtype={'chain': str})
D['ctrl'] = D.grp.str.startswith('ctrl')
has_clean = set(D[(D.grp == 'clean_in_case') & D.allowed.notna()].pdb)
D.loc[(D.grp == 'case_chain') & D.pdb.isin(has_clean), 'grp2'] = 'case_chain | entry has clean V chain'
D.loc[(D.grp == 'case_chain') & ~D.pdb.isin(has_clean), 'grp2'] = 'case_chain | no clean V chain in entry'
D.loc[(D.grp == 'clean_in_case'), 'grp2'] = 'clean chain in case entry'
rows = []
for g in D.grp2.dropna().unique():
    for y in ['allowed', 'clash']:
        x = D[(D.grp2 == g) | D.ctrl].dropna(subset=[y]).copy(); x['expo'] = (x.grp2 == g).astype(int)
        m = smf.ols(f'{y} ~ expo + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
        rows.append(dict(group=g, metric=y, n=int(x.expo.sum()), mean=x[x.expo == 1][y].mean(), ctrl_mean=x[x.expo == 0][y].mean(),
                         effect_vs_ctrl=m.params['expo'], ci_lo=m.conf_int().loc['expo', 0], ci_hi=m.conf_int().loc['expo', 1], p=m.pvalues['expo']))
O = pd.DataFrame(rows); O.to_csv(f'{OUT}/v_decompose.csv', index=False)
pd.set_option('display.width', 250); print(O.round(4).to_string())
c = D[D.grp == 'case_chain']
print(pd.crosstab(c.cclass, c.pdb.isin(has_clean)))
