"""Where does the case-control excess come from?  Decompose by whether the case entry also contains error-free V chains
(within-entry comparators), and test the error-free V chains of case entries against control chains.
Uses v02 (strict E3 metrics of the sample units) and v04 (all V chains of case entries).
Also: robustness of the main effect to influential papers (leave-one-author-out, trimming top cases), and a
logistic model on 'any excess' (rank-free) to check that the mean is not driven by a few extreme chains.
Output tables/v05_*.csv
"""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_confounding'
M = pd.read_csv(f'{OUT}/tables/v02_unit_strict.csv', dtype={'chain': str})
W = pd.read_csv(f'{OUT}/tables/v04_within_entry_chains.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain', 'sars2', 'year', 'res', 'last_author', 'paper'])
ent = pd.read_csv(f'{SP}/deeper/entries.csv').set_index('pdb')
comp = W[W.err == 0].groupby('pdb').size()
nerr = W[W.err == 1].groupby('pdb').size()
M['has_comp'] = M.pdb.map(comp).fillna(0).gt(0).astype(int)
M['frac_err_chains'] = M.pdb.map(nerr) / M.pdb.map(ent.n_V_chains)
rows = []


def fit(x, y, lab, term='case', rhs='C(cclass) + C(ybin) + res'):
    x = x.dropna(subset=[y])
    m = smf.ols(f'{y} ~ {term} + {rhs}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.author_c)[0]})
    lo, hi = m.conf_int().loc[term]
    rows.append(dict(metric=y, model=lab, n_1=int(x[term].sum()), n_0=int((1 - x[term]).sum()),
                     mean_1=x[x[term] == 1][y].mean(), mean_0=x[x[term] == 0][y].mean(),
                     median_1=x[x[term] == 1][y].median(), median_0=x[x[term] == 0][y].median(),
                     effect=m.params[term], ci_lo=lo, ci_hi=hi, p=m.pvalues[term]))


ctrl = M[M.case == 0]
for y in ['allowed_E3', 'clash_E3']:
    fit(M, y, 'all cases vs controls (E3)')
    fit(pd.concat([M[(M.case == 1) & (M.has_comp == 1)], ctrl]), y, 'cases in entries that also hold error-free V chains')
    fit(pd.concat([M[(M.case == 1) & (M.has_comp == 0)], ctrl]), y, 'cases in entries where every V chain has an error')
    fit(M, y, 'all cases, adjusted for has_comp interaction', rhs='C(cclass) + C(ybin) + res + case:has_comp')
# error-free V chains of case entries vs control chains (entry-level property?)
meta = M[M.case == 1].drop_duplicates('pdb').set_index('pdb')[['ybin', 'res', 'author_c']]
E = W[W.comp.isin(['partner of error chain', 'different antibody', 'same antibody, error-free copy'])].join(meta, on='pdb')
E = E.rename(columns={'allowed': 'allowed_E3', 'clash': 'clash_E3'}); E['case'] = 1
CT = ctrl[['pdb', 'cclass', 'ybin', 'res', 'author_c', 'allowed_E3', 'clash_E3']].assign(case=0)
EC = pd.concat([E[['pdb', 'cclass', 'ybin', 'res', 'author_c', 'allowed_E3', 'clash_E3', 'case', 'comp']], CT])
for y in ['allowed_E3', 'clash_E3']:
    fit(EC, y, 'error-FREE V chains in case entries vs control chains')
    fit(EC[(EC.case == 0) | (EC.comp == 'different antibody')], y, 'error-free DIFFERENT antibody in case entries vs controls')
    fit(EC[(EC.case == 0) | (EC.comp == 'partner of error chain')], y, 'error-free PARTNER in case entries vs controls')
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/tables/v05_decomposition.csv', index=False)
pd.set_option('display.width', 250)
print(T.round(4).to_string())

# ---- influence: leave-one-author-out, drop top-k cases
rows2 = []
y = 'allowed_E3'
base = M.dropna(subset=[y])
effs = []
for a in base[base.case == 1].author_c.unique():
    x = base[base.author_c != a]
    m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.author_c)[0]})
    effs.append((a, m.params['case'], m.pvalues['case']))
E2 = pd.DataFrame(effs, columns=['dropped_author', 'effect', 'p'])
rows2.append(dict(check='leave-one-last-author-out', min_effect=E2.effect.min(), max_effect=E2.effect.max(), max_p=E2.p.max()))
for k in [10, 20, 40]:
    top = base[base.case == 1].nlargest(k, y).index
    x = base.drop(top)
    m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.author_c)[0]})
    rows2.append(dict(check=f'drop the {k} highest-valued cases', min_effect=m.params['case'], max_effect=m.params['case'], max_p=m.pvalues['case']))
# rank-free: fraction above control 90th percentile
thr = base[base.case == 0][y].quantile(0.9)
base = base.assign(hi=(base[y] > thr).astype(int))
m = smf.logit('hi ~ case + C(cclass) + C(ybin) + res', base).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(base.author_c)[0]})
rows2.append(dict(check=f'P(> control 90th pct = {thr:.1f}): case {base[base.case==1].hi.mean():.2f} vs ctrl {base[base.case==0].hi.mean():.2f}; OR',
                  min_effect=np.exp(m.params['case']), max_effect=np.exp(m.params['case']), max_p=m.pvalues['case']))
# quantile regression at the median
mq = smf.quantreg(f'{y} ~ case + C(cclass) + C(ybin) + res', base).fit(q=0.5)
rows2.append(dict(check='median (quantile) regression', min_effect=mq.params['case'], max_effect=mq.params['case'], max_p=mq.pvalues['case']))
I = pd.DataFrame(rows2)
I.to_csv(f'{OUT}/tables/v05_influence.csv', index=False)
print(I.round(4).to_string())
print(M[M.case == 1].groupby('has_comp').cclass.value_counts().unstack())
