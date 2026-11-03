"""Re-test the within-entry V-minus-rest gap: reproduce, fix the empty-rest bug, drop amyloid fibrils, mask the
error sites (project run table), use only error-free V chains, cluster by paper/lab, leave-one-paper/lab-out."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUTC = f'{SP}/deeper/cache/verify_struct_unusual_rest'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_integrity'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
R = pd.read_pickle(f'{OUTC}/res2.pkl')
R['allowed'] = (R.rama == 'Allowed').astype(float)
R['nonfav'] = R.rama.isin(['Allowed', 'OUTLIER']).astype(float)
R['clash'] = R.clash.astype(float)
R['rotout'] = (R.rota == 'OUTLIER').astype(float)
MET = ['allowed', 'nonfav', 'clash', 'rotout', 'q']


def agg(df, tag):
    g = df.groupby('pdb')
    o = pd.DataFrame({f'{tag}_n': g.size()})
    for k in MET:
        o[f'{tag}_{k}'] = (100 if k != 'q' else 1) * g[k].mean()
    return o


rest = R[~R.isV]
V = R[R.isV]
VS = {'V_all': V, 'V_norun': V[~V.pinrun], 'V_m3': V[~V.pm3], 'V_m10': V[~V.pm10],
      'V_errfree_chains': V[~V.pchain_run], 'V_errchains_m10': V[V.pchain_run & ~V.pm10]}
E = pd.concat([agg(v, k) for k, v in VS.items()] + [agg(rest, 'rest'), agg(rest[~rest.is_vchain], 'restO'),
                                                    agg(rest[rest.is_vchain], 'restC')], axis=1)
S = pd.read_csv(f'{INV}/sample.csv', dtype={'chain': str})
meta = S.groupby('pdb').agg(role=('role_cc', 'first'), res=('res', 'first'), ybin=('ybin', 'first'), year=('year', 'first'),
                            paper=('paper', 'first'), lab=('last_author', 'first'), sars2=('sars2', 'max'))
ent = pd.read_csv(f'{SP}/deeper/entries.csv').set_index('pdb')
E = meta.join(E)
E['case'] = (E.role == 'case').astype(int)
E['amyloid'] = ent.loc[E.index, 'title'].str.contains('amyloid|fibril', case=False, na=False).values
E['sars2'] = E.sars2.astype(float)
E['paper'] = E.paper.fillna(pd.Series(E.index, index=E.index))
E['lab'] = E.lab.fillna(pd.Series(E.index, index=E.index))
E['rest_n'] = E.rest_n.fillna(0)
for v in VS:
    for k in MET:
        E[f'gap_{v}_{k}'] = E[f'{v}_{k}'] - E[f'rest_{k}']
# investigator's (buggy) version: rest metric = 0 when the entry has no non-V residues
for k in ['allowed', 'clash']:
    E[f'bug_gap_{k}'] = E[f'V_all_{k}'] - E[f'rest_{k}'].fillna(0)
E.to_csv(f'{OUT}/entry_metrics_verify2.csv')
print('amyloid entries by role', E.groupby('case').amyloid.sum().to_dict(), ' rest_n==0 by role', E[E.rest_n == 0].groupby('case').size().to_dict())

rows = []


def fit(y, df, label, cl='pdb', extra=''):
    x = df[[y, 'case', 'ybin', 'res', 'sars2', 'paper', 'lab', 'year']].copy()
    x['pdb'] = df.index
    x = x[np.isfinite(x[y])]
    f = f'{y} ~ case + C(ybin) + res' + extra
    m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[cl])[0]})
    a, b = x[x.case == 1][y], x[x.case == 0][y]
    o = dict(metric=y, set=label, cluster=cl, extra=extra, n_case=len(a), n_ctrl=len(b), case_mean=a.mean(), ctrl_mean=b.mean(),
             case_median=a.median(), ctrl_median=b.median(), effect=m.params['case'], lo=m.conf_int().loc['case', 0],
             hi=m.conf_int().loc['case', 1], p=m.pvalues['case'], p_mw=stats.mannwhitneyu(a, b).pvalue)
    rows.append(o)
    return o


A = E
F = E[(E.rest_n > 0)]
G = E[~E.amyloid & (E.rest_n > 0)]
# 1. reproduce
fit('bug_gap_allowed', A, 'investigator (rest=0 when empty), all 344')
fit('bug_gap_clash', A, 'investigator (rest=0 when empty), all 344')
# 2. fixes
for k in ['allowed', 'clash', 'nonfav', 'rotout', 'q']:
    fit(f'gap_V_all_{k}', F, 'rest empty -> NaN')
    fit(f'gap_V_all_{k}', G, 'no amyloid')
# 3. site masking (no amyloid)
for v in ['V_norun', 'V_m3', 'V_m10', 'V_errfree_chains', 'V_errchains_m10']:
    for k in ['allowed', 'clash', 'nonfav']:
        fit(f'gap_{v}_{k}', G, 'no amyloid')
# 4. components separately
for c in ['V_all', 'V_m10', 'V_errfree_chains', 'rest', 'restO', 'restC']:
    for k in ['allowed', 'clash', 'rotout', 'q']:
        fit(f'{c}_{k}', G, 'no amyloid')
    for k in ['allowed', 'clash']:
        fit(f'{c}_{k}', A[A.rest_n > 0] if c.startswith('rest') else A, 'with amyloid (NaN rest)')
# 5. clustering and covariates
for y in ['gap_V_all_allowed', 'gap_V_m10_allowed', 'gap_V_errfree_chains_allowed', 'gap_V_all_clash', 'gap_V_m10_clash',
          'gap_V_errfree_chains_clash']:
    fit(y, G, 'no amyloid', cl='paper'); fit(y, G, 'no amyloid', cl='lab')
    fit(y, G, 'no amyloid', extra=' + sars2')
    fit(y, G[G.year.between(2020, 2022)], 'no amyloid, 2020-22')
    fit(y, G[G.sars2 == 0], 'no amyloid, non-SARS-CoV-2')
T = pd.DataFrame(rows)
T['q_bh'] = multipletests(T.p, method='fdr_bh')[1]
T.to_csv(f'{OUT}/tests_verify.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(T[['metric', 'set', 'cluster', 'extra', 'n_case', 'n_ctrl', 'case_mean', 'ctrl_mean', 'case_median', 'ctrl_median', 'effect', 'lo', 'hi', 'p', 'p_mw']].round(4).to_string())

# 6. leave-one-paper-out / leave-one-lab-out (case papers/labs) on the no-amyloid set
lo = []
for y in ['gap_V_all_allowed', 'gap_V_m10_allowed', 'gap_V_errfree_chains_allowed', 'gap_V_all_clash', 'gap_V_m10_clash', 'gap_V_errfree_chains_clash']:
    for unit in ['paper', 'lab']:
        ps, effs = [], []
        for u in G[G.case == 1][unit].unique():
            x = G[G[unit] != u]
            x = x[np.isfinite(x[y])]
            m = smf.ols(f'{y} ~ case + C(ybin) + res', x).fit(cov_type='HC1')
            ps.append(m.pvalues['case']); effs.append(m.params['case'])
        lo.append(dict(metric=y, leave_out=unit, n_dropped_units=len(ps), max_p=max(ps), min_effect=min(effs), max_effect=max(effs)))
LO = pd.DataFrame(lo)
LO.to_csv(f'{OUT}/leave_one_out.csv', index=False)
print(LO.round(5).to_string())
