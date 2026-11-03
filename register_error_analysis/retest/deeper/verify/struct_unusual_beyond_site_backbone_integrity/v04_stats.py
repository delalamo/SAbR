"""Reproduce the adjusted test and stress it: alternative clustering, count models, rank models, one-unit-per-paper,
leave-one-paper/lab-out, lab fixed effects, block permutation, alternative site exclusions."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
U = pd.read_csv(f'{OUT}/v_unit_metrics.csv', dtype={'chain': str})
U['lab'] = U.last_author.fillna('NA')
U['paper'] = U.paper.fillna(U.pdb)
F = 'case + C(cclass) + C(ybin) + res'
rows = []


def ols(y, x, f=F, grp='pdb', label=''):
    x = x.dropna(subset=[y]).copy()
    m = smf.ols(f'{y} ~ {f}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[grp])[0]})
    r = dict(metric=y, model=label, cluster=grp, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
             effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case'])
    rows.append(r); return r


for y in ['allowed_out', 'clash_out', 'rota_out']:
    ols(y, U, label='reproduce (cluster entry)')
    ols(y, U, grp='paper', label='cluster paper')
    ols(y, U, grp='lab', label='cluster last author')
    ols(y, U, f=F + ' + sars2', grp='lab', label='+sars2, cluster last author')
for y in ['allowed_eval_out', 'nonfav_eval_out', 'allowed_outdis', 'allowed_out10', 'clash_out10', 'ramaout_out']:
    ols(y, U, grp='lab', label='alt definition, cluster last author')

# count model: Poisson with offset, cluster lab
for y, k in [('allowed_out', 'k_allowed_out'), ('clash_out', 'k_clash_out')]:
    x = U.dropna(subset=[y]).copy()
    m = smf.glm(f'{k} ~ {F}', x, family=sm.families.Poisson(), offset=np.log(x.n_out)).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(x.lab)[0]})
    rows.append(dict(metric=y, model='Poisson rate ratio (offset n), cluster lab', n_case=int(x.case.sum()),
                     n_ctrl=int((1 - x.case).sum()), effect=np.exp(m.params['case']),
                     ci_lo=np.exp(m.conf_int().loc['case', 0]), ci_hi=np.exp(m.conf_int().loc['case', 1]), p=m.pvalues['case']))
    # rank-based: OLS on within-dataset rank
    x['rk'] = x[y].rank(pct=True)
    r = ols('rk', x, grp='lab', label=f'percentile rank of {y}, cluster lab'); r['metric'] = y
    # median regression
    q = smf.quantreg(f'{y} ~ {F}', x).fit(q=0.5)
    rows.append(dict(metric=y, model='median regression (iid SE)', n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                     effect=q.params['case'], ci_lo=q.conf_int().loc['case', 0], ci_hi=q.conf_int().loc['case', 1], p=q.pvalues['case']))

# one case unit per paper (first in sample order) and controls one per entry
c1 = U[U.case == 1].dropna(subset=['allowed_out']).groupby('paper').head(1)
k1 = U[U.case == 0].dropna(subset=['allowed_out']).groupby('pdb').head(1)
for y in ['allowed_out', 'clash_out']:
    ols(y, pd.concat([c1, k1]), grp='lab', label='one case per paper, one control per entry, cluster lab')
# paper-level aggregation
P = U.dropna(subset=['allowed_out']).groupby(['paper', 'case']).agg(allowed_out=('allowed_out', 'mean'),
     clash_out=('clash_out', 'mean'), res=('res', 'mean'), cclass=('cclass', 'first'), ybin=('ybin', 'first'),
     lab=('lab', 'first')).reset_index()
for y in ['allowed_out', 'clash_out']:
    ols(y, P, grp='lab', label='paper-level means (paper x role), cluster lab')
    a, b = P[P.case == 1][y], P[P.case == 0][y]
    rows.append(dict(metric=y, model='paper-level Mann-Whitney (unadjusted)', n_case=len(a), n_ctrl=len(b),
                     effect=a.median() - b.median(), p=stats.mannwhitneyu(a, b).pvalue))

# lab fixed effects among labs with both cases and controls
both = U.groupby('lab').case.agg(['min', 'max']); both = both[(both['min'] == 0) & (both['max'] == 1)].index
L = U[U.lab.isin(both)]
for y in ['allowed_out', 'clash_out']:
    ols(y, L, f=F + ' + C(lab)', grp='lab', label=f'lab fixed effects, labs with both roles (n labs={len(both)})')
    ols(y, L, f=F, grp='lab', label=f'same labs, no lab FE (n labs={len(both)})')

# leave-one-paper-out / leave-one-lab-out (cases removed by group)
for y in ['allowed_out', 'clash_out']:
    for grp in ['paper', 'lab']:
        effs, ps = [], []
        x0 = U.dropna(subset=[y])
        groups = x0[x0.case == 1][grp].unique()
        for gname in groups:
            x = x0[x0[grp] != gname]
            m = smf.ols(f'{y} ~ {F}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
            effs.append(m.params['case']); ps.append(m.pvalues['case'])
        i = int(np.argmax(ps))
        rows.append(dict(metric=y, model=f'leave-one-{grp}-out (n={len(groups)}): min effect / max p',
                         effect=min(effs), p=max(ps), note=f'worst group: {groups[i]}'))
    # drop the top 10 case labs entirely
    x0 = U.dropna(subset=[y])
    top = x0[x0.case == 1].lab.value_counts().head(10).index
    ols(y, x0[~x0.lab.isin(top)], grp='lab', label='drop top-10 case labs (both roles)')

# block permutation: shuffle case/control labels across ENTRIES within cclass x ybin strata
rng = np.random.default_rng(1)
for y in ['allowed_out', 'clash_out']:
    x = U.dropna(subset=[y]).copy()
    def eff(df):
        return smf.ols(f'{y} ~ {F}', df).fit().params['case']
    e0 = eff(x)
    E = x.groupby('pdb').agg(case=('case', 'first'), st=('ybin', 'first'))
    null = []
    for it in range(2000):
        lab = E.groupby('st').case.transform(lambda v: rng.permutation(v.values))
        x['case'] = x.pdb.map(lab)
        null.append(eff(x))
    null = np.array(null)
    rows.append(dict(metric=y, model='entry-level permutation within year bins (2000 perms)', effect=e0,
                     p=(1 + (np.abs(null) >= abs(e0)).sum()) / 2001, note=f'null sd {null.std():.2f}'))

T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/v_stats.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(T.round(6).to_string())
