"""Per-deposition-year version of the era comparison (report Figure 9), 2024 conkit flags.

A: flag rate per deposition year for antibody V domains and for non-antibody domains in entries
   without antibodies (the same classes as B_rates_by_era_method.csv), with 95% bootstrap CIs over
   entries.
B: antibody V flags observed / expected, per year. Expected = sum of predicted probabilities from a
   logistic model fitted to all other domains: flag ~ C(year) + C(method)*C(rb) + log(length), i.e.
   what a same-year, same-method, same-resolution, same-length non-antibody-V domain would show.
   Exact Poisson 95% CIs on the observed count. Unlike a per-year odds ratio, O/E stays defined in
   years with no flagged antibody domain. The same calculation by era is printed for comparison with
   the era odds ratios (0.08, 0.35, 1.45).
"""
import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
from scipy.stats import chi2
warnings.filterwarnings('ignore')
A = '/tmp/analysis'; O = '/home/user/SAbR/register_error_analysis/followup/tables'
d = pd.read_pickle(f'{A}/ecod_grouped.pkl'); d = d[d.dlen > 0].copy()
d['cls'] = np.select([d.group == 'Antibody V', d.group == 'Antibody C', d.ab == 'other'], ['Antibody V', 'Antibody C', 'Non-antibody'], 'TCR/other')
ab_ent = set(d.pdb[d.cls == 'Antibody V'])
d['cls2'] = np.where(d.cls.isin(['Antibody V', 'Antibody C']), d.cls,
                     np.where(d.pdb.isin(ab_ent) & (d.cls == 'Non-antibody'), 'Non-antibody, in antibody entries',
                              np.where(d.cls == 'Non-antibody', 'Non-antibody, other entries', 'TCR')))
d['rb'] = d.res_bin.astype(str); d['logL'] = np.log(d.dlen.clip(lower=20)); d['yc'] = d.year.astype(int).astype(str)
rng = np.random.default_rng(0)
def rate_ci(s, B=500):
    g = s.groupby('pdb').flag.agg(['sum', 'count']); a, c = g['sum'].values, g['count'].values; n = len(g)
    b = [a[i].sum() / c[i].sum() for i in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * a.sum() / c.sum(), *np.percentile(b, [2.5, 97.5]) * 100
rows = []
for c in ['Antibody V', 'Non-antibody, other entries']:
    for y, s in d[(d.cls2 == c) & (d.year >= 2010)].groupby('year'):
        r, lo, hi = rate_ci(s)
        rows.append(dict(cls=c, year=int(y), domains=len(s), flagged=int(s.flag.sum()), pct_EM=100 * (s.method == 'EM').mean(), flag_pct=r, lo=lo, hi=hi))
A_t = pd.DataFrame(rows)
# expected from all domains other than antibody V
oth = d[d.cls != 'Antibody V']
m = smf.logit('flag ~ C(yc) + C(method)*C(rb) + logL', data=oth).fit(disp=0, maxiter=200)
v = d[d.cls == 'Antibody V'].copy()
v = v[v.yc.isin(set(oth.yc))]
v['p'] = m.predict(v)
def poisson_ci(k):
    lo = 0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2
rows = []
for y, s in v[v.year >= 2010].groupby('year'):
    o, e = int(s.flag.sum()), s.p.sum(); lo, hi = poisson_ci(o)
    rows.append(dict(year=int(y), domains=len(s), observed=o, expected=e, OE=o / e, lo=lo / e, hi=hi / e, pct_EM=100 * (s.method == 'EM').mean()))
B_t = pd.DataFrame(rows)
v['era'] = pd.cut(v.year, [0, 2017, 2019, 2023], labels=['≤2017', '2018–19', '2020–22']).astype(str)
E_t = v.groupby('era').apply(lambda s: pd.Series(dict(observed=s.flag.sum(), expected=s.p.sum()))).assign(OE=lambda x: x.observed / x.expected)
out = A_t.merge(B_t[['year', 'observed', 'expected', 'OE', 'lo', 'hi']].rename(columns={'lo': 'OE_lo', 'hi': 'OE_hi'}), on='year', how='left')
out.loc[out.cls != 'Antibody V', ['observed', 'expected', 'OE', 'OE_lo', 'OE_hi']] = np.nan
out.round(3).to_csv(f'{O}/B_by_year.csv', index=False)
pd.set_option('display.width', 200)
print(A_t.round(2).to_string(index=False)); print(B_t.round(3).to_string(index=False)); print('by era (O/E):'); print(E_t.round(3).to_string())
print('pre-2010 antibody V:', len(d[(d.cls == 'Antibody V') & (d.year < 2010)]), 'domains,', int(d[(d.cls == 'Antibody V') & (d.year < 2010)].flag.sum()), 'flagged')
