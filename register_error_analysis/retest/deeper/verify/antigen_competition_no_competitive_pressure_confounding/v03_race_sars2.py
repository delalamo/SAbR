# Follow-up: (a) strata that failed with full adjustment (drop constant covariates), (b) the 'race' measures after adjustment,
# (c) categorical dose-response, (d) SARS-CoV-2 vs other antigens after adjusting for antibody/target features, SARS2 time course,
# earliest-25 check and the Omicron race window.
import warnings, re, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_confounding'
m = pd.read_csv(f'{OUT}/entries_frame.csv.gz')
m['l_labs_acc_1m'] = np.log2(1 + m.labs_acc_1m)
m['l_labs_acc_6m'] = np.log2(1 + m.labs_acc_6m)
R = []
AB = ['any_vhh', 'any_mouse', 'min_vid', 'any_atyp', 'any_lam']; MOD = ['af', 'expstart', 'rigid']


def fit(d, term, covs, fe='C(ycat)', y='y_first', cl='paper', label='', stratum=''):
    d = d.dropna(subset=[y, term, 'res'] + [c for c in covs if c != 'sars2']).copy()
    covs = [c for c in covs if d[c].nunique() > 1 and c != term]
    rhs = ' + '.join([fe, 'res'] + covs) if d['ycat'].nunique() > 1 or fe != 'C(ycat)' else ' + '.join(['res'] + covs)
    try:
        mod = smf.glm(f'{y} ~ {term} + {rhs}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cl])[0]})
        k = [x for x in mod.params.index if x.startswith(term)][0]; ci = np.exp(mod.conf_int().loc[k])
        r = dict(test=label, stratum=stratum, term=k, adj=rhs, OR=np.exp(mod.params[k]), lo=ci[0], hi=ci[1], p=mod.pvalues[k], n=len(d), n_err=int(d[y].sum()))
    except Exception as ex:
        r = dict(test=label, stratum=stratum, term=term, adj=rhs, OR=np.nan, p=np.nan, n=len(d), n_err=int(d[y].sum()), note=str(ex)[:60])
    R.append(r)
    print(f"{label:34s} {stratum:28s} {r['term']:22s} OR {r['OR']:.3f} ({r.get('lo', np.nan):.2f}-{r.get('hi', np.nan):.2f}) p={r['p']:.4f} n={r['n']} err={r['n_err']}")
    return r


base = m[m.keep_first & m.ab_status2.isin(['bound', 'fiducial-only']) & m.labs_acc_12m.notna()].copy()
EM = base[base.method == 'EM'].copy()
FULL = ['fidonly', 'capenz'] + AB + MOD

print('== A. Strata with full adjustment (constant covariates dropped)')
for sname, D in [('EM bound only', EM[EM.ab_status2 == 'bound']), ('EM non-SARS2 bound', EM[~EM.sars2 & (EM.ab_status2 == 'bound')]),
                 ('EM SARS2', EM[EM.sars2]), ('EM 2020-21', EM[EM.period == '2020-21']), ('EM non-SARS2 2020-21', EM[~EM.sars2 & (EM.period == '2020-21')]),
                 ('EM non-SARS2 non-capenz', EM[~EM.sars2 & (EM.capenz == 0)])]:
    for t in ['l_labs_acc_12m', 'l_labs_fam_12m']:
        fit(D, t, FULL, label='full adj', stratum=sname)

print('\n== B. Race measures (EM)')
for sname, D in [('EM', EM), ('EM non-SARS2', EM[~EM.sars2]), ('EM 2020-21', EM[EM.period == '2020-21']), ('EM 2022+', EM[EM.period == '2022+'])]:
    for t in ['race_1m', 'l_labs_acc_1m', 'l_labs_acc_6m']:
        fit(D, t, [], label='race, year+res', stratum=sname)
        fit(D, t, FULL, label='race, full adj', stratum=sname)
        fit(D, t, FULL, fe='C(half)', label='race, full adj half-year', stratum=sname)
print('\n race_1m cross-tab (EM non-SARS2):')
x = EM[~EM.sars2]; print(x.groupby(['period', 'race_1m']).y_first.agg(['mean', 'sum', 'size']).round(3).to_string())
print(x.groupby(['capenz', 'race_1m']).y_first.agg(['mean', 'sum', 'size']).round(3).to_string())

print('\n== C. Categorical dose-response (EM, full adj): labs same accession +-12m')
EM['labs_cat'] = pd.cut(EM.labs_acc_12m, [-1, 0, 2, 9, 49, 1e5], labels=['0', '1-2', '3-9', '10-49', '50+']).astype(str)
for sname, D in [('EM', EM), ('EM non-SARS2', EM[~EM.sars2])]:
    print(sname, D.groupby('labs_cat').y_first.agg(['mean', 'sum', 'size']).round(3).to_dict('index'))
    for fe, covs, lab in [('C(ycat)', [], 'year+res'), ('C(half)', FULL, 'full adj')]:
        d = D.dropna(subset=['res'] + covs).copy(); covs2 = [c for c in covs if d[c].nunique() > 1]
        mod = smf.glm(f"y_first ~ C(labs_cat, Treatment('0')) + {fe} + res" + ''.join(' + ' + c for c in covs2), d, family=sm.families.Binomial()).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
        for k in [k for k in mod.params.index if k.startswith('C(labs_cat')]:
            ci = np.exp(mod.conf_int().loc[k]); print(f'   {lab:9s} {k[-6:]:8s} OR {np.exp(mod.params[k]):.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={mod.pvalues[k]:.3f}')
            R.append(dict(test=f'dose-response {lab}', stratum=sname, term=k, OR=np.exp(mod.params[k]), lo=ci[0], hi=ci[1], p=mod.pvalues[k], n=len(d), n_err=int(d.y_first.sum())))

print('\n== D. SARS-CoV-2 vs other antigens (EM), sequential adjustment')
for fe, covs, lab in [('C(ycat)', [], 'year+res'), ('C(half)', [], 'half-year+res'), ('C(half)', AB, '+ antibody features'),
                      ('C(half)', AB + MOD, '+ ab + modelling'), ('C(half)', FULL, 'full'), ('C(half)', FULL + ['lqz_min'], 'full + lqz_min')]:
    fit(EM, 'sars2', covs, fe=fe, label=f'SARS2: {lab}', stratum='EM')
for per in ['2020-21', '2022+']:
    fit(EM[EM.period == per], 'sars2', [], label='SARS2: year+res', stratum=f'EM {per}')
    fit(EM[EM.period == per], 'sars2', FULL, fe='C(half)', label='SARS2: full', stratum=f'EM {per}')
# predicted vs observed for SARS2 entries from a model fitted on non-SARS2 entries (antibody + modelling + period)
d = EM.dropna(subset=['res'] + AB + MOD).copy()
mod = smf.glm('y_first ~ C(half) + res + ' + ' + '.join(AB + MOD + ['capenz', 'fidonly']), d[~d.sars2], family=sm.families.Binomial()).fit()
ds = d[d.sars2 & d.half.isin(d[~d.sars2].half.unique())].copy(); ds['p_hat'] = mod.predict(ds)
o, e_ = ds.y_first.sum(), ds.p_hat.sum(); v = (ds.p_hat * (1 - ds.p_hat)).sum(); z = (o - e_) / np.sqrt(v)
print(f'SARS2 EM entries observed {o} vs expected {e_:.1f} from non-SARS2 model (O/E {o/e_:.2f}, z={z:.2f}, p={2*stats.norm.sf(abs(z)):.3f}, n={len(ds)})')
R.append(dict(test='SARS2 observed/expected from non-SARS2 model (half-year, res, ab, modelling, class)', stratum='EM SARS2', term='O/E', OR=o / e_, p=2 * stats.norm.sf(abs(z)), n=len(ds), n_err=int(o)))
# antibody profile of SARS2 vs other
print('\n antibody profile, EM SARS2 vs non-SARS2:')
for c in AB + MOD + ['res', 'lqz_min', 'capenz', 'fidonly']:
    a, b = EM[EM.sars2][c].dropna(), EM[~EM.sars2][c].dropna()
    print(f'   {c:10s} {a.mean():.3f} vs {b.mean():.3f}  p={stats.mannwhitneyu(a, b).pvalue:.2g}')

print('\n== E. SARS2 time course (EM, investigator analysis set)')
S = EM[EM.sars2].sort_values('dep').copy()
S['q'] = pd.to_datetime(S.dep).dt.to_period('Q').astype(str)
N = EM[~EM.sars2].copy(); N['q'] = pd.to_datetime(N.dep).dt.to_period('Q').astype(str)
tq = pd.DataFrame({'sars2_rate': S.groupby('q').y_first.mean(), 'sars2_err': S.groupby('q').y_first.sum(), 'sars2_n': S.groupby('q').size(),
                   'other_rate': N.groupby('q').y_first.mean(), 'other_err': N.groupby('q').y_first.sum(), 'other_n': N.groupby('q').size()}).loc['2020Q1':'2024Q4']
print(tq.round(3).to_string())
tq.to_csv(f'{OUT}/v03_sars2_timecourse.csv')
print('\n earliest 25 SARS2 EM entries in analysis set:', S.head(25)[['pdb', 'deposit', 'y_first', 'last_author']].to_string(index=False)[:3000])
print(' errors among earliest 25:', S.head(25).y_first.sum(), '| earliest 50:', S.head(50).y_first.sum(), 'of', min(50, len(S)), '| date of 25th', S.head(25).deposit.iloc[-1])
# all SARS2 EM entries (incl. copied-only and all statuses) for the earliest check
allS = m[(m.method == 'EM') & m.sars2].sort_values('dep')
print(' earliest 25 SARS2 EM entries (all, incl. copied/any-run):', 'first-occ errors', allS.head(25).y_first.sum(), 'any-run', allS.head(25).y_any.sum(), 'last date', allS.head(25).deposit.iloc[-1])
# expected number of errors in the earliest 25 given their period/res and antibody features
s25 = ds[ds.pdb.isin(S.head(25).pdb)]
print(f' earliest 25 (with covariates n={len(s25)}): expected errors under non-SARS2 model {s25.p_hat.sum():.2f}; P(0 errors)={np.prod(1-s25.p_hat):.3f}')
R.append(dict(test='earliest 25 SARS2 EM entries all clean: P(0 errors) under non-SARS2 model', stratum='EM SARS2 earliest 25', term='P0', OR=np.nan, p=np.prod(1 - s25.p_hat), n=len(s25), n_err=0))
# 2020-21 rate around the race peak: deposited Sep 2020 - Jun 2021
print('\n== F. Omicron race window (SARS2 EM entries deposited 2021-12-01..2022-06-30 with Omicron/BA. in title)')
S['omicron'] = S.title.fillna('').str.contains(r'omicron|B\.1\.1\.529|\bBA\.\d', case=False, regex=True)
w = S[(S.dep >= '2021-12-01') & (S.dep <= '2022-06-30')]
print(' window entries', len(w), 'omicron-titled', w.omicron.sum(), '| rate omicron-titled', round(w[w.omicron].y_first.mean(), 3), w[w.omicron].y_first.sum(), '/', w.omicron.sum(),
      '| other SARS2 in window', round(w[~w.omicron].y_first.mean(), 3), w[~w.omicron].y_first.sum(), '/', (~w.omicron).sum())
nw = N[(N.dep >= '2021-12-01') & (N.dep <= '2022-06-30')]
print(' non-SARS2 EM same window', round(nw.y_first.mean(), 3), nw.y_first.sum(), '/', len(nw))
tab = [[w[w.omicron].y_first.sum(), w.omicron.sum() - w[w.omicron].y_first.sum()], [nw.y_first.sum(), len(nw) - nw.y_first.sum()]]
p_f = stats.fisher_exact(tab)[1]; print(' Fisher omicron-race vs non-SARS2 same window p=', round(p_f, 3))
R.append(dict(test='Omicron race window: Omicron-titled SARS2 EM vs non-SARS2 EM same window (Fisher)', stratum='EM 2021-12..2022-06', term='omicron',
              OR=((tab[0][0] + .5) * (tab[1][1] + .5)) / ((tab[0][1] + .5) * (tab[1][0] + .5)), p=p_f, n=int(w.omicron.sum() + len(nw)), n_err=int(tab[0][0] + tab[1][0])))
o2 = S[S.omicron]
print(' all Omicron-titled SARS2 EM entries:', len(o2), 'errors', o2.y_first.sum(), 'rate', round(o2.y_first.mean(), 3))
print(o2[o2.y_first == 1][['pdb', 'deposit', 'last_author', 'modes']].to_string(index=False))
pd.DataFrame(R).to_csv(f'{OUT}/v03_tests.csv', index=False)
