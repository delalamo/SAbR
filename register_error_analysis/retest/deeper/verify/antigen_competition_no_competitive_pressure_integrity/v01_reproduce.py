# Verify "no_competitive_pressure": rebuild the case/control set from entries.csv independently, re-join the
# investigator's per-entry competition table, reproduce the headline ORs, and run robustness variants.
import warnings, collections
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_integrity'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)

e = pd.read_csv(f'{SP}/deeper/entries.csv')
A = pd.read_csv(f'{INV}/tables/pdbwide_ab_entries_antigen.csv.gz')
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb)
assert not A.pdb.duplicated().any() and not e.pdb.duplicated().any()
m = e.merge(A.drop(columns=['date']), on='pdb', how='left', validate='one_to_one')
m = m[~m.pdb.isin(amy) & (m.err_first | ~m.err_any)].copy()
m['y'] = m.err_first.astype(int)
m['dep'] = pd.to_datetime(m.deposit); m['dyear'] = m.dep.dt.year
m['ycat'] = m.dyear.clip(lower=2017).astype(str)
m['yfull'] = m.dyear.astype(str)
m['yq'] = m.dep.dt.to_period('Q').astype(str)
m['ydec'] = m.dyear + (m.dep.dt.dayofyear - 1) / 365.25
m['period'] = pd.cut(m.dyear, [0, 2019, 2021, 2100], labels=['<=2019', '2020-21', '2022+']).astype(str)
m['paper'] = m.doi.fillna(m.last_author.fillna('') + '_' + m.dyear.astype(str))
m['sars2'] = m.sars2.astype(bool)
for c in ['labs_acc_1m', 'labs_acc_6m', 'labs_acc_12m', 'labs_fam_12m', 'labs_fam_6m', 'prev12_allpdb']:
    m['l_' + c] = np.log2(1 + m[c])
m['race_1m'] = (m.labs_acc_1m > 0).astype(float).where(m.labs_acc_1m.notna())
m['any12'] = (m.labs_acc_12m > 0).astype(float).where(m.labs_acc_12m.notna())
m['ge10_12'] = (m.labs_acc_12m >= 10).astype(float).where(m.labs_acc_12m.notna())
m['ab_status2'] = m.ab_status.replace({'no contact': 'other', 'no interface data': 'other'})

# compare to the investigator's saved focal table
F = pd.read_csv(f'{INV}/tables/focal_entries_antigen_features.csv.gz')
assert set(F.pdb) == set(m.pdb), 'analysis set differs'
chk = F.set_index('pdb')[['y', 'labs_acc_12m']].join(m.set_index('pdb')[['y', 'labs_acc_12m']], rsuffix='_v')
print('rebuilt set identical to investigator focal set:', len(m), '| y mismatches', (chk.y != chk.y_v).sum(),
      '| labs12 mismatches', ((chk.labs_acc_12m != chk.labs_acc_12m_v) & chk.labs_acc_12m.notna()).sum())

EM = m[m.method == 'EM']; XR = m[m.method == 'X-ray']
def tgt(D): return D[D.ab_status2.isin(['bound', 'fiducial-only']) & D.labs_acc_12m.notna()]
def fit(D, term, extra='C(ycat) + res', cluster='paper'):
    mod = smf.glm(f'y ~ {term} + {extra}', D, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D[cluster])[0]})
    k = [x for x in mod.params.index if x.startswith(term)][0]
    ci = mod.conf_int().loc[k]
    return dict(OR=np.exp(mod.params[k]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=mod.pvalues[k], n=len(D), n_err=int(D.y.sum()), n_err_papers=D[D.y == 1].paper.nunique())

R = []
def add(label, D, term, **kw):
    try:
        r = fit(D, term, **kw)
    except Exception as ex:
        print('fail', label, term, ex); return
    r.update(label=label, term=term); R.append(r)

E1 = tgt(EM)
print('\nEM analysis set', len(E1), 'errors', E1.y.sum(), '| error papers', E1[E1.y == 1].paper.nunique(), '| clean papers', E1[E1.y == 0].paper.nunique(),
      '| error labs', E1[E1.y == 1].last_author.nunique())
print('labs_acc_12m error median/mean', E1[E1.y == 1].labs_acc_12m.median(), round(E1[E1.y == 1].labs_acc_12m.mean(), 2),
      'clean', E1[E1.y == 0].labs_acc_12m.median(), round(E1[E1.y == 0].labs_acc_12m.mean(), 2))
# --- headline reproduction
for t in ['l_labs_acc_12m', 'l_labs_acc_6m', 'race_1m', 'l_labs_fam_12m']:
    add('EM headline (year clip2017 + res)', E1, t)
add('EM non-SARS2 headline', E1[~E1.sars2], 'l_labs_fam_12m')
add('EM non-SARS2 headline', E1[~E1.sars2], 'l_labs_acc_12m')
add('X-ray headline (period+res)', tgt(XR), 'l_labs_acc_12m', extra='C(period) + res')
# --- robustness of the per-doubling OR (same accession, +-12 months)
t = 'l_labs_acc_12m'
add('EM, full year categories', E1, t, extra='C(yfull) + res')
add('EM, deposit quarter categories', E1, t, extra='C(yq) + res')
add('EM, cubic spline of deposit date + spline(res)', E1, t, extra='bs(ydec, df=6) + bs(res, df=4)')
add('EM, no adjustment', E1, t, extra='1')
add('EM, cluster by lab instead of paper', E1, t, cluster='last_author')
add('EM, bound only (drop fiducial-only)', E1[E1.ab_status2 == 'bound'], t)
add('EM, + SARS2 indicator', E1, t, extra='C(ycat) + res + sars2')
add('EM, 2020-2022 deposits only', E1[E1.dyear.between(2020, 2022)], t, extra='C(yfull) + res')
for b in ['any12', 'ge10_12']:
    add('EM binary exposure', E1, b)
    add('EM non-SARS2 binary exposure', E1[~E1.sars2], b)
# outcome including copied/artifact errors (err_any) -> check that excluding copied errors does not create the null
m2 = e.merge(A.drop(columns=['date']), on='pdb', how='left')
m2 = m2[~m2.pdb.isin(amy)].copy(); m2['y'] = m2.err_any.astype(int)
m2['dyear'] = pd.to_datetime(m2.deposit).dt.year; m2['ycat'] = m2.dyear.clip(lower=2017).astype(str)
m2['paper'] = m2.doi.fillna(m2.last_author.fillna('') + '_' + m2.dyear.astype(str)); m2['l_labs_acc_12m'] = np.log2(1 + m2.labs_acc_12m)
m2['ab_status2'] = m2.ab_status.replace({'no contact': 'other', 'no interface data': 'other'}); m2['sars2'] = m2.sars2.astype(bool)
E2 = tgt(m2[m2.method == 'EM'])
add('EM outcome = any run incl. copied/artifact', E2, t)
add('EM non-SARS2 outcome = any run incl. copied', E2[~E2.sars2], t)
# copied-only entries counted as errors (artifact-only remain excluded is not separable at entry level; use err_any)
print('\nEM entries excluded as copied/artifact-only:', int(((m2.method == 'EM') & m2.err_any & ~m2.err_first).sum()),
      '| of these SARS2:', int(((m2.method == 'EM') & m2.err_any & ~m2.err_first & m2.sars2).sum()))
# paper-level
P = E1.groupby('paper').agg(y=('y', 'max'), ycat=('ycat', lambda s: s.mode().iat[0]), res=('res', 'median'), n=('y', 'size'),
                            sars2=('sars2', 'max'), labs12=('labs_acc_12m', 'median'), lab=('last_author', 'first')).reset_index()
P['ln'] = np.log2(P.n); P['l_labs12'] = np.log2(1 + P.labs12)
add('EM paper level (+log n entries)', P, 'l_labs12', extra='C(ycat) + res + ln')
add('EM paper level non-SARS2', P[~P.sars2], 'l_labs12', extra='C(ycat) + res + ln')
Rt = pd.DataFrame(R)[['label', 'term', 'OR', 'lo', 'hi', 'p', 'n', 'n_err', 'n_err_papers']]
print(Rt.round(4).to_string())
Rt.to_csv(f'{OUT}/v01_reproduce_robustness.csv', index=False)

# --- leave-one-paper-out and leave-one-lab-out for the headline per-doubling OR
lo_p = []
for pap in E1[E1.y == 1].paper.unique():
    r = fit(E1[E1.paper != pap], t); lo_p.append((pap, r['OR'], r['p']))
lo_l = []
for lab in E1[E1.y == 1].last_author.dropna().unique():
    r = fit(E1[E1.last_author != lab], t); lo_l.append((lab, r['OR'], r['p']))
lp = pd.DataFrame(lo_p, columns=['paper', 'OR', 'p']); ll = pd.DataFrame(lo_l, columns=['lab', 'OR', 'p'])
print('\nleave-one-error-paper-out (n=%d): OR range %.3f-%.3f, p range %.3f-%.3f, min p paper %s' % (len(lp), lp.OR.min(), lp.OR.max(), lp.p.min(), lp.p.max(), lp.loc[lp.p.idxmin(), 'paper']))
print('leave-one-error-lab-out (n=%d): OR range %.3f-%.3f, p range %.3f-%.3f' % (len(ll), ll.OR.min(), ll.OR.max(), ll.p.min(), ll.p.max()))
# same for the "fewer" claim (family, non-SARS2)
Ens = E1[~E1.sars2]
lo_f = [(pap,) + tuple(fit(Ens[Ens.paper != pap], 'l_labs_fam_12m')[k] for k in ('OR', 'p')) for pap in Ens[Ens.y == 1].paper.unique()]
lf = pd.DataFrame(lo_f, columns=['paper', 'OR', 'p'])
print('family non-SARS2 leave-one-error-paper-out: OR range %.3f-%.3f, p max %.4f' % (lf.OR.min(), lf.OR.max(), lf.p.max()))
lo_fl = [(lab,) + tuple(fit(Ens[Ens.last_author != lab], 'l_labs_fam_12m')[k] for k in ('OR', 'p')) for lab in Ens[Ens.y == 1].last_author.dropna().unique()]
lfl = pd.DataFrame(lo_fl, columns=['lab', 'OR', 'p'])
print('family non-SARS2 leave-one-error-lab-out: OR range %.3f-%.3f, p max %.4f (lab %s)' % (lfl.OR.min(), lfl.OR.max(), lfl.p.max(), lfl.loc[lfl.p.idxmax(), 'lab']))
pd.concat([lp.assign(kind='acc12 EM drop paper'), ll.rename(columns={'lab': 'paper'}).assign(kind='acc12 EM drop lab'),
           lf.assign(kind='fam12 EM nonSARS2 drop paper'), lfl.rename(columns={'lab': 'paper'}).assign(kind='fam12 EM nonSARS2 drop lab')]).to_csv(f'{OUT}/v01_leave_one_out.csv', index=False)

# --- permutation p (within deposit-year x SARS2 strata, permuting exposure across papers) for the headline
def perm_p(D, col, nperm=2000, seed=7):
    rng = np.random.default_rng(seed)
    Pp = D.groupby('paper').agg(y=('y', 'sum'), n=('y', 'size'), x=(col, 'mean'), yr=('ycat', 'first'), s=('sars2', 'max')).reset_index()
    obs = np.corrcoef(Pp.x, Pp.y / Pp.n)[0, 1]
    # residualise rate on stratum means
    Pp['st'] = Pp.yr + '_' + Pp.s.astype(str)
    def stat(x):
        d = Pp.assign(x=x); d['xr'] = d.x - d.groupby('st').x.transform('mean'); d['yr_'] = d.y / d.n - d.groupby('st').apply(lambda g: (g.y.sum() / g.n.sum())).reindex(d.st).values
        return np.sum(d.xr * d.yr_ * d.n)
    o = stat(Pp.x.values); sims = []
    for _ in range(nperm):
        x = Pp.x.values.copy()
        for st, idx in Pp.groupby('st').indices.items(): x[idx] = rng.permutation(x[idx])
        sims.append(stat(x))
    sims = np.array(sims)
    return o, 2 * min((sims <= o).mean(), (sims >= o).mean())
o, pp = perm_p(E1, 'l_labs_acc_12m', nperm=1000)
print('\npermutation (papers within year x SARS2 strata) two-sided p for l_labs_acc_12m: %.3f (stat %.2f)' % (pp, o))

# --- binned error rates
D = E1.copy(); D['bin'] = pd.cut(D.labs_acc_12m, [-1, 0, 2, 9, 49, 1e5], labels=['0', '1-2', '3-9', '10-49', '>=50'])
print('\nEM error rate by other labs same accession +-12m:')
print(D.groupby('bin').y.agg(['mean', 'size', 'sum']).round(3).to_string())
print(D.groupby(['period', 'bin']).y.agg(['mean', 'size', 'sum']).round(3).to_string())
# distribution of the exposure: share of the variance explained by SARS2
print('\nfraction of EM analysis entries with labs_acc_12m>=50 that are SARS2:', round(D[D.labs_acc_12m >= 50].sars2.mean(), 3), 'n', (D.labs_acc_12m >= 50).sum())
