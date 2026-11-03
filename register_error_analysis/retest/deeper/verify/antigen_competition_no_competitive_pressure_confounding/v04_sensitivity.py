# Sensitivity of the positives that appear after adjustment: pre-treatment-only confounders, SARS2 vs 'ordinary' antigens,
# local map (clean entries only, to avoid definitional overlap), paper/lab level, leave-one-lab-out, 2020Q3 cluster; then BH over all my tests.
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_confounding'
m = pd.read_csv(f'{OUT}/entries_frame.csv.gz')
m['l_labs_acc_1m'] = np.log2(1 + m.labs_acc_1m)
R = []
AB = ['any_vhh', 'any_mouse', 'min_vid', 'any_atyp', 'any_lam']; MOD = ['af', 'expstart', 'rigid']; PRE = ['fidonly', 'capenz'] + AB


def fit(d, term, covs, fe='C(ycat)', y='y_first', cl='paper', label='', stratum='', quiet=False):
    d = d.dropna(subset=[y, term, 'res'] + covs).copy()
    covs = [c for c in covs if d[c].nunique() > 1 and c != term]
    rhs = ' + '.join([fe, 'res'] + covs)
    mod = smf.glm(f'{y} ~ {term} + {rhs}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cl])[0]})
    k = [x for x in mod.params.index if x.startswith(term)][0]; ci = np.exp(mod.conf_int().loc[k])
    r = dict(test=label, stratum=stratum, term=k, adj=rhs, cluster=cl, OR=np.exp(mod.params[k]), lo=ci[0], hi=ci[1], p=mod.pvalues[k], n=len(d), n_err=int(d[y].sum()))
    if not quiet:
        R.append(r); print(f"{label:42s} {stratum:24s} {k:16s} OR {r['OR']:.3f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.4f} n={r['n']} err={r['n_err']}")
    return r


base = m[m.keep_first & m.ab_status2.isin(['bound', 'fiducial-only']) & m.labs_acc_12m.notna()].copy()
EM = base[base.method == 'EM'].copy()

print('== A. Pre-treatment confounders only (class + fiducial + antibody features; no modelling, no map)')
for t in ['l_labs_acc_12m', 'l_labs_fam_12m', 'race_1m', 'l_labs_acc_1m', 'sars2']:
    fit(EM, t, PRE, label='pre-treatment adj', stratum='EM')
    fit(EM, t, PRE, cl='lab2', label='pre-treatment adj, lab-clustered', stratum='EM')
for t in ['race_1m', 'l_labs_acc_1m']:
    fit(EM, t, PRE + ['sars2'], label='pre-treatment adj + sars2', stratum='EM')

print('\n== B. SARS2 vs ordinary antigens (EM bound, non-capsid/enzyme, non-fiducial)')
O = EM[(EM.capenz == 0) & (EM.fidonly == 0)]
print(' crude: SARS2', round(O[O.sars2].y_first.mean(), 4), O[O.sars2].y_first.sum(), '/', O.sars2.sum(), '| other', round(O[~O.sars2].y_first.mean(), 4), O[~O.sars2].y_first.sum(), '/', (~O.sars2).sum())
for per in ['2020-21', '2022+']:
    x = O[O.period == per]; print(f'  {per}: SARS2 {x[x.sars2].y_first.mean():.3f} ({x[x.sars2].y_first.sum()}/{x.sars2.sum()}) vs other {x[~x.sars2].y_first.mean():.3f} ({x[~x.sars2].y_first.sum()}/{(~x.sars2).sum()})')
fit(O, 'sars2', [], label='SARS2 vs ordinary, year+res', stratum='EM ordinary')
fit(O, 'sars2', [], fe='C(half)', label='SARS2 vs ordinary, half-year+res', stratum='EM ordinary')
fit(O, 'sars2', AB, fe='C(half)', label='SARS2 vs ordinary, + ab features', stratum='EM ordinary')
fit(O, 'sars2', AB, fe='C(half)', cl='lab2', label='SARS2 vs ordinary, + ab, lab-clust', stratum='EM ordinary')
fit(O, 'sars2', AB + ['lqz_min'], fe='C(half)', label='SARS2 vs ordinary, + ab + lqz_min', stratum='EM ordinary')
fit(O[O.res <= 3.5], 'sars2', AB, fe='C(half)', label='SARS2 vs ordinary, + ab, res<=3.5', stratum='EM ordinary')
fit(O[O.res > 3.5], 'sars2', AB, fe='C(half)', label='SARS2 vs ordinary, + ab, res>3.5', stratum='EM ordinary')
# viral glycoproteins only (HIV Env, flu HA, RSV F ... vs SARS2 spike)
V = EM[EM.tgt_cls == 'viral surface glycoprotein']
fit(V, 'sars2', [], label='SARS2 vs other viral glycoproteins', stratum='EM viral glycoprotein')
fit(V, 'sars2', AB, fe='C(half)', label='SARS2 vs other viral glycoproteins, + ab', stratum='EM viral glycoprotein')
# paper level
P = O.groupby('paper').agg(y_first=('y_first', 'max'), ycat=('ycat', lambda s: s.mode().iat[0]), res=('res', 'median'), n=('y_first', 'size'), sars2=('sars2', 'max'),
                           any_vhh=('any_vhh', 'max'), any_mouse=('any_mouse', 'max'), min_vid=('min_vid', 'min'), any_atyp=('any_atyp', 'max'), any_lam=('any_lam', 'max'),
                           lab2=('lab2', 'first')).reset_index(); P['ln'] = np.log2(P.n)
print(' paper level crude: SARS2 papers', round(P[P.sars2].y_first.mean(), 3), P[P.sars2].y_first.sum(), '/', P.sars2.sum(), '| other', round(P[~P.sars2].y_first.mean(), 3), P[~P.sars2].y_first.sum(), '/', (~P.sars2).sum())
fit(P, 'sars2', ['ln'], cl='lab2', label='SARS2 vs ordinary, paper level', stratum='EM ordinary papers')
fit(P, 'sars2', ['ln'] + AB, cl='lab2', label='SARS2 vs ordinary, paper level + ab', stratum='EM ordinary papers')

print('\n== C. Local map fit: SARS2 vs other among CLEAN entries only (avoids error->poor fit)')
cl_ = EM[EM.y_first == 0]
for c in ['lqz_min', 'lqz_mean']:
    a, b = cl_[cl_.sars2][c].dropna(), cl_[~cl_.sars2][c].dropna()
    print(f'  clean entries {c}: SARS2 median {a.median():.2f} (n={len(a)}) vs other {b.median():.2f} (n={len(b)}), MWU p={stats.mannwhitneyu(a, b).pvalue:.2g}')
# does map fit modify the SARS2 excess? stratify by lqz_min tertile defined on clean entries
q = cl_.lqz_min.quantile([1/3, 2/3]).values
O = O.copy(); O['lqz_t'] = pd.cut(O.lqz_min, [-99, q[0], q[1], 99], labels=['low', 'mid', 'high']).astype(str)
print(O.groupby(['lqz_t', 'sars2']).y_first.agg(['mean', 'sum', 'size']).round(3).to_string())

print('\n== D. Leave-one-lab-out for adjusted positives (pre-treatment adj)')
for t, D, cov in [('l_labs_acc_1m', EM, PRE), ('race_1m', EM, PRE), ('sars2', O, AB)]:
    full = fit(D, t, cov, fe='C(half)' if t == 'sars2' else 'C(ycat)', quiet=True)
    labs = D[D.y_first == 1].lab2.value_counts().index[:15]
    res_ = []
    for lb in labs:
        r = fit(D[D.lab2 != lb], t, cov, fe='C(half)' if t == 'sars2' else 'C(ycat)', quiet=True); res_.append((lb, r['OR'], r['p']))
    ors = [x[1] for x in res_]; ps = [x[2] for x in res_]
    worst = max(res_, key=lambda x: x[2])
    print(f'  {t}: full OR {full["OR"]:.2f} p={full["p"]:.3f}; dropping each of top-15 error labs: OR {min(ors):.2f}-{max(ors):.2f}, p {min(ps):.3f}-{max(ps):.3f}; largest p when dropping {worst[0]} (OR {worst[1]:.2f}, p={worst[2]:.3f})')
    R.append(dict(test=f'leave-one-lab-out max p ({t})', stratum='EM', term=t, OR=worst[1], p=worst[2], n=len(D), n_err=int(D.y_first.sum())))

print('\n== E. 2020Q3 SARS2 spike in errors: which papers/labs?')
S = EM[EM.sars2].sort_values('dep')
q3 = S[(S.dep >= '2020-07-01') & (S.dep < '2020-10-01')]
print(q3.groupby(['lab2', 'paper']).y_first.agg(['sum', 'size']).sort_values('sum', ascending=False).head(10).to_string())
print(' entries 26-50 by lab:', S.iloc[25:50].groupby('lab2').y_first.agg(['sum', 'size']).sort_values('sum', ascending=False).head(6).to_dict('index'))
n3 = EM[~EM.sars2 & (EM.dep >= '2020-07-01') & (EM.dep < '2020-10-01')]
print(' non-SARS2 2020Q3 errors by lab:', n3.groupby('lab2').y_first.agg(['sum', 'size']).sort_values('sum', ascending=False).head(6).to_dict('index'))
# SARS2 paper-level rates by half-year
SP_ = S.groupby('paper').agg(y=('y_first', 'max'), half=('half', 'first')).groupby('half').y.agg(['mean', 'sum', 'size'])
print(' SARS2 EM papers by half-year:'); print(SP_.round(3).to_string())

# ---------------- BH over all my tests (v02, v03, v04)
T = pd.concat([pd.read_csv(f'{OUT}/v02_tests.csv'), pd.read_csv(f'{OUT}/v03_tests.csv'), pd.DataFrame(R)], ignore_index=True)
T = T[T.p.notna()].copy(); T['q'] = multipletests(T.p, method='fdr_bh')[1]
T.to_csv(f'{OUT}/all_tests_bh.csv', index=False)
print('\n== BH over', len(T), 'tests; q<0.1:')
print(T[T.q < 0.1].sort_values('p')[['test', 'stratum', 'term', 'OR', 'lo', 'hi', 'p', 'q', 'n', 'n_err']].round(4).to_string())
key = T[T.test.str.contains('race, full adj$|pre-treatment adj$|SARS2 vs ordinary, \\+ ab features|SARS2: full$|observed/expected|reproduce|adj: full \\(', regex=True)]
print('\n key tests:'); print(key[['test', 'stratum', 'term', 'OR', 'p', 'q']].round(4).to_string())
