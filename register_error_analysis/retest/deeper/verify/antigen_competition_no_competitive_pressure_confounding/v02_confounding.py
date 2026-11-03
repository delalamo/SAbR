# Re-analysis of "no competitive pressure": reproduce, then adjust/stratify for period, resolution, fiducial/target class,
# antibody features, modelling, local map, SARS-CoV-2, lab/paper clustering, unit level and outcome definition.
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_confounding'
m = pd.read_csv(f'{OUT}/entries_frame.csv.gz')
U = pd.read_csv(f'{OUT}/units_frame.csv.gz')
R = []


def fit(d, term, rhs, y='y_first', cl='paper', label='', stratum=''):
    cols = [y, term] + [c for c in ['res', 'ycat', 'half', 'tnum', 'any_vhh', 'any_mouse', 'min_vid', 'any_atyp', 'any_lam', 'af', 'expstart',
                                    'rigid', 'lqz_min', 'capenz', 'fidonly', 'resbin', 'sars2', 'vhh', 'mouse', 'vid', 'atyp', 'lqz', 'ln']
                        if c in rhs and c in d.columns]
    d = d.dropna(subset=list(dict.fromkeys(cols))).copy()
    try:
        mod = smf.glm(f'{y} ~ {term} + {rhs}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cl])[0]})
        k = [x for x in mod.params.index if x.startswith(term)][0]
        ci = np.exp(mod.conf_int().loc[k])
        r = dict(test=label, stratum=stratum, term=term, adj=rhs, cluster=cl, OR=np.exp(mod.params[k]), lo=ci[0], hi=ci[1], p=mod.pvalues[k], n=len(d), n_err=int(d[y].sum()))
    except Exception as ex:
        r = dict(test=label, stratum=stratum, term=term, adj=rhs, cluster=cl, OR=np.nan, p=np.nan, n=len(d), n_err=int(d[y].sum()), note=str(ex)[:80])
    R.append(r)
    print(f"{label:55s} {stratum:22s} {term:16s} OR {r['OR']:.3f} ({r.get('lo', np.nan):.2f}-{r.get('hi', np.nan):.2f}) p={r['p']:.4f} n={r['n']} err={r['n_err']}  | {rhs}")
    return r


base = m[m.keep_first & m.ab_status2.isin(['bound', 'fiducial-only']) & m.labs_acc_12m.notna()].copy()
EM = base[base.method == 'EM'].copy(); XR = base[base.method == 'X-ray'].copy()
print('EM analysis set', len(EM), EM.y_first.sum(), '| X-ray', len(XR), XR.y_first.sum())

print('\n== 1. Reproduce investigator (EM, year + res, paper-clustered)')
fit(EM, 'l_labs_acc_12m', 'C(ycat) + res', label='reproduce', stratum='EM')
fit(EM, 'l_labs_fam_12m', 'C(ycat) + res', label='reproduce', stratum='EM')
fit(EM, 'race_1m', 'C(ycat) + res', label='reproduce', stratum='EM')

print('\n== 2. Sequential adjustment, EM, labs same accession +-12 m (per doubling)')
AB = 'any_vhh + any_mouse + min_vid + any_atyp + any_lam'
MOD = 'af + expstart + rigid'
specs = [('half-year FE', 'C(half) + res'),
         ('date spline', 'bs(tnum, df=6) + res'),
         ('year + resolution bins', 'C(ycat) + C(resbin)'),
         ('+ fiducial-only', 'C(ycat) + res + fidonly'),
         ('+ capsid/enzyme', 'C(ycat) + res + capenz'),
         ('+ antibody features', f'C(ycat) + res + {AB}'),
         ('+ modelling', f'C(ycat) + res + {MOD}'),
         ('+ local map fit (lqz_min)', 'C(ycat) + res + lqz_min'),
         ('+ SARS2 indicator', 'C(ycat) + res + sars2'),
         ('full (half-year, res, fid, capenz, ab, modelling)', f'C(half) + res + fidonly + capenz + {AB} + {MOD}'),
         ('full + lqz_min + sars2', f'C(half) + res + fidonly + capenz + {AB} + {MOD} + lqz_min + sars2')]
for term in ['l_labs_acc_12m', 'l_labs_fam_12m', 'race_1m']:
    for lab, rhs in specs:
        fit(EM, term, rhs, label=f'adj: {lab}', stratum='EM')
    print()

print('\n== 3. Clustering by lab (last author) instead of paper; lab fixed effects (conditional logit)')
for term in ['l_labs_acc_12m', 'l_labs_fam_12m']:
    fit(EM, term, 'C(ycat) + res', cl='lab2', label='lab-clustered', stratum='EM')
    fit(EM, term, f'C(half) + res + fidonly + capenz + {AB} + {MOD}', cl='lab2', label='lab-clustered full', stratum='EM')
    d = EM.dropna(subset=['res', 'min_vid', term]).copy()
    d = d[d.groupby('lab2').y_first.transform(lambda s: s.nunique() > 1)]
    X = pd.get_dummies(d[['res', 'ycat']], columns=['ycat'], drop_first=True).astype(float)
    X[term] = d[term]
    try:
        cm = ConditionalLogit(d.y_first.values, X.values, groups=d.lab2.values).fit(disp=0)
        j = list(X.columns).index(term); b = cm.params[j]; se = cm.bse[j]; p = 2 * stats.norm.sf(abs(b / se))
        R.append(dict(test='within-lab conditional logit', stratum='EM labs with both outcomes', term=term, adj='lab FE + year + res', OR=np.exp(b),
                      lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=p, n=len(d), n_err=int(d.y_first.sum())))
        print(f'within-lab conditional logit {term}: OR {np.exp(b):.3f} ({np.exp(b-1.96*se):.2f}-{np.exp(b+1.96*se):.2f}) p={p:.4f} n={len(d)} labs={d.lab2.nunique()} err={d.y_first.sum()}')
    except Exception as ex:
        print('clogit failed', ex)

print('\n== 4. Strata (EM): SARS2, period, resolution, fiducial removed')
for sname, D in [('EM bound only', EM[EM.ab_status2 == 'bound']), ('EM non-SARS2', EM[~EM.sars2]), ('EM non-SARS2 bound only', EM[~EM.sars2 & (EM.ab_status2 == 'bound')]),
                 ('EM SARS2', EM[EM.sars2]), ('EM non-SARS2 2020-21', EM[~EM.sars2 & (EM.period == '2020-21')]), ('EM non-SARS2 2022+', EM[~EM.sars2 & (EM.period == '2022+')]),
                 ('EM 2020-21', EM[EM.period == '2020-21']), ('EM 2022+', EM[EM.period == '2022+']),
                 ('EM res<=3.3', EM[EM.res <= 3.3]), ('EM res 3.3-3.6', EM[(EM.res > 3.3) & (EM.res <= 3.6)]), ('EM res>3.6', EM[EM.res > 3.6]),
                 ('EM non-SARS2 non-capsid/enzyme', EM[~EM.sars2 & (EM.capenz == 0)])]:
    rhs = 'C(ycat) + res' if D.ycat.nunique() > 1 else 'res'
    fit(D, 'l_labs_acc_12m', rhs, label='stratum', stratum=sname)
    fit(D, 'l_labs_acc_12m', rhs + f' + capenz + fidonly + {AB} + {MOD}', label='stratum, full adj', stratum=sname)
    fit(D, 'l_labs_fam_12m', rhs, label='stratum', stratum=sname)
    fit(D, 'l_labs_fam_12m', rhs + f' + capenz + fidonly + {AB} + {MOD}', label='stratum, full adj', stratum=sname)

print('\n== 5. Paper level (EM; one row per paper; error paper = any first-occurrence error)')
P = EM.groupby('paper').agg(y_first=('y_first', 'max'), ycat=('ycat', lambda s: s.mode().iat[0]), res=('res', 'median'), n=('y_first', 'size'),
                            sars2=('sars2', 'max'), l_labs_acc_12m=('l_labs_acc_12m', 'median'), l_labs_fam_12m=('l_labs_fam_12m', 'median'),
                            capenz=('capenz', 'max'), fidonly=('fidonly', 'min'), any_vhh=('any_vhh', 'max'), any_mouse=('any_mouse', 'max'),
                            min_vid=('min_vid', 'min'), any_atyp=('any_atyp', 'max'), any_lam=('any_lam', 'max'), af=('af', 'max'),
                            expstart=('expstart', 'max'), rigid=('rigid', 'max'), lab2=('lab2', 'first')).reset_index()
P['ln'] = np.log2(P.n)
for sname, D in [('EM papers', P), ('EM papers non-SARS2', P[~P.sars2])]:
    for term in ['l_labs_acc_12m', 'l_labs_fam_12m']:
        fit(D, term, 'C(ycat) + res + ln', cl='lab2', label='paper level', stratum=sname)
        fit(D, term, f'C(ycat) + res + ln + capenz + fidonly + {AB} + {MOD}', cl='lab2', label='paper level, full adj', stratum=sname)

print('\n== 6. Unit level (antibody-in-paper x chain type), EM, paper-clustered')
UU = U[(U.err | ~U.anyrun) & (U.method == 'EM') & U.status.isin(['bound', 'fiducial-only']) & U.labs12.notna()].copy()
UU['y_first'] = UU.err.astype(int)
UU['l_labs_acc_12m'] = UU.l_labs12
UU['l_labs_fam_12m'] = UU.l_fam12
for sname, D in [('EM units all chains', UU), ('EM units H', UU[UU.ctype == 'H']), ('EM units K', UU[UU.ctype == 'K']), ('EM units L', UU[UU.ctype == 'L']),
                 ('EM units non-SARS2', UU[~UU.sars2])]:
    fit(D, 'l_labs_acc_12m', 'C(ycat) + res', label='unit level', stratum=sname)
    fit(D, 'l_labs_acc_12m', 'C(ycat) + res + capenz + fidonly + vhh + mouse + vid + atyp + af + rigid', label='unit level, full adj', stratum=sname)
    fit(D, 'l_labs_fam_12m', 'C(ycat) + res + capenz + fidonly + vhh + mouse + vid + atyp + af + rigid', label='unit level, full adj', stratum=sname)

print('\n== 7. Outcome definition: any run (incl. copied/artifact) vs none, all entries kept')
ALL = m[m.ab_status2.isin(['bound', 'fiducial-only']) & m.labs_acc_12m.notna() & (m.method == 'EM')].copy()
print('EM entries all', len(ALL), 'any-run', ALL.y_any.sum(), 'copied/artifact-only', (ALL.err_any & ~ALL.err_first).sum())
ALL['copied_only'] = (ALL.err_any & ~ALL.err_first).astype(int)
for term in ['l_labs_acc_12m', 'l_labs_fam_12m']:
    fit(ALL, term, 'C(ycat) + res', y='y_any', label='outcome = any run', stratum='EM all entries')
    fit(ALL, term, f'C(ycat) + res + capenz + fidonly + {AB} + {MOD}', y='y_any', label='outcome = any run, full adj', stratum='EM all entries')
    fit(ALL[~ALL.err_first], term, 'C(ycat) + res', y='copied_only', label='copied/artifact-only among non-first', stratum='EM')
# first appearance of each unique antibody V sequence (removes re-deposit inflation of clean controls)
print('\n== 8. First appearance of each unique antibody (vseq), EM, chain-level outcome = first-occurrence error')
ch = pd.read_csv('/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/chains.csv.gz',
                 usecols=['pdb', 'chain', 'ctype', 'deposit', 'vseq', 'err_first', 'err_any', 'vhh', 'species', 'v_id', 'L1_atyp', 'L2_atyp', 'L3_atyp', 'irm', 'fit', 'method', 'res'])
ch['dep'] = pd.to_datetime(ch.deposit)
fa = ch.sort_values(['dep', 'pdb']).groupby('vseq').head(1).copy()
fa = fa.merge(m[['pdb', 'ycat', 'half', 'paper', 'lab2', 'sars2', 'l_labs_acc_12m', 'l_labs_fam_12m', 'labs_acc_12m', 'ab_status2', 'capenz', 'fidonly', 'amyloid']], on='pdb', how='inner')
fa = fa[(fa.method == 'EM') & fa.ab_status2.isin(['bound', 'fiducial-only']) & fa.labs_acc_12m.notna() & (fa.err_first | ~fa.err_any)].copy()
fa['y_first'] = fa.err_first.astype(int); fa['vhh'] = fa.vhh; fa['mouse'] = (fa.species == 'mouse').astype(int); fa['vid'] = fa.v_id
fa['atyp'] = ((fa.L1_atyp + fa.L2_atyp + fa.L3_atyp) > 0).astype(int); fa['af'] = (fa.irm == 'AlphaFold model').astype(int); fa['rigid'] = (fa.fit == 'rigid body only').astype(int)
for sname, D in [('EM first-appearance chains', fa), ('EM first-appearance non-SARS2', fa[~fa.sars2])]:
    fit(D, 'l_labs_acc_12m', 'C(ycat) + res', label='first appearance', stratum=sname)
    fit(D, 'l_labs_acc_12m', 'C(ycat) + res + capenz + fidonly + vhh + mouse + vid + atyp + af + rigid', label='first appearance, full adj', stratum=sname)
    fit(D, 'l_labs_fam_12m', 'C(ycat) + res + capenz + fidonly + vhh + mouse + vid + atyp + af + rigid', label='first appearance, full adj', stratum=sname)

print('\n== 9. X-ray')
fit(XR, 'l_labs_acc_12m', 'C(period) + res', label='X-ray', stratum='X-ray')
fit(XR, 'l_labs_acc_12m', f'C(period) + res + capenz + {AB}', label='X-ray, ab adj', stratum='X-ray')

pd.DataFrame(R).to_csv(f'{OUT}/v02_tests.csv', index=False)
