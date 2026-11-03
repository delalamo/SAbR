# (1) Is the log-linear null hiding a non-monotone (U-shaped) dose-response? Test SARS2 against other *contested*
#     antigens in the same period, with class and binding-status controls.
# (2) Re-derive the BH q values quoted in the finding from the investigator's lens-wide test table.
# (3) Power: what per-doubling OR / binary OR can be excluded?
import warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_integrity'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
F = pd.read_csv(f'{INV}/tables/focal_entries_antigen_features.csv.gz'); F['sars2'] = F.sars2.astype(bool)
EM = F[(F.method == 'EM') & F.ab_status2.isin(['bound', 'fiducial-only']) & F.labs_acc_12m.notna()].copy()
EM['s2'] = EM.sars2.astype(int); EM['fid'] = (EM.ab_status2 == 'fiducial-only').astype(int)
EM['capenz'] = EM.tgt_cls.isin(['viral capsid', 'enzyme', 'viral other']).astype(int)
EM['cls2'] = EM.tgt_cls.where(EM.tgt_cls.map(EM.tgt_cls.value_counts()) >= 30, 'rare')
EM['bin'] = pd.cut(EM.labs_acc_12m, [-1, 0, 9, 49, 1e5], labels=['0', '1-9', '10-49', '>=50'])
def fit(D, term, extra):
    mod = smf.glm(f'y ~ {term} + {extra}', D, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.paper)[0]})
    k = [x for x in mod.params.index if x.startswith(term)][0]; ci = mod.conf_int().loc[k]
    return dict(OR=np.exp(mod.params[k]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=mod.pvalues[k], n=len(D), n_err=int(D.y.sum()), n_err_pap=D[D.y == 1].paper.nunique())
R = []
print('2020-21 EM: error rate by bin x class group (entries)')
d21 = EM[EM.period == '2020-21']
print(d21.groupby(['bin', 'capenz', 'fid']).y.agg(['mean', 'size', 'sum']).round(3).to_string())
print('\n2020-21 contested non-SARS2 target classes:'); print(d21[~d21.sars2 & (d21.labs_acc_12m > 0)].groupby('tgt_cls').y.agg(['mean', 'size', 'sum']).round(3).to_string())
for per in ['2020-21', '2022+']:
    D = EM[(EM.period == per) & (EM.sars2 | (EM.labs_acc_12m > 0))]
    for lbl, DD, extra in [('vs contested non-SARS2, year+res', D, 'C(ycat)+res'),
                           ('vs contested non-SARS2, bound only', D[D.fid == 0], 'C(ycat)+res'),
                           ('vs contested non-SARS2, bound only + capsid/enzyme', D[D.fid == 0], 'C(ycat)+res+capenz'),
                           ('vs contested non-SARS2 surface glycoproteins only (viral)', D[(D.fid == 0) & (D.sars2 | (D.tgt_cls == 'viral surface glycoprotein'))], 'C(ycat)+res')]:
        try:
            r = fit(DD, 's2', extra); r.update(test=f'{per} SARS2 {lbl}', rate_s2=DD[DD.sars2].y.mean(), rate_other=DD[~DD.sars2].y.mean()); R.append(r)
        except Exception as ex: print('fail', per, lbl, ex)
# quadratic in log2 labs (non-monotone test) on full EM, and with class
EM['l12'] = np.log2(1 + EM.labs_acc_12m); EM['l12sq'] = EM.l12 ** 2
for lbl, extra in [('year+res', 'C(ycat)+res'), ('year+res+class', 'C(ycat)+res+C(cls2)')]:
    mod = smf.glm(f'y ~ l12 + l12sq + {extra}', EM, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(EM.paper)[0]})
    R.append(dict(test=f'EM quadratic term log2(1+labs12)^2, {lbl}', OR=np.exp(mod.params['l12sq']), lo=np.exp(mod.conf_int().loc['l12sq', 0]), hi=np.exp(mod.conf_int().loc['l12sq', 1]), p=mod.pvalues['l12sq'], n=len(EM), n_err=int(EM.y.sum())))
    R.append(dict(test=f'EM linear term in quadratic model, {lbl}', OR=np.exp(mod.params['l12']), lo=np.exp(mod.conf_int().loc['l12', 0]), hi=np.exp(mod.conf_int().loc['l12', 1]), p=mod.pvalues['l12'], n=len(EM), n_err=int(EM.y.sum())))
# class-adjusted binary and per-doubling exposure (is "fewer" just class mix?)
for t in ['l12']:
    r = fit(EM, t, 'C(ycat)+res+C(cls2)+fid'); r.update(test='EM per doubling, year+res+class+fiducial'); R.append(r)
EM['any12'] = (EM.labs_acc_12m > 0).astype(int)
for lbl, D, extra in [('EM any other lab, year+res', EM, 'C(ycat)+res'), ('EM any other lab, year+res+class+fid', EM, 'C(ycat)+res+C(cls2)+fid'),
                      ('EM non-SARS2 any other lab, year+res+class+fid', EM[~EM.sars2], 'C(ycat)+res+C(cls2)+fid'),
                      ('EM 2022+ any other lab, year+res+class+fid', EM[EM.period == '2022+'], 'C(ycat)+res+C(cls2)+fid')]:
    r = fit(D, 'any12', extra); r.update(test=lbl); R.append(r)
Rt = pd.DataFrame(R)[['test', 'rate_s2', 'rate_other', 'OR', 'lo', 'hi', 'p', 'n', 'n_err', 'n_err_pap']]
print(); print(Rt.round(4).to_string())
Rt.to_csv(f'{OUT}/v04_shape_tests.csv', index=False)

# (2) BH re-derivation
T = pd.read_csv(f'{INV}/tables/all_tests_bh_lens.csv')
print('\nlens-wide tests:', len(T), '| p non-null', T.p.notna().sum())
T2 = T[T.p.notna()].copy(); T2['q_re'] = multipletests(T2.p, method='fdr_bh')[1]
qcol = [c for c in T2.columns if c.startswith('q')]
sel = T2[T2.test.str.contains('competition') & T2.term.str.contains('labs_fam_12m|labs_acc_12m|race') & T2.stratum.isin(['EM', 'EM non-SARS2', 'X-ray'])]
print(sel[['test', 'stratum', 'term', 'OR', 'p'] + qcol + ['q_re']].round(4).to_string())
# (3) power / exclusion: upper CI bounds already in R; translate per-doubling bound into 0 -> 127 labs
r = fit(EM, 'l12', 'C(ycat)+res')
print('\nper-doubling OR %.3f (upper 95%% %.3f) -> implied OR for 0 vs 127 other labs: point %.2f, upper %.2f' % (r['OR'], r['hi'], r['OR'] ** 7, r['hi'] ** 7))
EM['race'] = (EM.labs_acc_1m > 0).astype(int)
r = fit(EM, 'race', 'C(ycat)+res'); print('race_1m OR %.2f (%.2f-%.2f): cannot exclude a %.0f%% increase' % (r['OR'], r['lo'], r['hi'], 100 * (r['hi'] - 1)))
r = fit(EM[~EM.sars2], 'race', 'C(ycat)+res'); print('race_1m non-SARS2 OR %.2f (%.2f-%.2f), exposed n=%d, exposed errors=%d' % (r['OR'], r['lo'], r['hi'], EM[~EM.sars2].race.sum(), EM[~EM.sars2 & (EM.race == 1)].y.sum()))
