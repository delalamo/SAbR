"""Sensitivity: drop the dominant paper (10.1038/s41422-021-00487-9, Cell Res 2021, 7CZR-7D00; 17 of 67 strict chains,
18 of 37 strict EM Fv instances) and re-run the key structural and sequence tests (strict set, same models as s04/s05).
Also: leave-one-paper-out range of the strict EM estimate for the deposited-orientation and site-error tests.
Output: tables/dominant_paper.csv"""
import sys, os, numpy as np, pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
from aoutil import TAB, logit_or, lin_diff, fisher

DOM = '10.1038/s41422-021-00487-9'
F = pd.read_csv(f'{TAB}/struct_fv_frame.csv', dtype={'H': str, 'L': str})
F = F[(F.kind == 'Fv') & ((F.role == 'control') | F.strict)].copy(); F['y'] = (F.role == 'case').astype(int)
CH = pd.read_csv(f'{TAB}/struct_chain_frame.csv', dtype={'chain': str})
SE = pd.read_csv(f'{TAB}/struct_site_err.csv', dtype={'chain': str})
X = pd.read_csv(f'{TAB}/units_frame.csv.gz', low_memory=False); X['res'] = X.res_med
rows = []


def s_test(name, d, metric, kind, covs, meth):
    dd = d if meth == 'all' else d[d.method == meth]
    cv = [c for c in covs if not (meth != 'all' and c == 'C(method)')]
    dd = dd.dropna(subset=[metric]).copy()
    if kind == 'cont':
        m = lin_diff(dd, metric, cv, y='y', groups='paper'); e = m.get('diff')
    else:
        k1, k0 = int(dd[dd.y == 1][metric].sum()), int(dd[dd.y == 0][metric].sum())
        dd['m'] = dd[metric]; m = logit_or(dd, 'y', cv, y='m', groups='paper') if 0 < k1 < dd.y.sum() else fisher(k1, int(dd.y.sum()), k0, int((dd.y == 0).sum()))
        e = m.get('OR')
    return dict(test=name, method=meth, n1=int(dd.y.sum()), papers1=dd[dd.y == 1].paper.nunique(), v1=dd[dd.y == 1][metric].mean(),
                v0=dd[dd.y == 0][metric].mean(), est=e, lo=m.get('lo'), hi=m.get('hi'), p=m.get('p'))


covF = ['C(yb)', 'res', 'C(method)', 'C(Lct)']
for lab, d in [('all strict', F), ('without dominant paper', F[F.paper != DOM])]:
    for meth in ['all', 'EM']:
        for metric, kind in [('maha_dep', 'cont'), ('outlier99', 'bin'), ('core_rms', 'cont'), ('HC2_sabr', 'cont'), ('LC1_sabr', 'cont'), ('pred_maha', 'cont')]:
            r = s_test(metric, d, metric, kind, covF, meth); r['set'] = lab; rows.append(r)
        u = d[d.undistorted == 1]
        r = s_test('outlier99 | undistorted cores', u, 'outlier99', 'bin', covF, meth); r['set'] = lab; rows.append(r)
        r = s_test('maha_dep | undistorted cores', u, 'maha_dep', 'cont', covF, meth); r['set'] = lab; rows.append(r)
CH['y'] = (CH.role == 'case').astype(int); CH['yb'] = np.where(CH.year <= 2017, '<=2017', np.where(CH.year <= 2019, '2018-19', np.where(CH.year >= 2021, '>=2021', '2020')))
CHs = CH[(CH.role == 'control') | CH.strict_chain]
for lab, d in [('all strict', CHs), ('without dominant paper', CHs[CHs.paper != DOM])]:
    for meth in ['all', 'EM']:
        r = s_test('pred_err_chain', d, 'pred_err_chain', 'cont', ['C(yb)', 'res', 'C(method)', 'C(cls)'], meth); r['set'] = lab; rows.append(r)
    for meth in ['all', 'EM']:
        s = SE[SE.strict & (SE.paper != DOM if lab != 'all strict' else True)]
        s = s if meth == 'all' else s[s.method == meth]
        m = smf.ols('diff ~ 1', data=s).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s.paper)[0]})
        ci = m.conf_int().iloc[0]
        rows.append(dict(test='site_err (case - matched controls)', method=meth, set=lab, n1=len(s), papers1=s.paper.nunique(), v1=s.site_err.mean(),
                         v0=s.ctrl_site_err.mean(), est=m.params.iloc[0], lo=ci[0], hi=ci[1], p=m.pvalues.iloc[0]))
# sequence features (M1 GEE, unit level)
for lab, d0 in [('all strict', X), ('without dominant paper', X[X.paper != DOM])]:
    d0 = d0[(d0.paper_strict | d0.control) & ~d0.any_amyloid].copy(); d0['y'] = d0.paper_strict.astype(int); d0['yb'] = d0.yb_w
    for f in ['f_rare_cdr_len5', 'f_has_hq_xray', 'f_alpaca', 'f_nonhuman_mouse', 'f_nanobody', 'f_first_V', 'f_nn_any_V', 'f_prior_same_ab']:
        for meth in ['all', 'EM']:
            d = d0 if meth == 'all' else d0[d0.method == meth]
            cov = ['C(yb)', 'res', 'C(ctype)'] + (['C(method)'] if meth == 'all' else [])
            d = d.dropna(subset=[f, 'res']).copy()
            isb = set(d[f].unique()) <= {0, 1}
            d['x'] = d[f] if isb else (d[f] - d[d.y == 0][f].mean()) / d[d.y == 0][f].std()
            k1 = int(d[d.y == 1][f].sum()) if isb else None
            m = logit_or(d, 'x', cov, groups='paper') if (not isb or 0 < k1 < d.y.sum()) else fisher(k1, int(d.y.sum()), int(d[d.y == 0][f].sum()), int((d.y == 0).sum()))
            rows.append(dict(test=f, method=meth, set=lab, n1=int(d.y.sum()), papers1=d[d.y == 1].paper.nunique(), v1=d[d.y == 1][f].mean(),
                             v0=d[d.y == 0][f].mean(), est=m.get('OR'), lo=m.get('lo'), hi=m.get('hi'), p=m.get('p')))
R = pd.DataFrame(rows)
R.to_csv(f'{TAB}/dominant_paper.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(R.round(3).to_string(index=False))
