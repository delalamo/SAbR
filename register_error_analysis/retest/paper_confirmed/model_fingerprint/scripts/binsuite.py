"""Generic case-control suite for a binary or continuous chain feature (feature modelled as the outcome of group, as in
a 'cis rate in errors vs controls' comparison; ORs are for feature present, case vs control)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from common import logit, ols, fisher, het_z, wald_equal, paper_weights, mde_or, mde_or_binary

PAPER = ['paper_strict']
DIRECTION = -1   # direction of the full-set effect for minimum-detectable-OR; set by caller
PAPER_INCL = ['paper_strict', 'paper_loose', 'paper_copy']


def _covs(d, years='ybin_w'):
    c = [f'C({years})', 'res']
    if d.method.nunique() > 1: c.append('C(method)')
    return c


def _one(d, feat, kind, years='ybin_w'):
    """d has column case (0/1). Returns dict of crude + adjusted + paper-weighted estimates."""
    d = d.dropna(subset=[feat, 'res']).copy()
    e, c = d[d.case == 1], d[d.case == 0]
    o = dict(n_case=len(e), n_ctrl=len(c), case_units=e.unit.nunique(), case_papers=e.cluster.nunique(),
             ctrl_papers=c.cluster.nunique())
    if len(e) == 0 or len(c) == 0:
        return o
    cov = _covs(d, years)
    d['w'] = paper_weights(d, 'case')
    if kind == 'bin':
        d[feat] = d[feat].astype(int)
        a, b = int(e[feat].sum()), len(e) - int(e[feat].sum()); cc, dd = int(c[feat].sum()), len(c) - int(c[feat].sum())
        o.update(case_val=f'{a}/{len(e)} ({100 * a / len(e):.0f}%)', ctrl_val=f'{cc}/{len(c)} ({100 * cc / len(c):.0f}%)')
        f = fisher(a, b, cc, dd)
        o.update(crude_OR=f['OR'], crude_lo=f['lo'], crude_hi=f['hi'], fisher_p=f['p'], crude_se=f['se'])
        o['case_val_pw'] = np.average(e[feat], weights=d.loc[e.index, 'w']); o['ctrl_val_pw'] = np.average(c[feat], weights=d.loc[c.index, 'w'])
        zero = min(a, b) == 0
        for tag, w in [('adj', None), ('pw', 'w')]:
            if zero or e.cluster.nunique() < 3:
                continue
            try:
                r = logit(d, feat, 'case', cov, 'cluster', weights=w)
                if r['se'] > 8: continue
                o.update({f'{tag}_OR': r['OR'], f'{tag}_lo': r['lo'], f'{tag}_hi': r['hi'], f'{tag}_p': r['p'], f'{tag}_se': r['se']})
            except Exception:
                pass
        # paper level: case papers' mean vs control papers' mean (one vote per paper), Mann-Whitney
        pe = e.groupby('cluster')[feat].mean(); pc = c.groupby('cluster')[feat].mean()
        from scipy.stats import mannwhitneyu
        o['paper_case_mean'] = pe.mean(); o['paper_ctrl_mean'] = pc.mean()
        o['paper_mw_p'] = mannwhitneyu(pe, pc).pvalue if len(pe) > 1 else np.nan
        p0 = cc / len(c)
        o['mde_OR_chains'] = mde_or_binary(len(e), len(c), p0, DIRECTION)
        o['mde_OR_papers'] = mde_or_binary(e.cluster.nunique(), c.cluster.nunique(), p0, DIRECTION)
    else:
        o.update(case_val=f'mean {e[feat].mean():.3g} (median {e[feat].median():.3g})',
                 ctrl_val=f'mean {c[feat].mean():.3g} (median {c[feat].median():.3g})')
        o['crude_diff'] = e[feat].mean() - c[feat].mean()
        from scipy.stats import mannwhitneyu
        o['mw_p'] = mannwhitneyu(e[feat], c[feat]).pvalue
        for tag, w in [('adj', None), ('pw', 'w')]:
            if e.cluster.nunique() < 2: continue
            r = ols(d, feat, 'case', cov, 'cluster', weights=w)
            o.update({f'{tag}_est': r['est'], f'{tag}_lo': r['lo'], f'{tag}_hi': r['hi'], f'{tag}_p': r['p'], f'{tag}_se': r['se']})
        # one row per paper per class (paper means), OLS with paper-mean resolution
        pp = d.groupby(['cluster', 'case']).agg(v=(feat, 'mean'), res=('res', 'mean'), year=('year', 'mean')).reset_index()
        if pp[pp.case == 1].shape[0] > 1:
            m = smf.ols('v ~ case + res + year', data=pp).fit(cov_type='HC1')
            o.update(paper_est=m.params['case'], paper_lo=m.conf_int().loc['case', 0], paper_hi=m.conf_int().loc['case', 1],
                     paper_p=m.pvalues['case'], n_case_papers=int((pp.case == 1).sum()), n_ctrl_papers=int((pp.case == 0).sum()))
        o['mde_80pct'] = 2.8 * o.get('adj_se', np.nan)
    return o


def suite(D, feat, kind, label, base_mask, case_groups_main=PAPER, extra_case_mask=None, methods=('EM', 'X-ray', 'all'),
          het=True):
    """D: chain frame; base_mask: rows eligible (e.g. kappa with Pro8). Returns list of result dicts."""
    out = []
    B = D[base_mask].copy()
    for cs_lab, groups in [('strict', case_groups_main), ('incl_copies', PAPER_INCL)]:
        for meth in methods:
            d = B if meth == 'all' else B[B.method == meth]
            m_case = d.grp.isin(groups)
            if extra_case_mask is not None:
                m_case &= extra_case_mask.reindex(d.index).fillna(False).astype(bool)
            d = d[m_case | (d.grp == 'control')].assign(case=lambda x: x.grp.isin(groups).astype(int))
            if extra_case_mask is not None:
                d = d[(d.case == 0) | extra_case_mask.reindex(d.index).fillna(False).astype(bool)]
            r = _one(d, feat, kind); r.update(test=label, case_set=cs_lab, method=meth, contrast='paper vs window controls')
            out.append(r)
    if het:
        # full-set style reference and heterogeneity: rest_case vs all clean chains (control + clean_other), all years
        for meth in [m for m in methods]:
            d = B if meth == 'all' else B[B.method == meth]
            rc = d.grp == 'rest_case'
            if extra_case_mask is not None:
                rc &= extra_case_mask.reindex(d.index).fillna(False).astype(bool)
            dd = d[rc | d.grp.isin(['control', 'clean_other'])].assign(case=lambda x: (x.grp == 'rest_case').astype(int))
            r = _one(dd, feat, kind, years='ybin'); r.update(test=label, case_set='rest_case', method=meth,
                                                           contrast='rest of deeper first-occurrence vs all clean (all years)')
            # z-test against the strict paper estimate
            ps = [x for x in out if x['case_set'] == 'strict' and x['method'] == meth][0]
            if kind == 'bin':
                b1 = np.log(ps.get('adj_OR', ps.get('crude_OR', np.nan))); s1 = ps.get('adj_se', ps.get('crude_se', np.nan))
                b2 = np.log(r.get('adj_OR', r.get('crude_OR', np.nan))); s2 = r.get('adj_se', r.get('crude_se', np.nan))
                # crude Haldane version always available
                r['het_p_crude'] = het_z(np.log(ps.get('crude_OR', np.nan)), ps.get('crude_se', np.nan),
                                         np.log(r.get('crude_OR', np.nan)), r.get('crude_se', np.nan)) if 'crude_OR' in ps and 'crude_OR' in r else np.nan
            else:
                b1, s1, b2, s2 = ps.get('adj_est', np.nan), ps.get('adj_se', np.nan), r.get('adj_est', np.nan), r.get('adj_se', np.nan)
            r['het_p_z'] = het_z(b1, s1, b2, s2) if np.all(np.isfinite([b1, s1, b2, s2])) else np.nan
            # 3-group model on one control pool (all clean chains, all years)
            m_p = d.grp.isin(case_groups_main)
            if extra_case_mask is not None:
                m_p &= extra_case_mask.reindex(d.index).fillna(False).astype(bool)
            g3 = d[m_p | rc | d.grp.isin(['control', 'clean_other'])].copy()
            g3['g'] = np.where(m_p.reindex(g3.index), 'paper', np.where(g3.grp == 'rest_case', 'rest', 'aclean'))
            g3 = g3.dropna(subset=[feat, 'res'])
            try:
                cov = ' + '.join(_covs(g3, 'ybin'))
                if kind == 'bin':
                    g3[feat] = g3[feat].astype(int)
                    if g3[g3.g == 'paper'][feat].nunique() < 2: raise ValueError('zero cell in paper group')
                    mm = smf.glm(f'{feat} ~ C(g) + {cov}', data=g3, family=sm.families.Binomial()).fit(
                        cov_type='cluster', cov_kwds={'groups': pd.factorize(g3.cluster)[0]})
                else:
                    mm = smf.ols(f'{feat} ~ C(g) + {cov}', data=g3).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(g3.cluster)[0]})
                r['het_p_3group'] = wald_equal(mm, 'C(g)[T.paper]', 'C(g)[T.rest]')
                r['g3_paper_coef'] = mm.params['C(g)[T.paper]']; r['g3_rest_coef'] = mm.params['C(g)[T.rest]']
            except Exception as ex:
                r['het_note'] = str(ex)[:60]
            out.append(r)
    return out
