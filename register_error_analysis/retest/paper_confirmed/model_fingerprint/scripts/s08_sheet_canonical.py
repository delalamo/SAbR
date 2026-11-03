"""T8a less beta-sheet across the V domain away from the site (PROMOTIF sheet fraction of V residues >5 from any error
run; pdb_meta 08_cis_sheet.py; recomputed with runs_all for copy chains in s01). Chain level + within entry (error chain vs
error-free V chain of the same type in the same entry; full set: error chain lower in 70% of 52 pairs, p=4e-5).
T8b deposited non-H3 CDRs in the same ABEGO backbone bins as the ImmuneBuilder prediction (struct_unusual s10,
tables/pred_compare_chain.csv). Only chains in the struct_unusual EM sample have predictions (ImmuneBuilder is not
installed here), so this is a subset test.
Output: tables/sheet_tests.csv, tables/sheet_within_entry.csv, tables/canonical_tests.csv"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import DEEP, TAB, load_frame, het_z
import binsuite
from binsuite import suite

D = load_frame(); D = D[~D.amyloid & D.v_sheet_away.notna()].copy()
D['cclass'] = np.where(D.vhh == 1, 'VHH', D.ctype)
binsuite.DIRECTION = -1
_orig = binsuite._covs
binsuite._covs = lambda d, years='ybin_w': _orig(d, years) + ['C(cclass)']
res = suite(D, 'v_sheet_away', 'cont', 'T8a V-domain sheet fraction >5 residues from the error run', D.v_sheet_away.notna())
binsuite._covs = _orig
pd.DataFrame(res).to_csv(f'{TAB}/sheet_tests.csv', index=False)

rows = []
for cs_lab, groups in [('strict', ['paper_strict']), ('incl_copies', ['paper_strict', 'paper_loose', 'paper_copy']), ('rest_case', ['rest_case'])]:
    for meth in ['EM', 'X-ray', 'all']:
        c = D if meth == 'all' else D[D.method == meth]
        w = c[c.grp.isin(groups) | ~c.err_any].copy(); w['case'] = w.grp.isin(groups).astype(int)
        # pairs: same entry and same chain class
        w['ek'] = w.pdb + '|' + w.cclass
        ok = w.groupby('ek').case.agg(['min', 'max'])
        ok = ok[(ok['min'] == 0) & (ok['max'] == 1)].index
        w = w[w.ek.isin(ok)]
        if len(ok) < 3:
            rows.append(dict(case_set=cs_lab, method=meth, n_pairs=len(ok))); continue
        pe = w.groupby(['ek', 'case']).v_sheet_away.mean().unstack(); dif = pe[1] - pe[0]
        mm = smf.ols('v_sheet_away ~ case + C(ek)', w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(w.cluster)[0]})
        pp = w.groupby(['cluster', 'case']).v_sheet_away.mean().unstack().dropna(); pdif = pp[1] - pp[0]
        rows.append(dict(case_set=cs_lab, method=meth, n_pairs=len(ok), n_papers=w.cluster.nunique(), frac_case_lower=(dif < 0).mean(),
                         n_lower=int((dif < 0).sum()), sign_p=stats.binomtest(int((dif < 0).sum()), int((dif != 0).sum())).pvalue,
                         median_diff=dif.median(), fe_est=mm.params['case'], fe_lo=mm.conf_int().loc['case', 0], fe_hi=mm.conf_int().loc['case', 1],
                         fe_p=mm.pvalues['case'], fe_se=mm.bse['case'], papers_lower=int((pdif < 0).sum()), papers_n=len(pdif),
                         paper_sign_p=stats.binomtest(int((pdif < 0).sum()), int((pdif != 0).sum())).pvalue if (pdif != 0).sum() else np.nan))
W = pd.DataFrame(rows)
for meth in ['EM', 'X-ray', 'all']:
    s_ = W[(W.case_set == 'strict') & (W.method == meth)].iloc[0]; r_ = W[(W.case_set == 'rest_case') & (W.method == meth)].iloc[0]
    if np.isfinite(s_.get('fe_se', np.nan)) and np.isfinite(r_.get('fe_se', np.nan)):
        W.loc[(W.case_set == 'strict') & (W.method == meth), 'het_p_vs_rest'] = het_z(s_.fe_est, s_.fe_se, r_.fe_est, r_.fe_se)
W.to_csv(f'{TAB}/sheet_within_entry.csv', index=False)
pd.set_option('display.width', 250)
print(W.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

# ---- T8b canonical CDR conformation (subset)
P = pd.read_csv(f'{DEEP}/struct_unusual/tables/pred_compare_chain.csv', dtype={'chain': str}).drop(columns=['cclass'])
L = D[['pdb', 'chain', 'grp', 'cluster', 'res', 'year', 'ybin_w', 'ybin', 'method', 'cclass', 'err_any', 'in_paper_window']].merge(P, on=['pdb', 'chain'])
L = L[L.cdr_bin_agree.notna()]
rows = []
def cmp(lab, a, b):
    x = pd.concat([a.assign(case=1), b.assign(case=0)])
    o = dict(test=lab, n_case=len(a), case_papers=a.cluster.nunique(), n_ctrl=len(b), case_mean=a.cdr_bin_agree.mean(), ctrl_mean=b.cdr_bin_agree.mean())
    if len(a) >= 3 and len(b) >= 3:
        o['mw_p'] = stats.mannwhitneyu(a.cdr_bin_agree, b.cdr_bin_agree).pvalue
        m = smf.ols('cdr_bin_agree ~ case + res + C(cclass)', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.cluster)[0]})
        o.update(est=m.params['case'], lo=m.conf_int().loc['case', 0], hi=m.conf_int().loc['case', 1], p=m.pvalues['case'], se=m.bse['case'])
        pp = x.groupby(['cluster', 'case']).cdr_bin_agree.mean().reset_index()
        o['paper_level_diff'] = pp[pp.case == 1].cdr_bin_agree.mean() - pp[pp.case == 0].cdr_bin_agree.mean()
        o['paper_level_mw_p'] = stats.mannwhitneyu(pp[pp.case == 1].cdr_bin_agree, pp[pp.case == 0].cdr_bin_agree).pvalue
    rows.append(o); return o
ctl_all = L[(L.role_cc == 'control')]
ctl_win = L[(L.role_cc == 'control') & (L.grp == 'control')]
for cs_lab, groups in [('strict', ['paper_strict']), ('incl_copies', ['paper_strict', 'paper_loose', 'paper_copy'])]:
    a = L[L.grp.isin(groups) & (L.role_cc == 'case')]
    cmp(f'T8b {cs_lab} case chains vs window controls in the sample', a, ctl_win)
    cmp(f'T8b {cs_lab} case chains vs all sample controls', a, ctl_all)
    nb = L[L.pdb.isin(set(L[L.grp.isin(groups)].pdb)) & ~L.err_any]
    cmp(f'T8b error-free chains of {cs_lab} case entries vs all sample controls', nb, ctl_all)
rc = L[(L.grp == 'rest_case') & (L.role_cc == 'case')]
o = cmp('T8b rest_case chains vs all sample controls', rc, ctl_all)
s_ = [r for r in rows if r['test'] == 'T8b strict case chains vs all sample controls'][0]
if 'se' in s_ and 'se' in o: o['het_p_vs_paper'] = het_z(s_['est'], s_['se'], o['est'], o['se'])
cmp('T8b all sample case chains vs all sample controls (full-set reproduction)', L[L.role_cc == 'case'], ctl_all)
C = pd.DataFrame(rows); C.to_csv(f'{TAB}/canonical_tests.csv', index=False)
print(C.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
