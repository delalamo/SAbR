"""T5 framework C-alpha RMSD to a reference (run excluded; 'fw_rmsd' of case_files_DE = rep_rmsd), and T7b within-entry
contrast. Values: tables/fw_computed.csv (this lens) + deeper/verify/case_files_DE_DE2_framework_distortion_integrity/
fw_recomputed*.csv (same code, reproduced to 2e-5 A) for EM rest_case chains and the verifier's EM clean chains.
Output: tables/fw_tests.csv, tables/fw_or_per_sd.csv, tables/fw_within_entry.csv"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import SP, DEEP, TAB, load_frame, het_z, logit
import binsuite
from binsuite import suite

D = load_frame(); D = D[~D.amyloid]
VDIR = f'{DEEP}/verify/case_files_DE_DE2_framework_distortion_integrity'
F = pd.read_csv(f'{TAB}/fw_computed.csv', dtype={'chain': str})[['pdb', 'chain', 'rep_rmsd', 'strict_rmsd', 'fr12_rmsd']]
F0 = pd.concat([pd.read_csv(f'{VDIR}/fw_recomputed.csv', dtype={'chain': str}), pd.read_csv(f'{VDIR}/fw_recomputed_extra.csv', dtype={'chain': str})])
F0 = F0[['pdb', 'chain', 'rep_rmsd', 'strict_rmsd', 'fr12_rmsd']].drop_duplicates(['pdb', 'chain'])
F0 = F0[~F0.set_index(['pdb', 'chain']).index.isin(F.set_index(['pdb', 'chain']).index)]
FW = pd.concat([F.assign(src='this lens'), F0.assign(src='DE2 verifier')])
X = FW.merge(D, on=['pdb', 'chain'])
X['cclass'] = np.where(X.vhh == 1, 'VHH', X.ctype)
X = X[X.rep_rmsd.notna()]
print(X.groupby(['grp', 'method', 'src']).size().to_string())
binsuite.DIRECTION = 1
_orig = binsuite._covs
binsuite._covs = lambda d, years='ybin_w': _orig(d, years) + ['C(cclass)']
res = []
res += suite(X, 'rep_rmsd', 'cont', 'T5 framework CA RMSD to reference, runs excluded (A)', X.rep_rmsd.notna())
res += suite(X, 'strict_rmsd', 'cont', 'T5b framework CA RMSD, every SAbR/ANARCI disagreement excluded (A)', X.strict_rmsd.notna())
res += suite(X, 'fr12_rmsd', 'cont', 'T5c FR1+FR2-only RMSD (away from CDR3/FR4) (A)', X.fr12_rmsd.notna())
binsuite._covs = _orig
R = pd.DataFrame(res); R.to_csv(f'{TAB}/fw_tests.csv', index=False)

# OR per SD (0.47 A, the DE2 scale) with case as outcome, for comparability with the full-set 'OR 3.2 per SD'
rows = []
for cs_lab, groups, ctl in [('strict', ['paper_strict'], ['control']), ('incl_copies', ['paper_strict', 'paper_loose', 'paper_copy'], ['control']),
                            ('rest_case', ['rest_case'], ['control', 'clean_other'])]:
    for meth in ['EM', 'X-ray', 'all']:
        d = X if meth == 'all' else X[X.method == meth]
        d = d[d.grp.isin(groups + ctl)].copy(); d['y'] = d.grp.isin(groups).astype(int); d['z'] = d.rep_rmsd / 0.47
        yb = 'ybin_w' if cs_lab != 'rest_case' else 'ybin'
        cov = [f'C({yb})', 'res', 'C(cclass)'] + (['C(method)'] if meth == 'all' else [])
        try:
            r = logit(d, 'y', 'z', cov, 'cluster')
        except Exception as ex:
            r = dict(note=str(ex)[:50])
        d['w'] = 1 / d.groupby(['cluster', 'y']).y.transform('size')
        try:
            rw = logit(d, 'y', 'z', cov, 'cluster', weights='w')
        except Exception:
            rw = {}
        rows.append(dict(case_set=cs_lab, method=meth, n_case=int(d.y.sum()), n_ctrl=int((1 - d.y).sum()), case_papers=d[d.y == 1].cluster.nunique(),
                         case_mean=d[d.y == 1].rep_rmsd.mean(), ctrl_mean=d[d.y == 0].rep_rmsd.mean(),
                         **{f'chain_{k}': v for k, v in r.items()}, **{f'pw_{k}': v for k, v in rw.items()}))
O = pd.DataFrame(rows)
for meth in ['EM', 'X-ray', 'all']:
    a = O[(O.case_set == 'strict') & (O.method == meth)].iloc[0]; b = O[(O.case_set == 'rest_case') & (O.method == meth)].iloc[0]
    O.loc[(O.case_set == 'strict') & (O.method == meth), 'het_p_vs_rest'] = het_z(np.log(a.chain_OR), a.chain_se, np.log(b.chain_OR), b.chain_se)
O.to_csv(f'{TAB}/fw_or_per_sd.csv', index=False)

# T7b within entry: case chains vs err_any==0 V chains of the same entry; neighbours vs window controls
rows = []
for cs_lab, groups in [('strict', ['paper_strict']), ('incl_copies', ['paper_strict', 'paper_loose', 'paper_copy']), ('rest_case', ['rest_case'])]:
    for meth in ['EM', 'X-ray', 'all']:
        c = X if meth == 'all' else X[X.method == meth]
        ents = set(c[c.grp.isin(groups)].pdb) & set(c[~c.err_any].pdb)
        w = c[c.pdb.isin(ents) & (c.grp.isin(groups) | ~c.err_any)].copy(); w['case'] = w.grp.isin(groups).astype(int)
        if len(ents) < 3:
            rows.append(dict(case_set=cs_lab, method=meth, n_entries=len(ents))); continue
        mm = smf.ols('rep_rmsd ~ case + C(pdb) + C(cclass)', w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(w.cluster)[0]})
        pe = w.groupby(['pdb', 'case']).rep_rmsd.mean().unstack(); dif = pe[1] - pe[0]
        rows.append(dict(case_set=cs_lab, method=meth, n_entries=len(ents), n_papers=w[w.case == 1].cluster.nunique(),
                         n_case_chains=int(w.case.sum()), n_clean_chains=int((1 - w.case).sum()),
                         fe_est=mm.params['case'], fe_lo=mm.conf_int().loc['case', 0], fe_hi=mm.conf_int().loc['case', 1], fe_p=mm.pvalues['case'],
                         fe_se=mm.bse['case'], entries_case_higher=int((dif > 0).sum()), entry_wilcoxon_p=stats.wilcoxon(dif).pvalue if len(dif) > 5 else np.nan))
W = pd.DataFrame(rows)
for meth in ['EM', 'X-ray', 'all']:
    s_ = W[(W.case_set == 'strict') & (W.method == meth)].iloc[0]; r_ = W[(W.case_set == 'rest_case') & (W.method == meth)].iloc[0]
    if np.isfinite(s_.get('fe_se', np.nan)) and np.isfinite(r_.get('fe_se', np.nan)):
        W.loc[(W.case_set == 'strict') & (W.method == meth), 'het_p_vs_rest'] = het_z(s_.fe_est, s_.fe_se, r_.fe_est, r_.fe_se)
nb = []
for meth in ['EM', 'X-ray', 'all']:
    c = X if meth == 'all' else X[X.method == meth]
    ents = set(c[c.grp == 'paper_strict'].pdb)
    w = c[(c.pdb.isin(ents) & ~c.err_any) | (c.grp == 'control')].copy(); w['case'] = w.pdb.isin(ents).astype(int)
    if w.case.sum() < 3: continue
    f = 'rep_rmsd ~ case + C(ybin_w) + res + C(cclass)' + (' + C(method)' if meth == 'all' else '')
    mm = smf.ols(f, w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(w.cluster)[0]})
    nb.append(dict(case_set='error-free V chains of strict-case entries vs window controls', method=meth, n_case_chains=int(w.case.sum()),
                   n_clean_chains=int((1 - w.case).sum()), n_papers=w[w.case == 1].cluster.nunique(), fe_est=mm.params['case'],
                   fe_lo=mm.conf_int().loc['case', 0], fe_hi=mm.conf_int().loc['case', 1], fe_p=mm.pvalues['case']))
W = pd.concat([W, pd.DataFrame(nb)]); W.to_csv(f'{TAB}/fw_within_entry.csv', index=False)
pd.set_option('display.width', 250)
print(O[['case_set', 'method', 'n_case', 'n_ctrl', 'case_papers', 'case_mean', 'ctrl_mean', 'chain_OR', 'chain_lo', 'chain_hi', 'chain_p', 'pw_OR', 'pw_lo', 'pw_hi', 'pw_p', 'het_p_vs_rest']].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print(W.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

# ---- D/E-only version (the full-set 1.56 vs 1.02 A headline was for CDR3/FR4 slips) and non-D/E modes
rows = []
for lab, m in [('strict, D/E (CDR3/FR4) chains', (X.grp == 'paper_strict') & X.modeDE), ('strict, non-D/E chains', (X.grp == 'paper_strict') & ~X.modeDE),
               ('rest_case, D/E chains', (X.grp == 'rest_case') & X.modeDE)]:
    for meth in ['EM', 'all']:
        ctl = ['control'] if lab.startswith('strict') else ['control', 'clean_other']
        d = X[m | X.grp.isin(ctl)]
        if meth != 'all': d = d[d.method == meth]
        d = d.copy(); d['case'] = m.reindex(d.index).astype(int)
        yb = 'ybin_w' if lab.startswith('strict') else 'ybin'
        f = f'rep_rmsd ~ case + C({yb}) + res + C(cclass)' + (' + C(method)' if meth == 'all' else '')
        mm = smf.ols(f, d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.cluster)[0]})
        pp = d.groupby(['cluster', 'case']).rep_rmsd.mean().reset_index()
        rows.append(dict(test=lab, method=meth, n_case=int(d.case.sum()), case_papers=d[d.case == 1].cluster.nunique(), case_mean=d[d.case == 1].rep_rmsd.mean(),
                         ctrl_mean=d[d.case == 0].rep_rmsd.mean(), est=mm.params['case'], lo=mm.conf_int().loc['case', 0], hi=mm.conf_int().loc['case', 1], p=mm.pvalues['case'],
                         paper_mean_case=pp[pp.case == 1].rep_rmsd.mean(), paper_mean_ctrl=pp[pp.case == 0].rep_rmsd.mean(),
                         paper_mw_p=stats.mannwhitneyu(pp[pp.case == 1].rep_rmsd, pp[pp.case == 0].rep_rmsd).pvalue))
DE = pd.DataFrame(rows); DE.to_csv(f'{TAB}/fw_by_mode.csv', index=False)
print(DE.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
