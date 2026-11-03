"""How conkit's selection could shape the paper-confirmed fingerprint.
(1) Same-era comparison: deeper first-occurrence chains IN the paper window that are NOT paper-confirmed (conkit missed
    them, or their entry is in Table S1 for another chain) vs the same window controls, for the chain-level fingerprint
    metrics; and a direct paper-confirmed vs conkit-missed contrast (adjusted for year bin, resolution, chain class).
(2) Descriptors of the case sets: entries in which every V chain is wrong, run length, pre-2018-05 share.
Output: tables/selection_tests.csv, tables/selection_descriptors.csv"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import SP, DEEP, TAB, load_frame

D = load_frame(); D = D[~D.amyloid].copy(); D['cclass'] = np.where(D.vhh == 1, 'VHH', D.ctype)
RC = pd.read_csv(f'{TAB}/rama_chain.csv.gz', dtype={'chain': str})[['pdb', 'chain', 'out_A', 'out_Cl']]
FWc = pd.read_csv(f'{TAB}/fw_computed.csv', dtype={'chain': str})[['pdb', 'chain', 'rep_rmsd']]
VDIR = f'{DEEP}/verify/case_files_DE_DE2_framework_distortion_integrity'
F0 = pd.concat([pd.read_csv(f'{VDIR}/fw_recomputed.csv', dtype={'chain': str}), pd.read_csv(f'{VDIR}/fw_recomputed_extra.csv', dtype={'chain': str})])[['pdb', 'chain', 'rep_rmsd']]
FW = pd.concat([FWc, F0]).drop_duplicates(['pdb', 'chain'])
X = D.merge(RC, on=['pdb', 'chain'], how='left').merge(FW, on=['pdb', 'chain'], how='left')
X['set'] = np.select([X.grp == 'paper_strict', X.rest_in_window, X.grp == 'control'], ['paper_strict', 'missed_in_window', 'control'], 'other')
rows = []
for feat, kind, base in [('out_A', 'cont', X.out_A.notna()), ('out_Cl', 'cont', X.out_A.notna()), ('rep_rmsd', 'cont', X.rep_rmsd.notna()),
                         ('v_sheet_away', 'cont', X.v_sheet_away.notna()), ('cis8', 'bin', (X.ctype == 'K') & X.pro8)]:
    for meth in ['EM', 'X-ray', 'all']:
        d = X[base & X.set.isin(['paper_strict', 'missed_in_window', 'control'])]
        if meth != 'all': d = d[d.method == meth]
        d = d.dropna(subset=[feat, 'res']).copy()
        if kind == 'bin': d[feat] = d[feat].astype(int)
        cov = 'C(ybin_w) + res' + (' + C(cclass)' if kind == 'cont' else '') + (' + C(method)' if meth == 'all' else '')
        o = dict(feature=feat, method=meth, n_paper=int((d.set == 'paper_strict').sum()), n_missed=int((d.set == 'missed_in_window').sum()),
                 n_ctrl=int((d.set == 'control').sum()), paper_mean=d[d.set == 'paper_strict'][feat].mean(),
                 missed_mean=d[d.set == 'missed_in_window'][feat].mean(), ctrl_mean=d[d.set == 'control'][feat].mean(),
                 missed_papers=d[d.set == 'missed_in_window'].cluster.nunique())
        try:
            if kind == 'cont':
                m = smf.ols(f'{feat} ~ C(set, Treatment("control")) + {cov}', d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.cluster)[0]})
            else:
                m = smf.glm(f'{feat} ~ C(set, Treatment("control")) + {cov}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.cluster)[0]})
            for s in ['paper_strict', 'missed_in_window']:
                k = f'C(set, Treatment("control"))[T.{s}]'
                if k in m.params: o.update({f'{s}_vs_ctrl': m.params[k], f'{s}_lo': m.conf_int().loc[k, 0], f'{s}_hi': m.conf_int().loc[k, 1], f'{s}_p': m.pvalues[k]})
            names = list(m.params.index); r = np.zeros(len(names))
            r[names.index('C(set, Treatment("control"))[T.paper_strict]')] = 1; r[names.index('C(set, Treatment("control"))[T.missed_in_window]')] = -1
            t = m.t_test(r); o['paper_minus_missed'] = float(np.squeeze(t.effect)); o['paper_minus_missed_p'] = float(t.pvalue)
            if kind == 'bin' and abs(o['paper_minus_missed']) > 10: o['note'] = 'zero cell in paper set; contrast not estimable'
        except Exception as ex:
            o['note'] = str(ex)[:60]
        rows.append(o)
S = pd.DataFrame(rows); S.to_csv(f'{TAB}/selection_tests.csv', index=False)
pd.set_option('display.width', 250)
print(S.to_string(index=False, float_format=lambda v: f'{v:.3g}'))

# descriptors
RA = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
rl = RA.groupby(['pdb', 'chain']).length.max().rename('max_run_len').reset_index()
X = X.merge(rl, on=['pdb', 'chain'], how='left')
ent_all_wrong = D.groupby('pdb').err_any.apply(lambda s: s.astype(bool).all())
X['entry_all_V_wrong'] = X.pdb.map(ent_all_wrong)
X['pre_af2'] = X.release.astype(str) < '2018-05-01'
desc = []
for s in ['paper_strict', 'missed_in_window']:
    x = X[X.set == s]
    for meth in ['EM', 'X-ray', 'all']:
        y = x if meth == 'all' else x[x.method == meth]
        desc.append(dict(set=s, method=meth, chains=len(y), entries=y.pdb.nunique(), papers=y.cluster.nunique(),
                         frac_entry_all_V_wrong=y.entry_all_V_wrong.mean(), median_run_len=y.max_run_len.median(),
                         frac_pre_2018_05=y.pre_af2.mean(), frac_modeA=y.modeA.mean(), frac_modeDE=y.modeDE.mean(),
                         median_res=y.res.median(), median_year=y.year.median(),
                         top_paper_share=y.cluster.value_counts(normalize=True).iloc[0] if len(y) else np.nan))
Dd = pd.DataFrame(desc); Dd.to_csv(f'{TAB}/selection_descriptors.csv', index=False)
print(Dd.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
