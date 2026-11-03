"""Canonical cis-peptide fingerprint on the paper-confirmed set:
  T1  kappa Pro8 (IMGT 8) modelled cis; also excluding chains with an FR1 (mode A) error
  T1b trans-Pro8 as a flag: PPV / NPV / sensitivity in the paper window
  T1c within-paper sign test (paper-level habit)
  T2  kappa CDR-L3 Pro115 (Kabat 95) cis; also excluding chains with any run reaching IMGT >=104 (position filter of
      verify/pdb_meta_cis_constant_and_L3_integrity)
  T9  constant-domain cis (C-kappa, CH1), with and without the dominant all-trans paper 10.1038/s41422-021-00487-9
Cis annotation: wwPDB CIS-PEPTIDE features mapped to ANARCI IMGT (pdb_meta/tables/chain_cis_sheet.csv.gz), validated
against coordinates 580/580 by verify/pdb_meta_cis_pro_kappa8_integrity.
Output: tables/cis_tests.csv, tables/pro8_flag.csv, tables/pro8_within_paper.csv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import SP, TAB, load_frame, fisher
from binsuite import suite
from scipy import stats

D = load_frame()
D = D[~D.amyloid & D.has_cs].copy()
K = (D.ctype == 'K')
res = []
# --- full-set reproduction (EM, all years): err_first vs err_any==0
for lab, m, f in [('kappa Pro8 cis', K & D.pro8, 'cis8'), ('kappa Pro115 cis', K & D.pro115, 'cis115')]:
    x = D[m & (D.method == 'EM') & (D.err_first | ~D.err_any)]
    a = int(x[x.err_first][f].sum()); n1 = int(x.err_first.sum()); c = int(x[~x.err_any][f].sum()); n0 = int((~x.err_any).sum())
    print(f'full-set reproduction {lab}: {a}/{n1} vs {c}/{n0}')

# --- T1 Pro8
res += suite(D, 'cis8', 'bin', 'T1 kappa Pro8 cis', K & D.pro8)
noA = ~D.modeA
res += suite(D, 'cis8', 'bin', 'T1 kappa Pro8 cis, cases without FR1 (mode A) error', K & D.pro8, extra_case_mask=noA)
# --- T2 Pro115
res += suite(D, 'cis115', 'bin', 'T2 kappa Pro115 (CDR-L3 Kabat 95) cis', K & D.pro115)
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
reach = set(zip(RA[RA.imgt1 >= 104].pdb, RA[RA.imgt1 >= 104].chain))
far = pd.Series([k not in reach for k in zip(D.pdb, D.chain)], index=D.index)
res += suite(D, 'cis115', 'bin', 'T2 kappa Pro115 cis, cases with no run reaching IMGT>=104', K & D.pro115, extra_case_mask=far)
nonA_far = far & ~D.modeA
res += suite(D, 'cis115', 'bin', 'T2 kappa Pro115 cis, cases with no run reaching IMGT>=104 and no FR1 error', K & D.pro115,
             extra_case_mask=nonA_far, het=False)
# --- T9 constant domains
DOM = '10.1038/s41422-021-00487-9'
for ct, nm in [('K', 'C-kappa'), ('H', 'CH1'), ('L', 'C-lambda')]:
    m = (D.ctype == ct) & D.const_cis_any.notna()
    res += suite(D, 'const_cis_any', 'bin', f'T9 any cis in {nm}', m)
    res += suite(D, 'const_cis_any', 'bin', f'T9 any cis in {nm}, dominant all-trans paper dropped', m & (D.cluster != DOM), het=False)
m = D.ctype.isin(['K', 'H', 'L']) & D.const_cis_any.notna()
res += suite(D, 'const_cis_any', 'bin', 'T9 any cis in constant domain (pooled)', m)
res += suite(D, 'const_cis_any', 'bin', 'T9 any cis in constant domain (pooled), dominant paper dropped', m & (D.cluster != DOM), het=False)

R = pd.DataFrame(res)
cols = ['test', 'case_set', 'method', 'contrast', 'n_case', 'case_units', 'case_papers', 'n_ctrl', 'ctrl_papers', 'case_val', 'ctrl_val',
        'crude_OR', 'crude_lo', 'crude_hi', 'fisher_p', 'adj_OR', 'adj_lo', 'adj_hi', 'adj_p', 'case_val_pw', 'ctrl_val_pw',
        'pw_OR', 'pw_lo', 'pw_hi', 'pw_p', 'paper_case_mean', 'paper_ctrl_mean', 'paper_mw_p', 'mde_OR_chains', 'mde_OR_papers',
        'het_p_crude', 'het_p_z', 'het_p_3group', 'g3_paper_coef', 'g3_rest_coef', 'het_note']
R = R[[c for c in cols if c in R.columns]]
R.to_csv(f'{TAB}/cis_tests.csv', index=False)

# --- T1b trans-Pro8 as a flag, paper window, kappa chains with Pro8
rows = []
W = D[K & D.pro8 & D.in_paper_window]
for meth in ['EM', 'X-ray', 'all']:
    w = W if meth == 'all' else W[W.method == meth]
    for lab, case, univ in [
        ('paper strict error (universe: strict cases + window controls)', w.grp == 'paper_strict', w.grp.isin(['paper_strict', 'control'])),
        ('paper error incl copies (universe: cases + window controls)', w.grp.isin(['paper_strict', 'paper_loose', 'paper_copy']),
         w.grp.isin(['paper_strict', 'paper_loose', 'paper_copy', 'control'])),
        ('any deeper first-occurrence error (universe: err_first + clean, window)', w.err_first, w.err_first | ~w.err_any)]:
        u = w[univ]; cs = case[univ]; tr = ~u.cis8
        tp = int((cs & tr).sum()); fp = int((~cs & tr).sum()); fn = int((cs & ~tr).sum()); tn = int((~cs & ~tr).sum())
        rows.append(dict(method=meth, outcome=lab, n=len(u), errors=tp + fn, trans=tp + fp,
                         err_rate_if_trans=f'{tp}/{tp + fp} ({100 * tp / max(tp + fp, 1):.2f}%)',
                         err_rate_if_cis=f'{fn}/{fn + tn} ({100 * fn / max(fn + tn, 1):.2f}%)',
                         PPV=tp / max(tp + fp, 1), NPV=tn / max(fn + tn, 1), sensitivity=tp / max(tp + fn, 1),
                         specificity=tn / max(tn + fp, 1), RR=(tp / max(tp + fp, 1)) / max(fn / max(fn + tn, 1), 1e-9)))
P = pd.DataFrame(rows); P.to_csv(f'{TAB}/pro8_flag.csv', index=False)

# --- T1c within-paper: case papers that also hold clean kappa chains with Pro8
rows = []
for cs_lab, groups in [('strict', ['paper_strict']), ('incl_copies', ['paper_strict', 'paper_loose', 'paper_copy'])]:
    Kp = D[K & D.pro8]
    for c, g in Kp[Kp.cluster.isin(Kp[Kp.grp.isin(groups)].cluster)].groupby('cluster'):
        e = g[g.grp.isin(groups)]; cl = g[~g.err_any]
        rows.append(dict(case_set=cs_lab, paper=c, n_case=len(e), cis_case=e.cis8.mean(), n_clean=len(cl),
                         cis_clean=cl.cis8.mean() if len(cl) else np.nan, n_all_kappa=len(g), cis_any_kappa=g.cis8.mean()))
WP = pd.DataFrame(rows); WP.to_csv(f'{TAB}/pro8_within_paper.csv', index=False)
for cs_lab, g in WP.groupby('case_set'):
    g2 = g.dropna(subset=['cis_clean'])
    lo = int((g2.cis_case < g2.cis_clean).sum()); hi = int((g2.cis_case > g2.cis_clean).sum()); tie = int((g2.cis_case == g2.cis_clean).sum())
    alltrans = int(((g2.cis_case == 0) & (g2.cis_clean == 0)).sum())
    p = stats.binomtest(lo, lo + hi).pvalue if lo + hi else np.nan
    print(f'within-paper {cs_lab}: {len(g)} case papers with kappa Pro8; {len(g2)} also have clean kappa; lower/higher/tie {lo}/{hi}/{tie} (all-trans ties {alltrans}); sign p={p:.3g}')
    print('   case papers where every kappa chain (case or not) is trans:', int((g.cis_any_kappa == 0).sum()), 'of', len(g))

pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_columns', 40)
show = ['test', 'case_set', 'method', 'n_case', 'case_papers', 'n_ctrl', 'case_val', 'ctrl_val', 'crude_OR', 'fisher_p', 'adj_OR', 'adj_lo', 'adj_hi', 'adj_p', 'pw_OR', 'pw_lo', 'pw_hi', 'pw_p', 'paper_mw_p', 'mde_OR_chains', 'mde_OR_papers', 'het_p_crude', 'het_p_z', 'het_p_3group']
print(R[[c for c in show if c in R.columns]].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print(P.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
