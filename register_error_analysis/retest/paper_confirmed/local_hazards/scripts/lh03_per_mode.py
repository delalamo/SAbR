"""Per-mode local features from retest/deepdive (README table) on the paper-confirmed cases.
Units and controls from lh02 (analysis_units_full.pkl.gz). For each mode: case units vs control units at the same
window, restricted to the chain classes (heavy / light) of that mode's cases, as deepdive.py did. Residue features are
tested within their chain type. Binary: Fisher exact (+ one-case-per-paper version); continuous: Mann-Whitney.
Heterogeneity: paper_strict vs rest cases at the same mode (Fisher / Mann-Whitney, case-only).
Output: ../tables/per_mode_features.csv, per_mode_features.log"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from lhutil import *
log = open(f'{O}/per_mode_features.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
U = pd.read_pickle(f'{O}/analysis_units_full.pkl.gz')
U['vfam'] = U.v_gene.astype(str).str.split('-').str[0].str.split('*').str[0]
U['mouse'] = U.species.eq('mouse')
pa = lambda pos: U.pos_aa.map(lambda d: d.get(pos) if isinstance(d, dict) else None)
U['H_Q1'] = np.where(U.ctype == 'H', pa('1').eq('Q'), np.nan)
U['H_E5'] = np.where(U.ctype == 'H', pa('5').eq('E'), np.nan)
U['K_I2'] = np.where(U.ctype == 'K', pa('2').eq('I'), np.nan)
U['H_notD81'] = np.where((U.ctype == 'H') & pa('81').notna(), ~pa('81').eq('D'), np.nan)
U['IGHV5'] = U.vfam.eq('IGHV5'); U['mouse_IGHV5'] = U.IGHV5 & U.mouse
U['IGKV8'] = U.vfam.eq('IGKV8'); U['mouse_IGKV8'] = U.IGKV8 & U.mouse
U['IGHJ6'] = U.j_gene.astype(str).str.startswith('IGHJ6')
U['gap'] = (U.unmod_near.fillna(0) > 0) | ((U['mode'] == 'A') & U.nterm_missing.fillna(False).astype(bool))
U['fit_rigid'] = U.fit.eq('rigid body only')
for c in ['nterm_missing', 'fv_only', 'vhh', 'sars2', 'L1_atyp', 'L2_atyp', 'L3_atyp']:
    U[c] = U[c].map(lambda v: np.nan if pd.isna(v) else float(bool(v)))
FEAT = {'A': [('first_imgt', 'num', 'first built IMGT residue'), ('nterm_missing', 'bin', 'model starts after IMGT 3'), ('gap', 'bin', 'unbuilt nearby or N-term missing'),
              ('H_Q1', 'bin', 'heavy Q1'), ('H_E5', 'bin', 'heavy E5'), ('K_I2', 'bin', 'kappa I2')],
        'F': [('mouse', 'bin', 'mouse'), ('L1_atyp', 'bin', 'atypical CDR1 length'), ('gap', 'bin', 'unbuilt nearby')],
        'B': [('L2_atyp', 'bin', 'atypical CDR2 length'), ('vhh', 'bin', 'nanobody'), ('v_id', 'num', 'V germline identity'), ('gap', 'bin', 'unbuilt nearby')],
        'C': [('mouse_IGHV5', 'bin', 'mouse IGHV5'), ('IGHV5', 'bin', 'IGHV5 (any species)'), ('H_notD81', 'bin', 'heavy: not Asp at IMGT 81'), ('L3_atyp', 'bin', 'atypical CDR3 length'), ('gap', 'bin', 'unbuilt nearby')],
        'D+': [('gap', 'bin', 'unbuilt nearby'), ('lq_z', 'num', 'domain map-quality z'), ('win_fit', 'num', 'site Q-score (EM)'), ('L3', 'num', 'CDR3 length'),
               ('IGHJ6', 'bin', 'IGHJ6'), ('fv_only', 'bin', 'Fv-only model'), ('sars2', 'bin', 'SARS-CoV-2')],
        'D-': [('win_fit', 'num', 'site Q-score (EM)'), ('vhh', 'bin', 'nanobody'), ('v_id', 'num', 'V germline identity'), ('fit_rigid', 'bin', 'rigid-body only'), ('res', 'num', 'resolution')],
        'E': [('mouse_IGKV8', 'bin', 'mouse IGKV8'), ('mouse', 'bin', 'mouse'), ('L1_atyp', 'bin', 'atypical CDR1 length'), ('fv_only', 'bin', 'Fv-only model'), ('gap', 'bin', 'unbuilt nearby')]}
FULL = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deepdive/tables/deepdive_features.csv')
FULLMAP = {'nterm_missing': 'nterm_missing', 'first_imgt': 'first_imgt', 'mouse': 'mouse', 'L1_atyp': 'L1_atyp', 'L2_atyp': 'L2_atyp', 'vhh': 'vhh', 'v_id': 'v_id',
           'IGHV5': 'vfam=IGHV5', 'L3_atyp': 'L3_atyp', 'lq_z': 'lq_z', 'win_fit': 'win_fit', 'L3': 'L3', 'IGHJ6': 'j_gene=IGHJ6', 'fv_only': 'fv_only', 'sars2': 'sars2',
           'fit_rigid': 'fit_rigid', 'res': 'res', 'IGKV8': 'vfam=IGKV8'}
rows = []
def subset(cset, m, meth):
    d = U[(U.cset == cset) & (U['mode'] == m)]
    if meth == 'EM': d = d[d.EM]
    return d.sort_values(['pdb', 'chain']).drop_duplicates(['mode', 'paper', 'vseq'])
for m, feats in FEAT.items():
    for meth in ['EM', 'all']:
        for cset, ctl in [('paper_strict', 'ctl_paper'), ('paper_incl', 'ctl_paper'), ('full', 'ctl_full')]:
            c = subset(cset, m, meth)
            k = subset(ctl, m, meth); k = k[k.light.isin(set(c.light))] if len(c) else k
            r_ = subset('rest', m, meth)
            for f, kind, lab in feats:
                a, b, rr = c[f].dropna(), k[f].dropna(), r_[f].dropna()
                if f == 'win_fit':   # Q-score only (EM)
                    a, b, rr = c[c.EM][f].dropna(), k[k.EM][f].dropna(), r_[r_.EM][f].dropna()
                row = dict(mode=m, method=meth, set=cset, feature=lab, n_case=len(a), n_ctrl=len(b), case_papers=c.loc[a.index, 'paper'].nunique() if len(a) else 0)
                if len(a) == 0: rows.append(row); continue
                if kind == 'num':
                    row.update(case=a.median(), ctrl=b.median(), case_mean=a.mean(), ctrl_mean=b.mean(), p=mannwhitneyu(a, b).pvalue if len(a) >= 2 else np.nan)
                    if cset == 'paper_strict' and len(rr) >= 2 and len(a) >= 2: row['het_p_vs_rest'] = mannwhitneyu(a, rr).pvalue; row['rest_case'] = rr.median()
                else:
                    a, b = a.astype(bool), b.astype(bool)
                    t = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
                    OR, p = fisher_exact(t)
                    t2 = Table2x2(np.array(t, float) + 0.5); lo, hi = t2.oddsratio_confint()
                    row.update(case=100 * a.mean(), ctrl=100 * b.mean(), OR=OR, lo=lo, hi=hi, p=p, min_det_OR80=min_detectable_or(len(a), b.mean(), len(b)))
                    # one case per paper
                    c1 = c.loc[a.index].sort_values(['pdb', 'chain']).drop_duplicates('paper')[f].astype(bool)
                    row.update(opp_n=len(c1), opp_case=100 * c1.mean(), opp_p=fisher_exact([[c1.sum(), len(c1) - c1.sum()], [b.sum(), len(b) - b.sum()]])[1])
                    if cset == 'paper_strict' and len(rr):
                        rr = rr.astype(bool)
                        row['rest_case'] = 100 * rr.mean(); row['het_p_vs_rest'] = fisher_exact([[a.sum(), len(a) - a.sum()], [rr.sum(), len(rr) - rr.sum()]])[1]
                if cset == 'full':
                    ff = FULL[(FULL['mode'] == m) & (FULL.feature == FULLMAP.get(f, '#'))]
                    if len(ff): row['deepdive_case'] = ff.case.iloc[0]; row['deepdive_ctrl'] = ff.control.iloc[0]
                rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(f'{O}/per_mode_features.csv', index=False)
cols = ['feature', 'set', 'n_case', 'case_papers', 'n_ctrl', 'case', 'ctrl', 'OR', 'lo', 'hi', 'p', 'opp_n', 'opp_case', 'opp_p', 'min_det_OR80', 'rest_case', 'het_p_vs_rest', 'deepdive_case', 'deepdive_ctrl']
for m in FEAT:
    for meth in ['EM', 'all']:
        P(f'\n=== mode {m} ({meth})')
        t = T[(T['mode'] == m) & (T.method == meth)]
        P(t[[c for c in cols if c in t]].round(3).to_string(index=False))
log.close()
