"""Collect the verification tests, add antibody-level map-free reproduction and high-res any-length disagreement tests, BH-correct.
Output: tables/v08_all_tests_bh.csv, tables/v08_feature_units_evidence.csv"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
P = pd.read_csv(f'{OUT}/tables/v03_predicted_B.csv')
P = P[P.method == 'EM']
fv = P.groupby('fv_id').agg(feat=('feat', 'max'), rep=('reproduced', 'max'), pdbs=('pdbs', 'first')).reset_index()
a, n1 = int(fv[fv.feat == 1].rep.sum()), int((fv.feat == 1).sum()); b, n0 = int(fv[fv.feat == 0].rep.sum()), int((fv.feat == 0).sum())
rows = []
o, p = fisher_exact([[a, n1 - a], [b, n0 - b]])
rows.append(dict(test='map-free: EM B antibody whose run window also disagrees (>=3 res) on an ImmuneBuilder or IgFold model', error_value=f'feature {a}/{n1}',
                 control_value=f'no feature {b}/{n0}', OR=o, p=p))
D = pd.read_pickle(f'{OUT}/tables/sabr_window_all.pkl'); D = D[D.contiguous]
for grp in ['X-ray 3-5 A', 'EM 3-5 A']:
    for thr in [1, 3]:
        g = D[D.grp == grp]; y = g.n_dis_55_80 >= thr
        a, n1 = int(y[g.feat == 1].sum()), int((g.feat == 1).sum()); b, n0 = int(y[g.feat == 0].sum()), int((g.feat == 0).sum())
        o, p = fisher_exact([[a, n1 - a], [b, n0 - b]])
        rows.append(dict(test=f'{grp}: chains with >={thr} SAbR/ANARCI disagreeing residues in IMGT 55-80, by modelled 66-75 indel',
                         error_value=f'indel {a}/{n1} ({a/max(n1,1):.0%})', control_value=f'no indel {b}/{n0} ({b/n0:.1%})', OR=o, p=p))
S = pd.read_csv(f'{OUT}/tables/v06_stats.csv')
for r in S.itertuples():
    rows.append(dict(test=r.test, error_value=r.error_value, control_value=r.control_value, OR=r.OR_adj if r.OR_adj == r.OR_adj else r.OR_fisher,
                     lo=r.lo, hi=r.hi, p=r.p_adj if r.p_adj == r.p_adj else r.p_fisher))
F = pd.read_csv(f'{OUT}/tables/v07_fit_B_runs.csv')
from scipy.stats import mannwhitneyu
mw = mannwhitneyu(F[F.feat == 1].pct_in_V, F[F.feat == 0].pct_in_V)
rows.append(dict(test='EM B runs: run Q-score percentile within its V domain, feature vs no feature', error_value=f'median {F[F.feat==1].pct_in_V.median():.0f} (n={int((F.feat==1).sum())} chains)',
                 control_value=f'median {F[F.feat==0].pct_in_V.median():.0f} (n={int((F.feat==0).sum())})', p=mw.pvalue))
T = pd.DataFrame(rows); T['q_BH'] = multipletests(T.p.fillna(1), method='fdr_bh')[1]
T.to_csv(f'{OUT}/tables/v08_all_tests_bh.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 110)
print(T.round(5).to_string())
ev = pd.DataFrame([
 dict(pdb='8bw0', lab='Rak, A.', pattern='H ins 70A', run='63-70A, -1', predicted_IB='identical 9 res', predicted_IgFold='identical 9 res', hires_same_pattern='9lad_B 2.02 A: 3 res (69-70A)', verdict='artifact (exact)'),
 dict(pdb='8r4d (+8r4b)', lab='Versees, W.', pattern='VHH ins 67A-C', run='62-67C, -3', predicted_IB='identical 9 res', predicted_IgFold='identical 9 res', hires_same_pattern='6oca_C 2.11 A: 10 res (61-67C)', verdict='artifact (exact)'),
 dict(pdb='9pyh', lab='Ward, A.B.', pattern='K del 66', run='67-69, -1', predicted_IB='identical 3 res', predicted_IgFold='2 res', hires_same_pattern='9pyn_B 2.34 A, SAME antibody: 4 res (67-70)', verdict='artifact (exact)'),
 dict(pdb='8snc', lab='Zimmer, J.', pattern='VHH del 68', run='62-67, +1', predicted_IB='identical 6 res', predicted_IgFold='identical 6 res', hires_same_pattern='none in PDB', verdict='artifact (exact, map-free only)'),
 dict(pdb='9om5', lab='Zhou, T.', pattern='H del 74-75', run='63-72, -1 (3 of 6 NCS copies; other 3 agree)', predicted_IB='10 res, opposite shift', predicted_IgFold='10 res, opposite shift', hires_same_pattern='none; 8dua (same pattern, EM) clean; v10: c/g/k vs a/e/i NCS copies not register-shifted', verdict='not a register shift (conformational variant)'),
 dict(pdb='8jag (+8jam, 8jap)', lab='Li, Y.J.', pattern='H del 66-68 (CDR2 4)', run='57-69, -1', predicted_IB='1-5 res, other placement', predicted_IgFold='3 res, shifted by one', hires_same_pattern='none in PDB; v09: 88-100% closer to neighbour vs IgFold', verdict='possibly genuine (unresolved)'),
 dict(pdb='22ge', lab='Xu, Y.', pattern='rabbit H ins 69A (CDR2 9)', run='53-60, -1 (not at the indel)', predicted_IB='0 res in run (9 elsewhere in 55-80)', predicted_IgFold='0 res in run', hires_same_pattern='none', verdict='off-site; possible genuine CDR2 error'),
])
ev.to_csv(f'{OUT}/tables/v08_feature_units_evidence.csv', index=False)
print(ev.to_string())
