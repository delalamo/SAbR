"""EM 3-5 A units vs high-resolution (<=2.5 A X-ray) chains with the SAME FR3 indel pattern: is the rate of SAbR/ANARCI
FR3 disagreement (>=3 residues in IMGT 66-104) higher in EM models than in high-resolution crystal structures?
EM side: one row per EM unit with an atypical FR3 (any chain of the unit with a run overlapping 66-104, incl. copied /
artifact runs = 'flagged'; first-occurrence C run = 'C_first'). Reference: tables/hires_detector_rate.csv (v06).
Also the expected number of flagged EM units if EM models behaved like high-resolution crystal structures.
Output: tables/em_vs_hires_matched.csv
"""
import numpy as np, pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_confounding'
U = pd.read_csv(f'{OUT}/tables/units_em_flags.csv.gz')
U['first_modes'] = U.first_modes.fillna('')
H = pd.read_csv(f'{OUT}/tables/hires_detector_rate.csv')
H = H[H.built_same == True].assign(flag=lambda d: d.n_dis >= 3)
H['ctk'] = np.where(H.ct == 'H', 'H', H.ct)
U['ctk'] = np.where(U.cclass.isin(['H', 'VHH']), 'H', U.cclass)
A = U[U.fr3_atyp == 1].copy()
A['flag'] = A.any_run_in_FR3win.astype(bool)
A['C_first'] = A.first_modes.str.contains('C_FR3')
hr = H.groupby(['ctk', 'pattern']).agg(h_n=('flag', 'size'), h_flag=('flag', 'sum')).reset_index()
M = A.merge(hr, left_on=['ctk', 'fr3_pattern'], right_on=['ctk', 'pattern'], how='left')
G = M.groupby(['ctk', 'fr3_pattern']).agg(em_units=('flag', 'size'), em_flagged=('flag', 'sum'), em_C_first=('C_first', 'sum'),
                                         h_n=('h_n', 'first'), h_flag=('h_flag', 'first')).reset_index().sort_values('em_units', ascending=False)
G.to_csv(f'{OUT}/tables/em_vs_hires_matched.csv', index=False)
pd.set_option('display.width', 200)
print(G.to_string())
k = G[G.h_n > 0]
print('patterns with high-res reference: EM units', k.em_units.sum(), 'flagged', k.em_flagged.sum(), 'C_first', k.em_C_first.sum())
exp = (k.em_units * k.h_flag / k.h_n).sum()
print('expected flagged EM units if they behaved like high-res crystals: %.1f  (observed %d)' % (exp, k.em_flagged.sum()))
tabs = []
for r in k.itertuples():
    t = np.array([[r.em_flagged, r.em_units - r.em_flagged], [r.h_flag, r.h_n - r.h_flag]], float)
    if t[:, 0].sum() > 0 and t[:, 1].sum() > 0: tabs.append(t + 0.5)
if tabs:
    st = StratifiedTable(tabs)
    print('pattern-matched MH OR (EM flagged vs high-res flagged): %.2f (%.2f-%.2f), p=%.3g, informative strata %d' % (st.oddsratio_pooled, *st.oddsratio_pooled_confint(), st.test_null_odds().pvalue, len(tabs)))
print('EM FR3-atypical units flagged overall: %d/%d = %.1f%%; high-res FR3-atypical chains flagged: %d/%d = %.1f%%; high-res FR3-typical: %d/%d' % (
    A.flag.sum(), len(A), 100 * A.flag.mean(), H[H.atyp].flag.sum(), H.atyp.sum(), 100 * H[H.atyp].flag.mean(), H[~H.atyp].flag.sum(), (~H.atyp).sum()))
