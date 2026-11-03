"""Re-test T7 after numbering-free reclassification of the 7 feature-carrying B units (v09/v10):
 artifact (deposited model in register with an experimental or predicted reference of the same sequence): 8bw0, 8r4d, 9pyh, 8snc
 not a register shift (copies c/g/k vs uninvolved NCS copies a/e/i of the same map): 9om5
 unresolved / possibly genuine: 8jag (at the indel), 22ge (run at 53-60, off the indel)
(a) drop reclassified units; (b) move them to the control pool as correctly modelled feature carriers. Fisher exact (1-2 events).
Output: tables/v11_reclassified.csv"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
U = pd.read_pickle(f'{OUT}/tables/units_em.pkl')
U['feat'] = U.fr3start_indel.fillna(0).astype(int)
B = U[U.yB == 1]; Cl = U[U.err_any == False]
Cl = Cl[Cl.cclass.isin(set(B.cclass))]
ART = {'8bw0', '8r4d', '9pyh', '8snc', '9om5'}
rows = []
def t(name, e_feat, e_n, c_feat, c_n):
    o, p = fisher_exact([[e_feat, e_n - e_feat], [c_feat, c_n - c_feat]])
    # Haldane-corrected 95% CI (Woolf)
    a, b, c, d = e_feat + .5, e_n - e_feat + .5, c_feat + .5, c_n - c_feat + .5
    lor, se = np.log(a * d / (b * c)), np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    rows.append(dict(test=name, error_value=f'{e_feat}/{e_n} ({e_feat/e_n:.1%})', control_value=f'{c_feat}/{c_n} ({c_feat/c_n:.2%})',
                     OR_fisher=o, lo_woolf=np.exp(lor - 1.96 * se), hi_woolf=np.exp(lor + 1.96 * se), p=p))
eB = B[~B.pdb.isin(ART)]
t('original T7 (Fisher)', int(B.feat.sum()), len(B), int(Cl.feat.sum()), len(Cl))
t('(a) 5 non-shift units dropped; 8jag+22ge kept', int(eB.feat.sum()), len(eB), int(Cl.feat.sum()), len(Cl))
t('(b) 5 non-shift units moved to controls; 8jag+22ge kept', int(eB.feat.sum()), len(eB), int(Cl.feat.sum()) + 5, len(Cl) + 5)
eB2 = eB[eB.pdb != '22ge']
t('(c) as (b), and off-site 22ge dropped (only 8jag at the indel)', int(eB2.feat.sum()), len(eB2), int(Cl.feat.sum()) + 5, len(Cl) + 5)
T = pd.DataFrame(rows); T['q_BH_within'] = multipletests(T.p, method='fdr_bh')[1]
T.to_csv(f'{OUT}/tables/v11_reclassified.csv', index=False)
pd.set_option('display.width', 250); print(T.round(5).to_string())
