"""Mode specificity of 'model ends inside FR4' and of SEQRES FR4 truncation (EM units, first-occurrence errors vs clean)."""
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings('ignore')
D_ = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN2_trunc_FR4_modesDE_confounding'
U = pd.read_csv(f'{D_}/tables/unit_features.csv.gz', low_memory=False)
U['modes_u'] = U.modes_u.fillna('')
U['trunc_true'] = (U.seq_fr4_trunc & ~(U.seq_tail_after_v > 5)).astype(int)
U['model_end_fr4'] = (U.mod_last_anarci < 125).astype(float).where(U.mod_last_anarci.notna())
U['model_start_gt3'] = (U.mod_first_anarci > 3).astype(float).where(U.mod_first_anarci.notna())
EM = U.method_u.eq('EM'); ctrl = EM & U.clean
rows = []
for m in ['A_FR1Nterm', 'F_CDR1', 'B_CDR2FR3', 'C_FR3', 'Dplus_CDR3short', 'Dminus_CDR3long', 'D0', 'E_FR4bulge']:
    cs = EM & U.err_first_u & U.modes_u.str.contains(m); cc = ctrl & U.ctype.isin(U[cs].ctype.unique())
    for f in ['trunc_true', 'model_end_fr4', 'model_start_gt3']:
        a = U[cs][f].dropna(); b = U[cc][f].dropna()
        o, p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        rows.append(dict(mode=m, feature=f, n_err=len(a), err_pct=round(100 * a.mean(), 1), n_ctrl=len(b), ctrl_pct=round(100 * b.mean(), 2), OR=round(o, 1), p=p))
T = pd.DataFrame(rows); T.to_csv(f'{D_}/tables/v04_mode_specificity.csv', index=False)
pd.set_option('display.width', 200); print(T.to_string())
