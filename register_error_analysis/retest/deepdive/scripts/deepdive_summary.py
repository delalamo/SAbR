"""How much of each mode is covered by (1) missing density next to the error, (2) an antibody atypical at that
location, (3) poor local map fit; cases vs correct controls at the same location (EM). Also re-tests the non-gap
errors (excluding runs whose origin was 'chain end / next to an unbuilt loop')."""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/deepdive/tables'
D = pd.read_pickle(f'{SP}/unexpl/deepdive_units_full.pkl'); D = D[D.method == 'EM'].copy()
R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv'); gapchains = set(zip(R[R.origin.str.startswith('3')].pdb, R[R.origin.str.startswith('3')].chain))
ADJ = {'A': None, 'F': 'L1_atyp', 'B': 'L2_atyp', 'C': 'L3_atyp', 'D+': 'L3_atyp', 'D-': 'L3_atyp', 'E': 'L3_atyp'}
rows = []
for m, g in D.groupby('mode'):
    q25 = g[~g.case].win_fit_pct.quantile(0.25)
    g = g.assign(gap=(g.unmod_near.fillna(0) > 0) | (g.nterm_missing.fillna(False).astype(bool) if m == 'A' else False),
                 atyp=(g.vhh.astype(bool) | (g.v_id < 0.80) | g.species.ne('human') | (g[ADJ[m]].fillna(False).astype(bool) if ADJ[m] else False)),
                 poorfit=g.win_fit_pct < q25)
    g['any'] = g.gap | g.atyp | g.poorfit
    c, k = g[g.case], g[~g.case]
    for f in ['gap', 'atyp', 'poorfit', 'any']:
        p = fisher_exact([[c[f].sum(), len(c) - c[f].sum()], [k[f].sum(), len(k) - k[f].sum()]])[1]
        rows.append(dict(mode=m, feature=f, case_pct=round(100 * c[f].mean()), control_pct=round(100 * k[f].mean()), n_case=len(c), p=p))
    # non-gap errors only
    ng = c[np.array([(p_, ch) not in gapchains for p_, ch in zip(c.pdb, c.chain)], dtype=bool) & ~c.gap.values]
    for f in ['atyp', 'poorfit']:
        if len(ng) >= 3:
            p = fisher_exact([[ng[f].sum(), len(ng) - ng[f].sum()], [k[f].sum(), len(k) - k[f].sum()]])[1]
            rows.append(dict(mode=m, feature=f'{f} (errors not next to a gap)', case_pct=round(100 * ng[f].mean()), control_pct=round(100 * k[f].mean()), n_case=len(ng), p=p))
T = pd.DataFrame(rows); pd.set_option('display.width', 200); print(T.to_string(index=False)); T.to_csv(f'{O}/deepdive_summary.csv', index=False)
