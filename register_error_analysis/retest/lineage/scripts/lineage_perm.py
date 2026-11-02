"""Permutation test for lineage propagation (fixes the self-exclusion bias of the stratified OR in lineage.py).

Statistic S = number of erroneous units (paper x antibody) that have >= 1 erroneous predecessor of the same class in the
given tier (different antibody, same chain type, released within the 3 years before the unit's deposition).
Null: error labels shuffled within strata of germline gene x EM/X-ray x period (2000 permutations). This keeps each
germline's and period's error count, so intrinsic susceptibility of a germline or an era does not count as propagation.
Tiers (cumulative): close relative (>= 90% V identity), same V+J gene, same germline gene (J for junction/bulge,
V for cdr12/gap). Each split into other lab / same lab.
Also run on the Trp/Phe118-exposed subset of the junction class (anchor pulled out of the core, w118_burial.csv).
"""
import json, numpy as np, pandas as pd, os
from scipy import sparse
from Bio import Align
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/lineage/tables'
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lineage_units.py')).read())  # builds U (units) and R (runs)
rng = np.random.default_rng(0)
al = Align.PairwiseAligner(mode='global', match_score=1, mismatch_score=0, open_gap_score=-1, extend_gap_score=-0.5)
WIN = np.timedelta64(3 * 365, 'D')
dep = U.dep.values.astype('datetime64[D]'); rel = U.rel.values.astype('datetime64[D]')
def pairs(gk):
    """candidate predecessor pairs (u, v): v released before u's deposition, within 3 y, same chain type, different antibody, same gene."""
    out = {k: ([], []) for k in ['close_other', 'close_same', 'vj_other', 'vj_same', 'g_other', 'g_same']}
    for g, idx in U.groupby(gk).indices.items():
        idx = np.array(idx)
        for u in idx:
            ok = idx[(rel[idx] < dep[u]) & (rel[idx] >= dep[u] - WIN) & (U.seq.values[idx] != U.seq.values[u])]
            for v in ok:
                lab = 'same' if U.lab.values[v] == U.lab.values[u] else 'other'
                out[f'g_{lab}'][0].append(u); out[f'g_{lab}'][1].append(v)
                if U.V.values[v] == U.V.values[u] and U.J.values[v] == U.J.values[u]:
                    out[f'vj_{lab}'][0].append(u); out[f'vj_{lab}'][1].append(v)
                    a, b = U.seq.values[u], U.seq.values[v]
                    if al.score(a, b) / max(len(a), len(b)) >= 0.9:
                        out[f'close_{lab}'][0].append(u); out[f'close_{lab}'][1].append(v)
    n = len(U)
    return {k: sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)) for k, (r, c) in out.items()}
def perm_test(A, y, strata, B=2000):
    S = lambda yy: int((yy & ((A @ yy.astype(float)) > 0)).sum())
    obs = S(y); null = np.empty(B)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    for b in range(B):
        yp = y.copy()
        for g in groups: yp[g] = y[g][rng.permutation(len(g))]
        null[b] = S(yp)
    return obs, null.mean(), (null >= obs).mean()
rows = []; cache = {}
for c, gk in [('junction', 'J'), ('junction_w118out', 'J'), ('bulge', 'J'), ('cdr12', 'V'), ('gap', 'V')]:
    if gk not in cache: cache[gk] = pairs(gk)
    A = cache[gk]
    y = U[f'y_{c}'].values.astype(bool)
    strata = (U[gk] + '|' + U.EM.astype(str) + '|' + U.period).values
    for tier in ['close', 'vj', 'g']:
        for lab in ['other', 'same']:
            obs, exp, p = perm_test(A[f'{tier}_{lab}'], y, strata)
            rows.append(dict(cls=c, tier={'close': 'close relative (>=90%)', 'vj': 'same V+J gene', 'g': 'same germline gene'}[tier], lab=lab,
                             err_units=int(y.sum()), observed=obs, expected=round(exp, 2), obs_minus_exp=round(obs - exp, 2), p=round(p, 4)))
    print('done', c, flush=True)
T = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(T.to_string(index=False)); T.to_csv(f'{O}/lineage_permutation.csv', index=False)
