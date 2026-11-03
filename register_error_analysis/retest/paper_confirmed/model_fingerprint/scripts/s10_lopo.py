"""Leave-one-paper-out and drop-the-dominant-paper (10.1038/s41422-021-00487-9, Zhou Q. Cell Res 2021, 7CZP-7D0D; 17 of 67
strict chains) for the paper-confirmed (strict) estimates vs window controls.
Continuous: adjusted (year bin, resolution, chain class [+method]) chain-level difference, SE clustered by paper.
Binary (kappa cis): crude Haldane OR and Fisher p (zero cells).
Output: tables/lopo.csv"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import DEEP, TAB, load_frame, fisher

DOM = '10.1038/s41422-021-00487-9'
D = load_frame(); D = D[~D.amyloid].copy(); D['cclass'] = np.where(D.vhh == 1, 'VHH', D.ctype)
RC = pd.read_csv(f'{TAB}/rama_chain.csv.gz', dtype={'chain': str})[['pdb', 'chain', 'out_A']]
FWc = pd.read_csv(f'{TAB}/fw_computed.csv', dtype={'chain': str})[['pdb', 'chain', 'rep_rmsd']]
E = pd.read_csv(f'{TAB}/rama_entry.csv.gz')
X = D.merge(RC, on=['pdb', 'chain'], how='left').merge(FWc, on=['pdb', 'chain'], how='left')
X = X[X.grp.isin(['paper_strict', 'control'])].copy(); X['case'] = (X.grp == 'paper_strict').astype(int)
E = E[E.grp.isin(['paper_strict', 'control'])].copy(); E['case'] = (E.grp == 'paper_strict').astype(int)
rows = []


def cont(d, y, chain_level=True):
    d = d.dropna(subset=[y, 'res'])
    f = f'{y} ~ case + C(ybin_w) + res' + (' + C(cclass)' if chain_level else '') + (' + C(method)' if d.method.nunique() > 1 else '')
    m = smf.ols(f, d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.cluster)[0]})
    return m.params['case'], m.pvalues['case'], int(d.case.sum())


def binr(d, y):
    e, c = d[d.case == 1], d[d.case == 0]
    a = int(e[y].sum()); cc = int(c[y].sum())
    r = fisher(a, len(e) - a, cc, len(c) - cc); return r['OR'], r['p'], len(e)


specs = [('T3 beyond-site %allowed', X, 'out_A', cont, True), ('T4 V minus non-V gap', E, 'gap_V_A', cont, False),
         ('T5 framework RMSD', X, 'rep_rmsd', cont, True), ('T8a sheet away', X, 'v_sheet_away', cont, True),
         ('T1 kappa Pro8 cis', X[(X.ctype == 'K') & X.pro8], 'cis8', binr, None), ('T2 kappa Pro115 cis', X[(X.ctype == 'K') & X.pro115], 'cis115', binr, None)]
for lab, d0, y, fn, cl in specs:
    for meth in ['EM', 'all']:
        d = d0 if meth == 'all' else d0[d0.method == meth]
        args = (y, cl) if cl is not None else (y,)
        full = fn(d, *args)
        nod = fn(d[d.cluster != DOM], *args)
        lo = []
        for p in d[d.case == 1].cluster.unique():
            try:
                lo.append((p, *fn(d[d.cluster != p], *args)))
            except Exception:
                pass
        L = pd.DataFrame(lo, columns=['paper', 'est', 'p', 'n'])
        rows.append(dict(test=lab, method=meth, est_all=full[0], p_all=full[1], n_case=full[2], est_drop_dominant=nod[0], p_drop_dominant=nod[1],
                         n_case_drop_dominant=nod[2], lopo_min=L.est.min(), lopo_max=L.est.max(), lopo_max_p=L.p.max(),
                         lopo_worst_paper=L.loc[L.p.idxmax(), 'paper'] if len(L) else None))
R = pd.DataFrame(rows); R.to_csv(f'{TAB}/lopo.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
print(R.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
