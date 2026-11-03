"""Do CDR3/FR4 slips in VH and VL of the same Fab co-occur? Entry level, cryo-EM Fab entries (non-nanobody) containing
both a heavy and a light V chain. Main analysis drops D/E runs not confirmed by the independent superposition check."""
import sys
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
CH = chains(); R = runs()
ER = pd.read_pickle(f'{CACHE}/err_struct.pkl')
bad = set(zip(ER[ER.confirm == 'not confirmed'].pdb, ER[ER.confirm == 'not confirmed'].chain))
out = []
for label, drop in [('all D/E runs', set()), ('confirmed D/E runs only', bad)]:
    md = {}
    for r in R.itertuples():
        if r.mode in MODES and (r.pdb, r.chain) in drop: continue
        md.setdefault((r.pdb, r.chain), set()).add(r.mode)
    X = CH[(CH.method == 'EM') & (CH.vhh == 0)].copy()
    X['DE'] = [bool(md.get(k, set()) & set(MODES)) for k in zip(X.pdb, X.chain)]
    X['oth'] = [bool(md.get(k, set()) - set(MODES)) for k in zip(X.pdb, X.chain)]
    rows = []
    for p, x in X.groupby('pdb'):
        H = x[x.ctype == 'H']; L = x[x.ctype != 'H']
        if len(H) and len(L):
            rows.append(dict(pdb=p, paper=x.paper.iloc[0], year=x.year.iloc[0], res=x.res.iloc[0], H_DE=H.DE.any(), L_DE=L.DE.any(), H_oth=H.oth.any(), L_oth=L.oth.any()))
    T = pd.DataFrame(rows)
    for a, b in [('H_DE', 'L_DE'), ('H_DE', 'L_oth'), ('H_oth', 'L_DE'), ('H_oth', 'L_oth')]:
        tab = pd.crosstab(T[a], T[b]).reindex(index=[False, True], columns=[False, True], fill_value=0)
        o, p = fisher_exact(tab.values)
        pa = T[T[a]][b].mean(); pn = T[~T[a]][b].mean()
        z = T.assign(y=T[b].astype(int), x=T[a].astype(int), yb=pd.cut(T.year, [0, 2019, 2021, 2023, 2030]).astype(str))
        try:
            m = smf.logit('y ~ x + C(yb) + res', data=z).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(z.paper)[0]})
            aor, ap = float(np.exp(m.params['x'])), float(m.pvalues['x'])
        except Exception: aor, ap = np.nan, np.nan
        out.append(dict(set=label, given=a, outcome=b, n_entries=len(T), n_given=int(T[a].sum()), both=int((T[a] & T[b]).sum()),
                        rate_if_given=pa, rate_if_not=pn, fisher_OR=o, fisher_p=p, adj_OR=aor, adj_p=ap))
O = pd.DataFrame(out); O.to_csv(f'{OUT}/tables/hl_cooccurrence.csv', index=False)
pd.set_option('display.width', 220); print(O.round(4).to_string())
