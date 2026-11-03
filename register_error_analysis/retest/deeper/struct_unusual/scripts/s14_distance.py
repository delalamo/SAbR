"""Is the excess of non-favoured backbone geometry in error chains concentrated near the error site or spread over
the domain? Ramachandran 'allowed'+'outlier' and clash rates by sequence distance from the site (case runs; the
matched case's runs for controls). Output tables/geometry_by_distance.csv"""
import numpy as np, pandas as pd
from common import SP, TAB
from common import C as CACHE
R = pd.read_pickle(f'{CACHE}/residues.pkl'); R['chain'] = R.chain.astype(str)
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
rows = []
for _, u in s.iterrows():
    g = R[(R.pdb == u.pdb) & (R.chain == u.chain)].sort_values('idx').reset_index(drop=True)
    inV = (g.apos.notna() | g.spos.notna()).values
    hit = np.zeros(len(g), bool)
    for a, b in set_runs.get(u.set_id, []):
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    if not hit.any():
        continue
    hi = np.where(hit)[0]
    dist = np.array([np.min(np.abs(hi - i)) for i in range(len(g))])
    bins = pd.cut(dist, [-1, 0, 3, 10, 20, 200], labels=['run', '1-3', '4-10', '11-20', '>20'])
    nonfav = np.isin(g.rama.values, ['Allowed', 'OUTLIER'])
    clash = g.nclash.fillna(0).values > 0
    for bname in bins.categories:
        m = (bins == bname) & inV & g.rama.notna().values
        if m.sum():
            rows.append(dict(role_cc=u.role_cc, unit=u.unit, bin=bname, n=int(m.sum()),
                             nonfav=nonfav[m].sum(), clash=clash[m].sum()))
D = pd.DataFrame(rows)
T = D.groupby(['bin', 'role_cc'])[['n', 'nonfav', 'clash']].sum()
T['nonfav_pct'] = 100 * T.nonfav / T.n; T['clash_pct'] = 100 * T.clash / T.n
T = T.reset_index()
T['bin'] = pd.Categorical(T['bin'], ['run', '1-3', '4-10', '11-20', '>20']); T = T.sort_values(['bin', 'role_cc'])
T.to_csv(f'{TAB}/geometry_by_distance.csv', index=False)
print(T.round(2).to_string())
