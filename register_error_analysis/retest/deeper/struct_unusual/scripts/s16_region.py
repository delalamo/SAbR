"""Beyond-site non-favoured backbone (Ramachandran allowed+outlier) and clashes split by framework vs CDR
(structure-based IMGT position), per unit; adjusted case-control tests. Output tables/region_geometry.csv"""
import numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf
from common import SP, TAB
from common import C as CACHE
warnings.filterwarnings('ignore')
R = pd.read_pickle(f'{CACHE}/residues.pkl'); R['chain'] = R.chain.astype(str)
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
CDR = set(range(27, 39)) | set(range(56, 66)) | set(range(105, 118))
rows = []
for _, u in s.iterrows():
    g = R[(R.pdb == u.pdb) & (R.chain == u.chain)].sort_values('idx').reset_index(drop=True)
    pos = g.spos.fillna(g.apos)
    inV = pos.notna().values & g.rama.notna().values
    hit = np.zeros(len(g), bool)
    for a, b in set_runs.get(u.set_id, []):
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    near = np.zeros(len(g), bool)
    for i in np.where(hit)[0]:
        near[max(0, i - 3):i + 4] = True
    nonfav = np.isin(g.rama.values, ['Allowed', 'OUTLIER']); clash = g.nclash.fillna(0).values > 0
    iscdr = pos.isin(CDR).values
    o = dict(unit=u.unit, pdb=u.pdb, role_cc=u.role_cc, case=int(u.role_cc == 'case'), cclass=u.cclass,
             ybin=u.ybin, res=u.res)
    for reg, m in [('fw', inV & ~near & ~iscdr), ('cdr', inV & ~near & iscdr)]:
        if m.sum() >= 5:
            o[f'nonfav_{reg}'] = 100 * nonfav[m].mean(); o[f'clash_{reg}'] = 100 * clash[m].mean()
    rows.append(o)
D = pd.DataFrame(rows)
out = []
for y in ['nonfav_fw', 'nonfav_cdr', 'clash_fw', 'clash_cdr']:
    x = D.dropna(subset=[y])
    m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
    out.append(dict(metric=y, case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(),
                    n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()), effect=m.params['case'],
                    ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case']))
O = pd.DataFrame(out); O.to_csv(f'{TAB}/region_geometry.csv', index=False)
print(O.round(4).to_string())
