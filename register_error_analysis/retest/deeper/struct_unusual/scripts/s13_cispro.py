"""Canonical cis-Pro at kappa CDR-L3 position 95 (Kabat; IMGT 115) in 9-residue L3s (North L3-9-cis7 class):
is the cis peptide preserved in deposited models? Case vs control, with starting-model type / fitting / software
breakdown and ImmuneBuilder predicted state. Output tables/cispro95.csv"""
import numpy as np, pandas as pd, os
import statsmodels.formula.api as smf
from scipy import stats
from common import SP, TAB, dihedral
from common import C as CACHE
R = pd.read_pickle(f'{CACHE}/residues.pkl'); R['chain'] = R.chain.astype(str)
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
U = pd.read_csv(f'{TAB}/unit_metrics.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
sw = [c for c in d.columns if c.startswith('sw_')]
s = s.merge(U[['unit', 'pdb', 'modes']], on=['unit', 'pdb'], how='left').merge(
    d[['pdb', 'chain'] + sw + ['title']], on=['pdb', 'chain'], how='left')
out = []
for _, u in s[s.cclass == 'K'].iterrows():
    g = R[(R.pdb == u.pdb) & (R.chain == u.chain)].sort_values('idx').reset_index(drop=True)
    ii = g.index[(g.apos == 115) & (g.ains == '')]
    if len(ii) == 0 or ii[0] == 0 or g.rn[ii[0]] != 'PRO':
        continue
    i = ii[0]
    om = g.omega[i - 1]
    out.append(dict(pdb=u.pdb, chain=u.chain, role_cc=u.role_cc, case=int(u.role_cc == 'case'), L3=u.L3, res=u.res,
                    ybin=u.ybin, year=u.year, irm=u.irm, fit=u.fit, modes=u.modes, omega=om,
                    cis=float(abs(om) < 30) if np.isfinite(om) else np.nan,
                    q115=g.q[i], **{k: u[k] for k in sw}))
o = pd.DataFrame(out)
o = o[(o.L3 == 9) & o.cis.notna()]
# ImmuneBuilder predicted state for the same chains (controls c_*, cases x_* or earlier predictions)
o.to_csv(f'{TAB}/cispro95.csv', index=False)
print(o.groupby('role_cc').cis.agg(['mean', 'sum', 'size']))
a = o.groupby('case').cis.agg(['sum', 'size'])
print('Fisher', stats.fisher_exact([[a.loc[1, 'sum'], a.loc[1, 'size'] - a.loc[1, 'sum']],
                                    [a.loc[0, 'sum'], a.loc[0, 'size'] - a.loc[0, 'sum']]]))
m = smf.logit('cis ~ case + C(ybin) + res', o).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(o.pdb)[0]})
print('adjusted OR', np.exp(m.params['case']).round(3), np.exp(m.conf_int().loc['case']).round(3).tolist(), m.pvalues['case'])
print(o.groupby(['role_cc', 'irm']).cis.agg(['mean', 'size']))
print(o.groupby(['role_cc', 'fit']).cis.agg(['mean', 'size']))
for k in sw:
    print(k, o.groupby(['role_cc', k]).cis.agg(['mean', 'size']).to_dict())
