"""Extra confounder checks on the fixed, amyloid-free entry table: antibody role (auxiliary reagent / fiducial vs
antibody of interest), nanobody content, continuous year, V share of the model; and the scale of the amyloid issue
in the project's whole error set."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_integrity'
E = pd.read_csv(f'{OUT}/entry_metrics_verify2.csv', index_col=0)
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
dd = d[d.pdb.isin(E.index)]
r = dd.groupby('pdb').agg(aux=('aux_reagent', 'max'), vhh=('vhh', 'max'), nv=('chain', 'size'),
                          role_any=('role', lambda x: ';'.join(sorted(set(map(str, x))))))
E = E.join(r)
E['aux'] = E.aux.astype(float); E['vhh'] = E.vhh.astype(float)
E['vshare'] = E.V_all_n / (E.V_all_n + E.rest_n)
G = E[~E.amyloid & (E.rest_n > 0)].copy()
print(G.groupby('case')[['aux', 'vhh', 'vshare', 'nv']].mean().round(3))
print(G.role_any.value_counts().head(8))
for y in ['gap_V_all_allowed', 'gap_V_m10_allowed', 'gap_V_errfree_chains_allowed', 'gap_V_all_clash']:
    for adj in ['', ' + aux', ' + aux + vhh + vshare', ' + year + I(year**2)']:
        x = G.dropna(subset=[y, 'aux', 'vhh', 'vshare'])
        m = smf.ols(f'{y} ~ case + C(ybin) + res{adj}', x).fit(cov_type='HC1')
        print(f'{y:30s} adj[{adj:24s}] effect {m.params.case:.3f} p {m.pvalues.case:.3g}')
    for a in [0, 1]:
        x = G[G.aux == a].dropna(subset=[y])
        m = smf.ols(f'{y} ~ case + C(ybin) + res', x).fit(cov_type='HC1')
        print(f'   stratum aux={a}: n={int(x.case.sum())}/{int((1-x.case).sum())} effect {m.params.case:.3f} p {m.pvalues.case:.3g}  case mean {x[x.case==1][y].mean():.2f} ctrl {x[x.case==0][y].mean():.2f}')
# scale of the amyloid issue in the whole project
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
ent['amyloid'] = ent.title.str.contains('amyloid|fibril', case=False, na=False)
print('amyloid/fibril entries by method x err_first:\n', pd.crosstab([ent.method, ent.amyloid], ent.err_first))
