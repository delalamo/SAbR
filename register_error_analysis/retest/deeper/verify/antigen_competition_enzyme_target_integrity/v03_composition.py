# Composition checks: antibody role ("tool" claim), error origin categories, V-chain counts, unit-level leave-one-paper-out
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore'); pd.set_option('display.width', 250)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
AC = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_integrity'
U = pd.read_csv(f'{OUT}/units_em_enzyme.csv.gz')
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['unit', 'role', 'aux_reagent', 'method', 'pdb'])
r = ch[ch.method == 'EM'].groupby('unit').agg(role=('role', lambda s: s.mode().iat[0]), aux=('aux_reagent', 'max')).reset_index()
U = U.merge(r, on='unit', how='left')
U['grp'] = np.where(U.agcls == 'enzyme', 'host enzyme', np.where(U.agcls == 'viral other', 'viral other', np.where(U.agcls.isin(['channel/transporter', 'GPCR']), 'transporter/GPCR', 'other')))
print('== role (text rule) by contacted-antigen group, EM units: share and error rate')
print(pd.crosstab(U.grp, U.role, normalize='index').round(2).to_string())
print(U.groupby(['grp', 'role']).y.agg(['sum', 'size']).to_string())
print('\n== aux_reagent flag share by group'); print(U.groupby('grp').aux.mean().round(2).to_string())
# error origin categories
m = pd.read_csv(f'{AC}/focal_entries_antigen_features.csv.gz', usecols=['pdb', 'method', 'tgt_cls', 'y'])
fo = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv').merge(m[['pdb', 'tgt_cls']], on='pdb', how='inner')
fo = fo[fo.method == 'EM']; fo['grp'] = np.where(fo.tgt_cls == 'enzyme', 'host enzyme', np.where(fo.tgt_cls == 'viral other', 'viral other', 'other'))
print('\n== origin category of first-occurrence runs (entry-level, EM): share of entries with an origin-3 (chain end / next to unbuilt loop) run')
eo = fo.groupby(['grp', 'pdb']).origin.agg(lambda s: int(s.str.startswith('3').any())).reset_index()
print(eo.groupby('grp').origin.agg(['sum', 'size']).to_string())
a = eo[eo.grp == 'host enzyme'].origin; b = eo[eo.grp == 'other'].origin
print('  Fisher host enzyme vs other:', stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]))
# V chains per entry
e = pd.read_csv(f'{SP}/deeper/entries.csv', usecols=['pdb', 'n_V_chains'])
mm = m[m.method == 'EM'].merge(e, on='pdb'); mm['grp'] = np.where(mm.tgt_cls == 'enzyme', 'host enzyme', np.where(mm.tgt_cls == 'viral other', 'viral other', 'other'))
print('\n== V chains per EM entry'); print(mm.groupby('grp').n_V_chains.describe().round(2).to_string())
# unit-level leave-one-paper-out
U['ycat'] = U.dyear.clip(lower=2017).astype(str)
errp = U[(U.host_enz == 1) & (U.y == 1)].paper.unique(); res = []
for p in errp:
    d = U[U.paper != p]
    mod = smf.glm('y ~ host_enz + C(ycat) + res', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
    res.append((p, np.exp(mod.params['host_enz']), mod.pvalues['host_enz']))
R = pd.DataFrame(res, columns=['paper', 'OR', 'p']); print('\n== unit-level host-enzyme leave-one-paper-out: OR', R.OR.min().round(2), '-', R.OR.max().round(2), 'max p', R.p.max().round(4))
# drop the 3 papers with most error units
top = U[(U.host_enz == 1) & (U.y == 1)].paper.value_counts()
print('  error units per enzyme error paper:', top.to_dict())
