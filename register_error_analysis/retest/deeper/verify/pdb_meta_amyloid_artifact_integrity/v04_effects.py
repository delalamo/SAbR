"""Effects of the fibril exclusion: antibody-alone test (Fisher + year/res adjusted), mode composition,
per-year EM error-entry rate, run-length separation, V-gene tallies, and the per-unit lambda discrepancy."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
PM = '/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
AMY = {'6ic3', '6z1o', '7nsl', '9els', '9faa', '9fab', '9fac', '9liv', '9liw', '9lix', '9liy', '9lj0', '9lja', '9ljb', '9v91'}
pd.set_option('display.width', 250)

# --- antibody-alone (investigator feature) before/after
d = pd.read_csv(f'{PM}/em_entry_analysis_frame_with_amyloid.csv.gz', usecols=['pdb', 'y', 'ab_alone', 'ybin', 'res', 'l2nV', 'cluster'])
d['amy'] = d.pdb.isin(AMY)
print('frame: n', len(d), 'err', d.y.sum(), 'clean', (d.y == 0).sum(), 'amyloid in frame', d.amy.sum())
for lab, x in [('with fibrils', d), ('without fibrils', d[~d.amy])]:
    t = pd.crosstab(x.y, x.ab_alone)
    o, p = stats.fisher_exact(t.values[::-1, ::-1])
    m = smf.glm('y ~ ab_alone + C(ybin) + res + l2nV', data=x, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.cluster)[0]})
    print(f'{lab}: err ab_alone {t.loc[1,1]}/{t.loc[1].sum()} vs clean {t.loc[0,1]}/{t.loc[0].sum()}; Fisher OR {o:.2f} p {p:.2g}; adj OR {np.exp(m.params.ab_alone):.2f} p {m.pvalues.ab_alone:.2g}')
# true antibody-alone after correcting the fusion misclassification (9zxd/9zxc nanobody-AT1R fusion; 9d7u-x GFP/transporter fusion)
fus = {'9zxd', '9zxc', '9d7u', '9d7v', '9d7w', '9d7x'}
x = d[~d.amy].copy(); x.loc[x.pdb.isin(fus), 'ab_alone'] = 0
print('without fibrils, fusions reclassified: err', int(x[x.y == 1].ab_alone.sum()), '/', int(x.y.sum()), ' clean', int(x[x.y == 0].ab_alone.sum()), '/', int((x.y == 0).sum()))

# --- entries: EM error-entry rate by deposition year with and without fibrils
ent = pd.read_csv(f'{SP}/entries.csv')
ent['amy'] = ent.pdb.isin(AMY); ent['dy'] = pd.to_datetime(ent.deposit).dt.year
em = ent[(ent.method == 'EM') & (ent.err_first | ~ent.err_any)]
tab = em.groupby('dy').agg(n=('pdb', 'size'), err=('err_first', 'sum'), amy=('amy', 'sum'))
tab['rate_with'] = (tab.err / tab.n * 100).round(2)
tab['rate_without'] = ((tab.err - tab.amy) / (tab.n - tab.amy) * 100).round(2)
print(tab[tab.index >= 2016].to_string())
print('fibril share of EM error entries deposited >=2024:', int(tab.loc[tab.index >= 2024, 'amy'].sum()), '/', int(tab.loc[tab.index >= 2024, 'err'].sum()))

# --- mode composition: units per mode (all methods) and fibril share
ch = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'chain', 'ctype', 'method', 'unit', 'err_first', 'err_any', 'err_modes', 'v_gene', 'L1'])
ch['amy'] = ch.pdb.isin(AMY)
e = ch[ch.err_first].copy()
e['modes'] = e.err_modes.fillna('').str.split(r'[|;,]')
e = e.explode('modes').reset_index(drop=True); e['modes'] = e.modes.str.strip()
mc = e.groupby('modes').agg(units=('unit', 'nunique')).join(e[e.amy].groupby('modes').agg(amy_units=('unit', 'nunique'))).fillna(0)
mc['amy_pct'] = (mc.amy_units / mc.units * 100).round(1)
print(mc.to_string())
runs = pd.read_csv(f'{SP}/runs_first_occurrence.csv'); runs['amy'] = runs.pdb.isin(AMY)
print('runs by mode (fibril/total):', runs.groupby('mode').amy.agg(['sum', 'size']).to_dict('index'))
print('run length: fibril min', runs[runs.amy].length.min(), '; non-fibril max', runs[~runs.amy].length.max(),
      '; non-fibril runs >= 63:', (runs[~runs.amy].length >= 63).sum(), 'of', (~runs.amy).sum())
print('run |delta|: fibril median', runs[runs.amy].delta.abs().median(), ' non-fibril median', runs[~runs.amy].delta.abs().median(),
      ' non-fibril |delta|>=4:', (runs[~runs.amy].delta.abs() >= 4).sum())

# --- V-gene tallies
a = ch[ch.amy]
print('IGLV1-44 entries:', sorted(a[a.v_gene.str.startswith('IGLV1-44')].pdb.unique()), ' units:', a[a.v_gene.str.startswith('IGLV1-44')].unit.nunique())

# --- per-unit lambda discrepancy: unit flag by max vs first chain
lam = ch[(ch.ctype == 'L') & ch.L1.isin([8, 9])]
for agg in ['max', 'first', 'all']:
    for amy in [True, False]:
        x = lam if amy else lam[~lam.amy]
        f = {'max': 'max', 'first': 'first', 'all': 'min'}[agg]
        u = x.groupby('unit').agg(y=('err_first', f), L1=('L1', 'first'))
        r8 = u[u.L1 == 8].y; r9 = u[u.L1 == 9].y
        print(f'unit flag={agg:5s} amyloid={"with" if amy else "without"}: L1=8 {r8.mean()*100:.1f}% ({int(r8.sum())}/{len(r8)}) vs L1=9 {r9.mean()*100:.1f}% ({int(r9.sum())}/{len(r9)})')
mixed = lam.groupby('unit').err_first.agg(['min', 'max'])
mixed = mixed[(mixed['min'] != mixed['max'])]
print('lambda L1 8/9 units with mixed chain flags:', len(mixed))
print(lam[lam.unit.isin(mixed.index)][['pdb', 'chain', 'L1', 'err_first', 'method']].to_string(index=False))
