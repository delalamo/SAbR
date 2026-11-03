"""(1) Adjust the beyond-site effect for recorded refinement software (Phenix, Coot, ISOLDE/Rosetta) and starting
model / fitting protocol. (2) Recompute framework-only and CDR-only beyond-site rates from the independent table."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
U = pd.read_csv(f'{OUT}/v_unit_metrics.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
sw = [c for c in d.columns if c.startswith('sw_')]
U = U.merge(d[['pdb', 'chain', 'irm', 'fit'] + sw], on=['pdb', 'chain'], how='left')
U['sw_refine_other'] = ((U.sw_ISOLDE == 1) | (U.sw_Rosetta == 1)).astype(int)
U['lab'] = U.last_author.fillna('NA')
rows = []
for y in ['allowed_out', 'clash_out', 'rota_out']:
    for lab, f in [('base', ''), ('+ Phenix, Coot, ISOLDE/Rosetta recorded', ' + sw_Phenix + sw_Coot + sw_refine_other'),
                   ('+ software + starting model + fitting', ' + sw_Phenix + sw_Coot + sw_refine_other + C(irm) + C(fit)')]:
        x = U.dropna(subset=[y])
        m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res{f}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.lab)[0]})
        rows.append(dict(metric=y, model=lab, n=len(x), effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0],
                         ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case']))
O = pd.DataFrame(rows)
# region split from the independent residue table
R = pd.read_pickle(f'{SP}/deeper/cache/verify_beyond/res_all.pkl')
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
G = {k: g.sort_values('idx').reset_index(drop=True) for k, g in R.groupby(['pdb', 'chain'])}
CDR = set(range(27, 39)) | set(range(56, 66)) | set(range(105, 118))
rr = []
for _, u in s.iterrows():
    g = G.get((u.pdb, u.chain))
    if g is None: continue
    inV = (g.apos.notna() | g.spos.notna()).values
    hit = np.zeros(len(g), bool)
    for a, b in set_runs.get(u.set_id, []):
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    hit &= inV
    near = np.zeros(len(g), bool)
    for i in np.where(hit)[0]:
        near[max(0, i - 3):i + 4] = True
    pos = g.spos.fillna(g.apos)
    iscdr = pos.isin(CDR).values
    ev = g.rama.notna().values
    nonfav = np.isin(g.rama.values, ['Allowed', 'OUTLIER'])
    o = dict(pdb=u.pdb, case=int(u.role_cc == 'case'), cclass=u.cclass, ybin=u.ybin, res=u.res, lab=u.last_author)
    for reg, m in [('fw', inV & ev & ~near & ~iscdr), ('cdr', inV & ev & ~near & iscdr)]:
        if m.sum() >= 5:
            o[f'nonfav_{reg}'] = 100 * nonfav[m].mean()
    rr.append(o)
D = pd.DataFrame(rr)
for y in ['nonfav_fw', 'nonfav_cdr']:
    x = D.dropna(subset=[y])
    m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.lab.fillna('NA'))[0]})
    O = pd.concat([O, pd.DataFrame([dict(metric=y, model=f'region (case {x[x.case==1][y].mean():.2f} vs ctrl {x[x.case==0][y].mean():.2f}), cluster lab',
                                         n=len(x), effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0],
                                         ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case'])])])
O.to_csv(f'{OUT}/v_software_region.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); print(O.round(5).to_string())
