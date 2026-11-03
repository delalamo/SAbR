"""Re-derive the 'beyond the site' geometry metrics from the investigator's per-residue table with stricter exclusions,
and test where (sequence and 3-D distance from the site) the excess sits.

Exclusion levels for the 'outside' region (all also exclude the first-occurrence run +-3, as in the original):
  E0  original definition (run +-3); denominator = all V residues outside (original) -> reproduces unit_metrics
  E0r same, denominator = residues with a Ramachandran classification
  E1  + every run in runs_all for that chain (copied / artifact runs) +-3
  E2  + every residue where ANARCI and SAbR IMGT positions disagree (or one is missing) +-3
  E3  + residues within 3 of an internal chain break or of the first/last built V residue
  E4  E3 + residues within 12 A (CA-CA) of any site residue (3-D exclusion)
  E5  E3 restricted to framework positions (structure-based IMGT, non-CDR)
Distance profiles: per-unit rates of non-favoured Ramachandran by sequence distance and by CA distance to the site.
Output tables/v02_*.csv
"""
import pickle, warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SU = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
CU = f'{SP}/deeper/cache/struct_unusual'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_confounding'
CDR = set(range(27, 39)) | set(range(56, 66)) | set(range(105, 118))

R = pd.read_pickle(f'{CU}/residues.pkl'); R['chain'] = R.chain.astype(str)
V = pickle.load(open(f'{CU}/vchains.pkl', 'rb'))
s = pd.read_csv(f'{SU}/sample.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain', 'last_author', 'paper', 'sars2'])
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
runs_all = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
case_origin = {k: ';'.join(sorted(set(g.origin.str.split(' ').str[0]))) for k, g in runs.groupby(['pdb', 'chain'])}
all_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs_all.groupby(['pdb', 'chain'])}
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
set_origin = {u.set_id: case_origin.get((u.pdb, u.chain), '') for _, u in s[s.role_cc == 'case'].iterrows()}
G = {k: g.sort_values('idx').reset_index(drop=True) for k, g in R.groupby(['pdb', 'chain'])}


def hitmask(g, ranges):
    hit = np.zeros(len(g), bool)
    for a, b in ranges:
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    return hit


def dilate(m, k):
    out = m.copy()
    for i in np.where(m)[0]:
        out[max(0, i - k):i + k + 1] = True
    return out


rows, drows = [], []
for _, u in s.iterrows():
    k = (u.pdb, u.chain)
    if k not in G:
        continue
    g = G[k]
    n = len(g)
    inV = (g.apos.notna() | g.spos.notna()).values
    site = hitmask(g, set_runs.get(u.set_id, [])) & inV
    if not site.any():
        continue
    ex0 = dilate(site, 3)
    ex1 = ex0 | dilate(hitmask(g, all_runs.get(k, [])) & inV, 3)
    dis = inV & ((g.apos.values != g.spos.values) | (g.ains.fillna('').values != g.sins.fillna('').values)
                 | g.apos.isna().values | g.spos.isna().values)
    ex2 = ex1 | dilate(dis, 3)
    brk = np.zeros(n, bool)
    vi = np.where(inV)[0]
    brk[vi.min()] = brk[vi.max()] = True
    gap = (g.prev_si_gap.fillna(0).values > 0)
    brk |= gap | np.roll(gap, -1)  # residue after and before a break
    ex3 = ex2 | dilate(brk, 3)
    # 3-D distance to site
    ca = np.array([V[k]['res'][i]['atoms'].get('CA', [np.nan] * 3) for i in g.idx], float)
    sca = ca[site]
    dd = np.sqrt(((ca[:, None, :] - sca[None, :, :]) ** 2).sum(-1))
    dmin = np.nanmin(np.where(np.isnan(dd), np.inf, dd), axis=1)
    ex4 = ex3 | (dmin < 12)
    pos = g.spos.fillna(g.apos)
    fw = ~pos.isin(CDR).values
    rama = g.rama.values
    has_r = pd.notna(rama) & (rama != None)
    allowed = rama == 'Allowed'; nonfav = np.isin(rama, ['Allowed', 'OUTLIER'])
    clash = g.nclash.fillna(0).values > 0
    rota_out = g.rota.values == 'OUTLIER'
    has_rota = pd.notna(g.rota.values) & (g.rota.values != None)
    o = dict(unit=u.unit, pdb=u.pdb, chain=u.chain, case=int(u.role_cc == 'case'), set_id=u.set_id, cclass=u.cclass,
             ybin=u.ybin, res=u.res, year=u.year, origin=set_origin.get(u.set_id, ''),
             n_dis_outside=int((dis & ~ex1).sum()), n_breaks=int(gap[inV].sum()))
    for lab, m in [('E0', inV & ~ex0), ('E0r', inV & ~ex0 & has_r), ('E1', inV & ~ex1 & has_r),
                   ('E2', inV & ~ex2 & has_r), ('E3', inV & ~ex3 & has_r), ('E4', inV & ~ex4 & has_r),
                   ('E5', inV & ~ex3 & has_r & fw)]:
        nn = int(m.sum())
        o[f'n_{lab}'] = nn
        if nn >= 10:
            o[f'allowed_{lab}'] = 100 * allowed[m].sum() / nn
            o[f'nonfav_{lab}'] = 100 * nonfav[m].sum() / nn
            o[f'clash_{lab}'] = 100 * clash[m].sum() / nn
            mr = m & has_rota
            o[f'rota_{lab}'] = 100 * rota_out[mr].sum() / max(mr.sum(), 1)
    rows.append(o)
    # distance profiles (per unit) on E2-clean residues (no disagreement, no other runs) excluding the site itself
    base = inV & has_r & ~ex2 | (inV & has_r & site)
    seqd = np.array([np.min(np.abs(np.where(site)[0] - i)) for i in range(n)])
    for lab, m in [('seq_run', site & inV & has_r), ('seq_1-3', inV & has_r & ~site & (seqd <= 3)),
                   ('seq_4-10', base & ~site & (seqd > 3) & (seqd <= 10)),
                   ('seq_11-20', base & ~site & (seqd > 10) & (seqd <= 20)), ('seq_>20', base & ~site & (seqd > 20)),
                   ('3d_<8A', base & ~ex0 & (dmin < 8)), ('3d_8-12A', base & ~ex0 & (dmin >= 8) & (dmin < 12)),
                   ('3d_12-16A', base & ~ex0 & (dmin >= 12) & (dmin < 16)),
                   ('3d_16-20A', base & ~ex0 & (dmin >= 16) & (dmin < 20)), ('3d_>=20A', base & ~ex0 & (dmin >= 20))]:
        nn = int(m.sum())
        if nn:
            drows.append(dict(unit=u.unit, pdb=u.pdb, case=o['case'], cclass=u.cclass, ybin=u.ybin, res=u.res,
                              bin=lab, n=nn, nonfav=int(nonfav[m].sum()), clash=int(clash[m].sum())))
M = pd.DataFrame(rows).merge(d, on=['pdb', 'chain'], how='left')
M['author_c'] = M.last_author.fillna(M.paper).fillna(M.pdb)
M.to_csv(f'{OUT}/tables/v02_unit_strict.csv', index=False)
D = pd.DataFrame(drows).merge(M[['unit', 'pdb', 'author_c']], on=['unit', 'pdb'], how='left')

res = []


def fit(x, y, lab, extra=''):
    x = x.dropna(subset=[y])
    if x.case.sum() < 5 or (1 - x.case).sum() < 5:
        return
    m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res{extra}', x).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(x.author_c)[0]})
    lo, hi = m.conf_int().loc['case']
    res.append(dict(metric=y, subset=lab, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                    case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(),
                    case_median=x[x.case == 1][y].median(), ctrl_median=x[x.case == 0][y].median(),
                    effect=m.params['case'], ci_lo=lo, ci_hi=hi, p=m.pvalues['case']))


for lab in ['E0', 'E0r', 'E1', 'E2', 'E3', 'E4', 'E5']:
    for met in ['allowed', 'clash', 'rota']:
        fit(M, f'{met}_{lab}', 'all')
# by case origin category (each origin's cases vs all controls)
M['orig1'] = M.origin.str.split(';').str[0]
for o_, x in M[M.case == 1].groupby('orig1'):
    if len(x) >= 8:
        fit(pd.concat([x, M[M.case == 0]]), 'allowed_E3', f'cases with origin {o_}')
T = pd.DataFrame(res)
T.to_csv(f'{OUT}/tables/v02_strict_models.csv', index=False)
pd.set_option('display.width', 250)
print(T.round(4).to_string())
print('disagreeing residues outside run+-3 (and other runs), per unit:')
print(M.groupby('case').n_dis_outside.describe())
print(M.groupby('case')[['n_E0', 'n_E2', 'n_E3', 'n_E4']].mean())

# distance profile: per-unit rates, adjusted case effect in each bin
D['nonfav_pct'] = 100 * D.nonfav / D.n; D['clash_pct'] = 100 * D.clash / D.n
prof = []
for b, x in D.groupby('bin'):
    x = x[x.n >= 3]
    m = smf.wls('nonfav_pct ~ case + C(cclass) + C(ybin) + res', x, weights=x.n).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(x.author_c)[0]})
    pooled = x.groupby('case')[['n', 'nonfav']].sum()
    prof.append(dict(bin=b, n_case_units=int(x.case.sum()), n_ctrl_units=int((1 - x.case).sum()),
                     case_pooled_pct=100 * pooled.loc[1, 'nonfav'] / pooled.loc[1, 'n'],
                     ctrl_pooled_pct=100 * pooled.loc[0, 'nonfav'] / pooled.loc[0, 'n'],
                     effect_pp=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1],
                     p=m.pvalues['case']))
P = pd.DataFrame(prof)
P.to_csv(f'{OUT}/tables/v02_distance_profile.csv', index=False)
print(P.round(4).to_string())
