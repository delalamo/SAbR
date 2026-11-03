"""Is the 'beyond the site' excess really away from the slip? Sequence exclusion (+-3) leaves spatial neighbours of
a misthreaded strand (paired beta strands, packed side chains) in the 'outside' set. Here residues are additionally
excluded by spatial proximity (min CA-CA or any-atom distance to a site residue), and residues next to unbuilt
gaps/termini are excluded. Uses CA/atom coordinates from the investigator's vchains.pkl (order verified identical to
the SAbR records)."""
import pickle, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
V = pickle.load(open(f'{SP}/deeper/cache/struct_unusual/vchains.pkl', 'rb'))
R = pd.read_pickle(f'{SP}/deeper/cache/verify_beyond/res_all.pkl')
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
G = {k: g.sort_values('idx').reset_index(drop=True) for k, g in R.groupby(['pdb', 'chain'])}


def dilate(mask, pad):
    out = np.zeros_like(mask)
    for i in np.where(mask)[0]:
        out[max(0, i - pad):i + pad + 1] = True
    return out


rows, resrows = [], []
for _, u in s.iterrows():
    k = (u.pdb, u.chain)
    if k not in G or k not in V:
        continue
    g = G[k]; rec = V[k]['res']
    inV = (g.apos.notna() | g.spos.notna()).values
    hit = np.zeros(len(g), bool)
    for a, b in set_runs.get(u.set_id, []):
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    hit &= inV
    if not hit.any():
        continue
    ca = np.array([r['atoms'].get('CA', [np.nan] * 3) for r in rec], float)
    allat = [np.array(list(r['atoms'].values())) if r['atoms'] else np.full((1, 3), np.nan) for r in rec]
    H = np.where(hit)[0]
    hca = ca[H]
    hat = np.vstack([allat[i] for i in H])
    dca = np.array([np.nanmin(np.linalg.norm(hca - ca[i], axis=1)) if np.isfinite(ca[i]).all() else np.nan
                    for i in range(len(g))])
    dat = np.array([np.nanmin(np.linalg.norm(hat[:, None, :] - allat[i][None, :, :], axis=2)) for i in range(len(g))])
    seqd = np.array([np.min(np.abs(H - i)) for i in range(len(g))])
    # residues next to unbuilt residues (si gap) or the first/last 3 built V residues
    si = g.si.values
    gap = np.zeros(len(g), bool)
    gap[1:] |= (np.diff(si) > 1); gap[:-1] |= (np.diff(si) > 1)
    vi = np.where(inV)[0]
    gapnear = dilate(gap, 3)
    term = np.zeros(len(g), bool)
    if len(vi):
        term[vi[:3]] = True; term[vi[-3:]] = True
    allw = g.rama.values == 'Allowed'; cl = g.nclash.fillna(0).values > 0; rot = g.rota.values == 'OUTLIER'
    for i in np.where(inV)[0]:
        resrows.append((u.unit, u.role_cc, seqd[i], dca[i], dat[i], gapnear[i], term[i], allw[i], cl[i], rot[i]))
    o = dict(unit=u.unit, pdb=u.pdb, case=int(u.role_cc == 'case'), cclass=u.cclass, ybin=u.ybin, res=u.res,
             lab=u.last_author, paper=u.paper, n_gapres=int((gapnear & inV).sum()))
    for tag, m in [('seq3', seqd > 3),
                   ('seq3_ca10', (seqd > 3) & (dca > 10)),
                   ('seq3_ca14', (seqd > 3) & (dca > 14)),
                   ('seq3_at6', (seqd > 3) & (dat > 6)),
                   ('seq3_ca10_nogap', (seqd > 3) & (dca > 10) & ~gapnear & ~term),
                   ('seq3_nogap', (seqd > 3) & ~gapnear & ~term)]:
        mm = m & inV
        n = int(mm.sum()); o[f'n_{tag}'] = n
        if n >= 10:
            o[f'allowed_{tag}'] = 100 * allw[mm].sum() / n
            o[f'clash_{tag}'] = 100 * cl[mm].sum() / n
            o[f'rota_{tag}'] = 100 * rot[mm].sum() / n
    rows.append(o)
U = pd.DataFrame(rows); U.to_csv(f'{OUT}/v_spatial_units.csv', index=False)
RR = pd.DataFrame(resrows, columns=['unit', 'role_cc', 'seqd', 'dca', 'dat', 'gapnear', 'term', 'allowed', 'clash', 'rota'])
RR.to_pickle(f'{SP}/deeper/cache/verify_beyond/spatial_res.pkl')
out = []
for tag in ['seq3', 'seq3_nogap', 'seq3_at6', 'seq3_ca10', 'seq3_ca14', 'seq3_ca10_nogap']:
    for met in ['allowed', 'clash', 'rota']:
        y = f'{met}_{tag}'
        x = U.dropna(subset=[y])
        m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.lab.fillna('NA'))[0]})
        out.append(dict(region=tag, metric=met, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                        case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(),
                        resid_case=x[x.case == 1][f'n_{tag}'].median(), resid_ctrl=x[x.case == 0][f'n_{tag}'].median(),
                        effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1],
                        p=m.pvalues['case']))
O = pd.DataFrame(out); O.to_csv(f'{OUT}/v_spatial_tests.csv', index=False)
pd.set_option('display.width', 250); print(O.round(5).to_string())
# pooled residue-level rates by spatial distance bin (seq distance > 3 only)
RR = RR[RR.seqd > 3].copy()
RR['dbin'] = pd.cut(RR.dca, [0, 6, 8, 10, 12, 16, 20, 100])
T = RR.groupby(['dbin', 'role_cc'])[['allowed', 'clash', 'rota']].agg(['mean', 'size'])
print((T * [100, 1, 100, 1, 100, 1]).round(2).to_string())
