"""Per-unit beyond-site metrics from the independent residue table; several site definitions."""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OC = f'{SP}/deeper/cache/verify_beyond'
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
R = pd.read_pickle(f'{OC}/res_all.pkl')
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
runs_all = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str},
                usecols=['pdb', 'chain', 'ctype', 'vhh', 'err_first', 'err_any', 'paper', 'last_author', 'sars2',
                         'irm', 'fit', 'deposit'])
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
G = {k: g.sort_values('idx').reset_index(drop=True) for k, g in R.groupby(['pdb', 'chain'])}


def dilate(mask, pad):
    out = np.zeros_like(mask)
    for i in np.where(mask)[0]:
        out[max(0, i - pad):i + pad + 1] = True
    return out


def rates(g, excl, tag):
    inV = (g.apos.notna() | g.spos.notna()).values
    m = inV & ~excl
    n = int(m.sum()); ne = int((m & g.rama.notna().values).sum())
    o = {f'n_{tag}': n, f'neval_{tag}': ne}
    if n == 0:
        return o
    allw = (g.rama.values == 'Allowed'); out = (g.rama.values == 'OUTLIER')
    rot = (g.rota.values == 'OUTLIER'); cl = g.nclash.fillna(0).values > 0
    o[f'allowed_{tag}'] = 100 * allw[m].sum() / n           # investigator's denominator (all V residues)
    o[f'allowed_eval_{tag}'] = 100 * allw[m].sum() / max(ne, 1)  # per residue with a Ramachandran class
    o[f'nonfav_eval_{tag}'] = 100 * (allw | out)[m].sum() / max(ne, 1)
    o[f'ramaout_{tag}'] = 100 * out[m].sum() / n
    o[f'rota_{tag}'] = 100 * rot[m].sum() / n
    o[f'clash_{tag}'] = 100 * cl[m].sum() / n
    o[f'k_allowed_{tag}'] = int(allw[m].sum()); o[f'k_clash_{tag}'] = int(cl[m].sum())
    return o


rows = []
for _, u in s.iterrows():
    k = (u.pdb, u.chain)
    if k not in G:
        continue
    g = G[k]
    inV = (g.apos.notna() | g.spos.notna()).values
    # (1) investigator's definition: first-occurrence run IMGT ranges on either numbering, +-3
    hit = np.zeros(len(g), bool)
    for a, b in set_runs.get(u.set_id, []):
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    hit &= inV
    ex_runs = dilate(hit, 3)
    # (2) every residue with any ANARCI/SAbR disagreement (incl. single-residue and copied/artifact runs), +-3
    dis = inV & ((g.apos.fillna(-1).values != g.spos.fillna(-1).values) | (g.ains.values != g.sins.values))
    ex_dis = dilate(dis | hit, 3)
    # (3) stricter: +-10 around (1)|(2)
    ex_10 = dilate(dis | hit, 10)
    o = dict(unit=u.unit, pdb=u.pdb, chain=u.chain, role_cc=u.role_cc, case=int(u.role_cc == 'case'),
             set_id=u.set_id, cclass=u.cclass, ybin=u.ybin, res=u.res, year=u.year, paper=u.paper,
             last_author=u.last_author, sars2=float(u.sars2), n_dis=int(dis.sum()), n_hit=int(hit.sum()))
    o.update(rates(g, ex_runs, 'out'))
    o.update(rates(g, ex_dis, 'outdis'))
    o.update(rates(g, ex_10, 'out10'))
    o.update(rates(g, np.zeros(len(g), bool), 'V'))
    rows.append(o)
U = pd.DataFrame(rows)
U.to_csv(f'{OUT}/v_unit_metrics.csv', index=False)
# compare with the investigator's table
I = pd.read_csv(f'{TAB}/unit_metrics.csv', dtype={'chain': str})
M = U.merge(I[['unit', 'pdb', 'chain', 'rama_allowed_out', 'clash_out', 'rota_out_out', 'n_out']], on=['unit', 'pdb', 'chain'])
for a, b in [('allowed_out', 'rama_allowed_out'), ('clash_out', 'clash_out_y' if 'clash_out_y' in M else 'clash_out'),
             ('rota_out', 'rota_out_out'), ('n_out', 'n_out_y' if 'n_out_y' in M else 'n_out')]:
    pass
M = U.merge(I[['unit', 'pdb', 'chain', 'rama_allowed_out', 'clash_out', 'rota_out_out', 'n_out']].rename(
    columns={'clash_out': 'I_clash_out', 'n_out': 'I_n_out'}), on=['unit', 'pdb', 'chain'])
print('rows', len(U), 'matched to investigator', len(M))
for a, b in [('allowed_out', 'rama_allowed_out'), ('clash_out', 'I_clash_out'), ('rota_out', 'rota_out_out'), ('n_out', 'I_n_out')]:
    x = M[[a, b]].dropna()
    print(f'{a:12s} vs {b:16s}: n={len(x)} max|diff|={np.abs(x[a]-x[b]).max():.4f} corr={x[a].corr(x[b]):.4f}')
print('case chains with extra disagreement residues beyond first-occurrence runs:',
      int(((U.case == 1) & (U.n_dis > 0)).sum()), 'controls with any disagreement residue', int(((U.case == 0) & (U.n_dis > 0)).sum()))
print(U.groupby('role_cc')[['allowed_out', 'allowed_eval_out', 'nonfav_eval_out', 'clash_out', 'rota_out', 'ramaout_out',
                            'allowed_outdis', 'allowed_out10', 'clash_out10']].agg(['mean', 'median', 'count']).T.round(2))
