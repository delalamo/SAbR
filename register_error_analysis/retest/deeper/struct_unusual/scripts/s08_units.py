"""Chain-level metrics for every sample unit (and for the VH/VL partner of paired units).
Regions: whole V domain; the error site (case runs, or the matched case's runs for controls; +-2 residues);
'outside' = V domain minus the site +-3 residues.  Output tables/unit_metrics.csv"""
import numpy as np, pandas as pd
from common import SP, C, TAB

R = pd.read_pickle(f'{C}/residues.pkl')
R['chain'] = R.chain.astype(str)
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
runs_all = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
pairs = pd.read_csv(f'{TAB}/pairs.csv', dtype={'H': str, 'L': str, 'unit_chain': str})
h3 = pd.read_csv(f'{TAB}/h3base.csv', dtype={'chain': str}).set_index(['pdb', 'chain'])
entq = pd.read_csv(f'{TAB}/entry_q.csv').set_index('pdb').entry_q
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str},
                usecols=['pdb', 'chain', 'ctype', 'vhh', 'L1', 'L2', 'L3', 'err_first', 'err_any'])
dk = d.set_index(['pdb', 'chain'])

# CDR-length typicality: frequency of the length among all scanned chains of the same class (VH, VHH, K, L)
d['cclass'] = np.where(d.vhh == 1, 'VHH', d.ctype)
freq = {(cc, c): d[d.cclass == cc][c].value_counts(normalize=True).to_dict() for cc in d.cclass.unique()
        for c in ['L1', 'L2', 'L3']}

case_runs = {k: g[['imgt0', 'imgt1', 'mode']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_site = {}
for _, u in s[s.role_cc == 'case'].iterrows():
    set_site[u.set_id] = case_runs.get((u.pdb, u.chain), [])

G = {k: g.sort_values('idx').reset_index(drop=True) for k, g in R.groupby(['pdb', 'chain'])}


def site_idx(g, ranges, pad=2):
    inV = g.apos.notna() | g.spos.notna()
    hit = np.zeros(len(g), bool)
    for a, b, *_ in ranges:
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    idx = np.where(hit)[0]
    out = np.zeros(len(g), bool)
    for i in idx:
        out[max(0, i - pad):i + pad + 1] = True
    return out & inV.values


def metrics(g, site, prefix=''):
    inV = (g.apos.notna() | g.spos.notna()).values
    nxt_rn = g.rn.shift(-1).values
    cis = (np.abs(g.omega.values) < 30)
    cis_nonpro = cis & (nxt_rn != 'PRO')
    caca_out = (np.abs(g.caca_next.values - 3.80) > 0.15) & ~cis & ~np.isnan(g.caca_next.values)
    trunc = (g.natoms.values < g.nexp.values)
    rama_out = (g.rama.values == 'OUTLIER'); rama_allw = (g.rama.values == 'Allowed')
    rota_out = (g.rota.values == 'OUTLIER')
    clash = (g.nclash.fillna(0).values > 0)
    geo = (g.nbond.fillna(0).values + g.nangle.fillna(0).values) > 0
    q = g.q.values
    o = {}
    near = np.zeros(len(g), bool)
    for i in np.where(site)[0]:
        near[max(0, i - 1):i + 2] = True
    regions = {'V': inV, 'site': site, 'out': inV & ~near}
    for rname, m in regions.items():
        n = int(m.sum())
        o[f'{prefix}n_{rname}'] = n
        if n == 0:
            continue
        for k, arr in [('rama_out', rama_out), ('rama_allowed', rama_allw), ('rota_out', rota_out), ('clash', clash),
                       ('geo_out', geo), ('cis_nonpro', cis_nonpro), ('caca_out', caca_out), ('sc_trunc', trunc)]:
            if rname == 'site':
                o[f'{prefix}{k}_{rname}'] = int(arr[m].sum() > 0)
            else:
                o[f'{prefix}{k}_{rname}'] = 100 * arr[m].sum() / n
        o[f'{prefix}q_{rname}'] = np.nanmean(q[m]) if np.isfinite(q[m]).any() else np.nan
    o[f'{prefix}cis_pro_V'] = int((cis & (nxt_rn == 'PRO') & inV).sum())
    o[f'{prefix}bb_missing_V'] = int((~g.bb_ok.values & inV).sum())
    gi = g[inV]
    o[f'{prefix}first_apos'] = gi.apos.min()
    o[f'{prefix}n_gaps_int'] = int((gi.prev_si_gap.iloc[1:] > 0).sum())
    o[f'{prefix}unbuilt_int'] = float(gi.prev_si_gap.iloc[1:].clip(lower=0).sum())
    return o


rows = []
for _, u in s.iterrows():
    k = (u.pdb, u.chain)
    if k not in G:
        continue
    g = G[k]
    site = site_idx(g, set_site.get(u.set_id, []))
    o = dict(unit=u.unit, pdb=u.pdb, chain=u.chain, role_cc=u.role_cc, set_id=u.set_id, cclass=u.cclass,
             ybin=u.ybin, res=u.res, year=u.year, paper=u.paper,
             modes=';'.join(sorted({m for *_, m in set_site.get(u.set_id, [])})))
    o.update(metrics(g, site))
    o['q_entry'] = entq.get(u.pdb, np.nan)
    o['q_V_rel'] = o.get('q_V', np.nan) - o['q_entry']
    o['q_site_minus_out'] = o.get('q_site', np.nan) - o.get('q_out', np.nan)
    if k in h3.index:
        t = h3.loc[k, 'tau_sabr']
        o['h3_tau'] = t; o['h3_alpha'] = h3.loc[k, 'alpha_sabr']
        o['h3_kinked'] = float(-30 <= t <= 100) if np.isfinite(t) else np.nan
        ta = h3.loc[k, 'tau_anarci']
        o['h3_kinked_seqnum'] = float(-30 <= ta <= 100) if np.isfinite(ta) else np.nan
    # CDR-length typicality
    fr = []
    for c in (['L1', 'L2'] if u.ctype == 'H' else ['L1', 'L2', 'L3']):
        L = dk.loc[k, c] if k in dk.index else np.nan
        f = freq[(u.cclass, c)].get(L, 0.0) if np.isfinite(L) else np.nan
        o[f'freq_{c}'] = f; fr.append(f)
    o['min_cdr_len_freq'] = np.nanmin(fr) if fr else np.nan
    o['rare_cdr_len'] = float(o['min_cdr_len_freq'] < 0.02)
    o['cdr3_len'] = dk.loc[k, 'L3'] if k in dk.index else np.nan
    # partner chain (paired units): whole-V metrics, only if the partner has no disagreement run of any kind
    pr = pairs[(pairs.pdb == u.pdb) & (pairs.unit_chain == u.chain)]
    if len(pr) and pr.cdist.iloc[0] <= 28:
        pc = pr.L.iloc[0] if u.ctype == 'H' else pr.H.iloc[0]
        o['partner'] = pc
        o['partner_err_any'] = bool(dk.loc[(u.pdb, pc), 'err_any']) if (u.pdb, pc) in dk.index else np.nan
        if (u.pdb, pc) in G:
            gp = G[(u.pdb, pc)]
            pm = metrics(gp, np.zeros(len(gp), bool), prefix='p_')
            o.update({kk: v for kk, v in pm.items() if kk.endswith('_V') or kk in ('p_first_apos', 'p_n_gaps_int', 'p_unbuilt_int')})
    rows.append(o)
U = pd.DataFrame(rows)
U.to_csv(f'{TAB}/unit_metrics.csv', index=False)
print(U.shape, U.role_cc.value_counts().to_dict())
cols = ['rama_out_V', 'rama_out_out', 'rama_out_site', 'rota_out_V', 'clash_V', 'geo_out_V', 'cis_nonpro_V', 'caca_out_V',
        'sc_trunc_V', 'q_V', 'q_V_rel', 'q_out', 'q_site', 'q_site_minus_out', 'h3_kinked', 'rare_cdr_len',
        'p_rama_out_V', 'p_q_V', 'p_sc_trunc_V']
print(U.groupby('role_cc')[cols].mean().T.round(3))
