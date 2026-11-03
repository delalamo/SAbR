"""(1) Within-entry contrast: case chain vs every error-free V chain (err_any==0) in the same case entry, with the
same IMGT region (case run range +-3 positions) excluded from both.  (2) Controls: cached vs non-cached entries
(the control picker preferred cached entries). (3) Adjust for number of V chains in the entry (clean entries are
selected to have no run in any chain, which favours entries with few V chains)."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, os
from scipy import stats
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
R = pd.read_pickle(f'{SP}/deeper/cache/verify_beyond/res_all.pkl')
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain', 'ctype', 'vhh', 'err_first', 'err_any', 'vseq', 'unit'])
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
G = {k: g.sort_values('idx').reset_index(drop=True) for k, g in R.groupby(['pdb', 'chain'])}
dk = d.set_index(['pdb', 'chain'])


def rate(g, ranges, pad=3):
    inV = (g.apos.notna() | g.spos.notna()).values
    pos = g.apos.fillna(g.spos).values
    ex = np.zeros(len(g), bool)
    for a, b in ranges:
        ex |= (pos >= a - pad) & (pos <= b + pad)
    m = inV & ~ex
    n = m.sum()
    if n < 20:
        return np.nan, np.nan, n
    return 100 * (g.rama.values[m] == 'Allowed').sum() / n, 100 * (g.nclash.fillna(0).values[m] > 0).sum() / n, n


rows = []
for _, u in s[s.role_cc == 'case'].iterrows():
    rg = case_runs.get((u.pdb, u.chain), [])
    a0, c0, n0 = rate(G[(u.pdb, u.chain)], rg)
    for (p, ch), g in G.items():
        if p != u.pdb or ch == u.chain or (p, ch) not in dk.index:
            continue
        info = dk.loc[(p, ch)]
        same_type = (info.ctype == 'H') == (u.ctype == 'H')
        a1, c1, n1 = rate(g, rg if same_type else [])
        rows.append(dict(pdb=p, case_chain=u.chain, other=ch, cclass=u.cclass, other_ctype=info.ctype,
                         same_type=same_type, other_err_any=bool(info.err_any), other_err_first=bool(info.err_first),
                         same_seq=info.vseq == dk.loc[(p, u.chain), 'vseq'],
                         case_allowed=a0, other_allowed=a1, case_clash=c0, other_clash=c1))
W = pd.DataFrame(rows)
W.to_csv(f'{OUT}/v_within_entry_pairs.csv', index=False)
print('pairs', len(W), 'entries', W.pdb.nunique())
for lab, sub in [('error-free other V chains (any type)', W[~W.other_err_any]),
                 ('error-free, same chain type (other antibodies)', W[~W.other_err_any & W.same_type]),
                 ('error-free, opposite type (partner/other light/heavy)', W[~W.other_err_any & ~W.same_type]),
                 ('other chains WITH a run (copies etc.)', W[W.other_err_any])]:
    x = sub.dropna(subset=['case_allowed', 'other_allowed'])
    e = x.groupby(['pdb', 'case_chain']).agg(ca=('case_allowed', 'first'), oa=('other_allowed', 'mean'),
                                             cc=('case_clash', 'first'), oc=('other_clash', 'mean'))
    if len(e) < 5:
        print(lab, 'n', len(e)); continue
    dd = e.ca - e.oa; dc = e.cc - e.oc
    print(f'{lab}: n case chains={len(e)} entries={e.index.get_level_values(0).nunique()} | allowed case {e.ca.mean():.2f} other {e.oa.mean():.2f} '
          f'diff mean {dd.mean():.2f} median {dd.median():.2f} Wilcoxon p={stats.wilcoxon(dd).pvalue:.2g} | clash diff {dc.mean():.2f} p={stats.wilcoxon(dc).pvalue:.2g}')

# (2) cached vs non-cached controls; (3) n V chains in entry
cached = {x[:4] for x in os.listdir(f'{SP}/sabr/dl') if x.endswith('.cif')}
U = pd.read_csv(f'{OUT}/v_unit_metrics.csv', dtype={'chain': str})
U['cached'] = U.pdb.isin(cached).astype(int)
U = U.merge(ent[['pdb', 'n_V_chains']], on='pdb', how='left')
CC = U[U.case == 0].dropna(subset=['allowed_out'])
m = smf.ols('allowed_out ~ cached + C(cclass) + C(ybin) + res', CC).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(CC.pdb)[0]})
print('controls: cached', CC.cached.sum(), "non-cached", (1 - CC.cached).sum(), 'cached effect on allowed_out',
      round(m.params['cached'], 2), 'p', round(m.pvalues['cached'], 3))
print('n_V_chains median case/ctrl', U.groupby('case').n_V_chains.median().to_dict())
for y in ['allowed_out', 'clash_out']:
    x = U.dropna(subset=[y])
    m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res + np.log(n_V_chains)', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.last_author.fillna('NA'))[0]})
    print(y, '+ log n_V_chains: effect', round(m.params['case'], 2), 'CI', m.conf_int().loc['case'].round(2).tolist(), 'p', f"{m.pvalues['case']:.2g}")
    x2 = x[(x.case == 1) | (x.cached == 0)]
    m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res', x2).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x2.pdb)[0]})
    print(y, 'non-cached controls only: effect', round(m.params['case'], 2), 'p', f"{m.pvalues['case']:.2g}", 'n ctrl', int((x2.case == 0).sum()))
