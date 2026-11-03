"""Is the geometry excess a property of the chain that carries the error, or of the whole entry's antibody model?
Chain-level data set over every V chain in the sample entries, three groups:
  case_chain   : the representative error chain (run region excluded, IMGT range +-3)
  clean_in_case: V chains in case entries with no disagreement run of any kind (err_any==0)
  ctrl         : every V chain in the control entries (all err_any==0)
Same IMGT region excluded for same-type chains in the entry (case's runs; for controls the matched case's runs).
Adjusted for chain class, year bin, resolution; cluster by entry (and by last author)."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
R = pd.read_pickle(f'{SP}/deeper/cache/verify_beyond/res_all.pkl')
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str},
                usecols=['pdb', 'chain', 'ctype', 'vhh', 'err_first', 'err_any', 'last_author', 'year', 'res'])
d['cclass'] = np.where(d.vhh == 1, 'VHH', d.ctype)
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
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
        return dict(n=n)
    return dict(n=n, allowed=100 * (g.rama.values[m] == 'Allowed').sum() / n,
                clash=100 * (g.nclash.fillna(0).values[m] > 0).sum() / n,
                rota=100 * (g.rota.values[m] == 'OUTLIER').sum() / n)


ybin = s.groupby('pdb').ybin.first(); role = s.groupby('pdb').role_cc.first()
rows = []
for p in sorted(set(s.pdb)):
    units = s[s.pdb == p]
    for (pp, ch), g in G.items():
        if pp != p or (p, ch) not in dk.index:
            continue
        info = dk.loc[(p, ch)]
        # region to exclude: runs of the case unit(s) in this entry (or matched case) of the same chain type
        rg = []
        for _, u in units.iterrows():
            if (u.ctype == 'H') == (info.ctype == 'H'):
                rg += set_runs.get(u.set_id, [])
        is_case_chain = role[p] == 'case' and ((units.chain == ch) & (units.role_cc == 'case')).any()
        if role[p] == 'case':
            grp = 'case_chain' if is_case_chain else ('clean_in_case' if not info.err_any else 'other_err_in_case')
        else:
            grp = 'ctrl_unit' if (units.chain == ch).any() else 'ctrl_other'
        o = dict(pdb=p, chain=ch, grp=grp, cclass=info.cclass, ybin=ybin[p], res=info.res, lab=info.last_author,
                 err_first=bool(info.err_first))
        o.update(rate(g, rg))
        rows.append(o)
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/v_entry_level_chains.csv', index=False)
print(D.groupby('grp').agg(n=('pdb', 'size'), entries=('pdb', 'nunique'), allowed=('allowed', 'mean'),
                           allowed_med=('allowed', 'median'), clash=('clash', 'mean'), rota=('rota', 'mean')).round(2))
D['ctrl'] = D.grp.str.startswith('ctrl')
out = []
for g1 in ['case_chain', 'clean_in_case', 'other_err_in_case']:
    for y in ['allowed', 'clash', 'rota']:
        for cl in ['pdb', 'lab']:
            x = D[(D.grp == g1) | D.ctrl].dropna(subset=[y]).copy()
            x['expo'] = (x.grp == g1).astype(int)
            m = smf.ols(f'{y} ~ expo + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[cl].fillna('NA'))[0]})
            out.append(dict(exposed=g1, metric=y, cluster=cl, n_exp=int(x.expo.sum()), n_ctrl=int((1 - x.expo).sum()),
                            exp_mean=x[x.expo == 1][y].mean(), ctrl_mean=x[x.expo == 0][y].mean(),
                            effect=m.params['expo'], ci_lo=m.conf_int().loc['expo', 0], ci_hi=m.conf_int().loc['expo', 1], p=m.pvalues['expo']))
# direct: case chain vs clean chain within case entries, entry fixed effects
x = D[D.grp.isin(['case_chain', 'clean_in_case'])].dropna(subset=['allowed']).copy()
x['expo'] = (x.grp == 'case_chain').astype(int)
multi = x.groupby('pdb').expo.agg(['min', 'max']); multi = multi[(multi['min'] == 0) & (multi['max'] == 1)].index
x = x[x.pdb.isin(multi)]
for y in ['allowed', 'clash', 'rota']:
    m = smf.ols(f'{y} ~ expo + C(cclass) + C(pdb)', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
    out.append(dict(exposed='case_chain vs clean chains, entry fixed effects', metric=y, cluster='pdb', n_exp=int(x.expo.sum()),
                    n_ctrl=int((1 - x.expo).sum()), exp_mean=x[x.expo == 1][y].mean(), ctrl_mean=x[x.expo == 0][y].mean(),
                    effect=m.params['expo'], ci_lo=m.conf_int().loc['expo', 0], ci_hi=m.conf_int().loc['expo', 1], p=m.pvalues['expo']))
O = pd.DataFrame(out); O.to_csv(f'{OUT}/v_entry_level_tests.csv', index=False)
pd.set_option('display.width', 250); print(O.round(5).to_string())
print('entries with both case and clean chains:', len(multi))
