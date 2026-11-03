"""Within-entry comparison: in cryo-EM entries that contain a first-occurrence error, is the error chain's V domain
(outside the site) worse than the error-free V chains of the SAME entry (same map, lab, software, refinement run,
year, resolution)?  Entry fixed effects remove every entry-level confounder.
Comparators: (a) the VH/VL partner of the same Fab, (b) V chains of a different antibody in the same entry,
(c) all error-free V chains.  Comparator chains have the case's run IMGT ranges (+-3) excluded too.
Also repeats the case-control contrast for the error-free partner (whole V) with author-clustered SEs.
Output tables/v04_*.csv
"""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SU = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
CU = f'{SP}/deeper/cache/struct_unusual'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_confounding'

R = pd.read_pickle(f'{CU}/residues.pkl'); R['chain'] = R.chain.astype(str)
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
d['cclass'] = np.where(d.vhh == 1, 'VHH', d.ctype)
s = pd.read_csv(f'{SU}/sample.csv', dtype={'chain': str})
pairs = pd.read_csv(f'{SU}/pairs.csv', dtype={'H': str, 'L': str, 'unit_chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
runs_all = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
first_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
any_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs_all.groupby(['pdb', 'chain'])}
case_entries = sorted(set(s[s.role_cc == 'case'].pdb))
G = {k: g.sort_values('idx').reset_index(drop=True) for k, g in R[R.pdb.isin(case_entries)].groupby(['pdb', 'chain'])}
dk = d.set_index(['pdb', 'chain'])


def dilate(m, k):
    o = m.copy()
    for i in np.where(m)[0]:
        o[max(0, i - k):i + k + 1] = True
    return o


def chain_metric(g, ranges):
    inV = (g.apos.notna() | g.spos.notna()).values
    hit = np.zeros(len(g), bool)
    for a, b in ranges:
        hit |= ((g.apos >= a) & (g.apos <= b)).values | ((g.spos >= a) & (g.spos <= b)).values
    dis = inV & ((g.apos.values != g.spos.values) | g.apos.isna().values | g.spos.isna().values)
    gap = g.prev_si_gap.fillna(0).values > 0
    brk = gap | np.roll(gap, -1)
    vi = np.where(inV)[0]
    brk[vi.min()] = brk[vi.max()] = True
    ex = dilate(hit & inV, 3) | dilate(dis, 3) | dilate(brk, 3)
    hasr = g.rama.notna().values
    m = inV & ~ex & hasr
    n = int(m.sum())
    if n < 20:
        return None
    return dict(n=n, allowed=100 * (g.rama.values[m] == 'Allowed').mean(),
                nonfav=100 * np.isin(g.rama.values[m], ['Allowed', 'OUTLIER']).mean(),
                clash=100 * (g.nclash.fillna(0).values[m] > 0).mean(),
                rota=100 * (g.rota.values[m] == 'OUTLIER').sum() / max(1, (m & g.rota.notna().values).sum()),
                q=np.nanmean(g.q.values[m]) if np.isfinite(g.q.values[m].astype(float)).any() else np.nan)


rows = []
for p in case_entries:
    ch_err = [c for (pp, c) in first_runs if pp == p]
    site_ranges = sorted({tuple(r) for c in ch_err for r in first_runs[(p, c)]})
    for (pp, c), g in G.items():
        if pp != p or (p, c) not in dk.index:
            continue
        info = dk.loc[(p, c)]
        is_err = (p, c) in first_runs
        has_any = (p, c) in any_runs
        ranges = first_runs[(p, c)] if is_err else site_ranges
        mm = chain_metric(g, ranges + (any_runs.get((p, c), []) if not is_err else []))
        if mm is None:
            continue
        # partner of an error chain?
        pr = pairs[(pairs.pdb == p) & ((pairs.H == c) | (pairs.L == c))]
        partner = None
        if len(pr):
            partner = pr.L.iloc[0] if pr.H.iloc[0] == c else pr.H.iloc[0]
        rows.append(dict(pdb=p, chain=c, err=int(is_err), any_run=int(has_any), cclass=info.cclass, vseq=info.vseq,
                         unit=info.unit, partner=partner, partner_is_err=int((p, partner) in first_runs) if partner else 0,
                         **mm))
W = pd.DataFrame(rows)
# label comparators
err_units = set(W[W.err == 1].unit)
W['comp'] = np.where(W.err == 1, 'error chain',
                     np.where(W.any_run == 1, 'other run (copied/artifact)',
                              np.where(W.partner_is_err == 1, 'partner of error chain',
                                       np.where(W.unit.isin(err_units), 'same antibody, error-free copy', 'different antibody'))))
W.to_csv(f'{OUT}/tables/v04_within_entry_chains.csv', index=False)
print(W.comp.value_counts())
res = []
for lab, keep in [('all error-free chains (no run)', ['partner of error chain', 'different antibody', 'same antibody, error-free copy']),
                  ('partner of error chain', ['partner of error chain']),
                  ('different antibody', ['different antibody']),
                  ('same antibody, error-free copy', ['same antibody, error-free copy'])]:
    x = W[(W.comp == 'error chain') | W.comp.isin(keep)]
    # keep entries that have both
    ok = x.groupby('pdb').err.agg(lambda e: e.min() == 0 and e.max() == 1)
    x = x[x.pdb.isin(ok[ok].index)]
    if x.err.sum() < 5:
        continue
    for y in ['allowed', 'nonfav', 'clash', 'rota', 'q']:
        xx = x.dropna(subset=[y])
        f = f'{y} ~ err + C(pdb)' + (' + C(cclass)' if xx.cclass.nunique() > 1 else '')
        m = smf.ols(f, xx).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(xx.pdb)[0]})
        lo, hi = m.conf_int().loc['err']
        # paired per-entry difference (entry mean err minus entry mean comparator) for a rank test
        pe = xx.groupby(['pdb', 'err'])[y].mean().unstack().dropna()
        from scipy.stats import wilcoxon
        res.append(dict(comparator=lab, metric=y, n_entries=xx.pdb.nunique(), n_err_chains=int(xx.err.sum()),
                        n_comp_chains=int((1 - xx.err).sum()), err_mean=xx[xx.err == 1][y].mean(),
                        comp_mean=xx[xx.err == 0][y].mean(), effect_entryFE=m.params['err'], ci_lo=lo, ci_hi=hi,
                        p=m.pvalues['err'], median_paired_diff=(pe[1] - pe[0]).median(),
                        p_wilcoxon=wilcoxon(pe[1] - pe[0]).pvalue if len(pe) > 5 else np.nan))
T = pd.DataFrame(res)
T.to_csv(f'{OUT}/tables/v04_within_entry_models.csv', index=False)
pd.set_option('display.width', 250)
print(T.round(4).to_string())

# ---- partner (error-free) vs control partner, whole V, author-clustered
U = pd.read_csv(f'{SU}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
U = U.merge(d[['pdb', 'chain', 'last_author', 'sars2']], on=['pdb', 'chain'], how='left')
U['author_c'] = U.last_author.fillna(U.paper).fillna(U.pdb); U['case'] = (U.role_cc == 'case').astype(int)
P = U[U.partner_err_any == False].dropna(subset=['p_rama_allowed_V'])
out = []
for y in ['p_rama_allowed_V', 'p_clash_V', 'p_rota_out_V']:
    for lab, f in [('unadjusted', f'{y} ~ case'), ('adjusted (class, year bin, res)', f'{y} ~ case + C(cclass) + C(ybin) + res')]:
        m = smf.ols(f, P).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(P.author_c)[0]})
        out.append(dict(metric=y, model=lab, n_case=int(P.case.sum()), n_ctrl=int((1 - P.case).sum()),
                        case_mean=P[P.case == 1][y].mean(), ctrl_mean=P[P.case == 0][y].mean(),
                        effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1],
                        p=m.pvalues['case']))
O = pd.DataFrame(out); O.to_csv(f'{OUT}/tables/v04_partner_case_control.csv', index=False)
print(O.round(4).to_string())
print(P.groupby('case').cclass.value_counts().unstack())
