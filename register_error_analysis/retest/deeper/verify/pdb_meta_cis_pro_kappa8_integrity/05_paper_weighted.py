"""Paper-weighted and cluster-bootstrap versions of the kappa cis-Pro8 association (EM, non-amyloid, Pro8 modelled).
- weights: each paper's error chains sum to 1 and its clean chains sum to 1 (one vote per paper per class)
- cluster bootstrap over papers (1000 reps) of the year+resolution-adjusted log OR (non-FR1 cases)
- paper-level sign test inside error papers that also contain clean kappa chains
- leave-two-largest-error-papers-out"""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_integrity'
D = pd.read_csv(f'{OUT}/frame.csv.gz')
D = D[(D.method == 'EM') & (~D.amyloid) & (D.err_first | ~D.err_any) & (D.pro8 == True)].copy()
D['y'] = D.err_first.astype(int); D['x'] = D.cis8.astype(int); D['cluster'] = D.paper.fillna(D.pdb)
D['yr'] = pd.to_datetime(D.deposit).dt.year
D['ybin'] = np.where(D.yr <= 2018, '<=2018', np.where(D.yr >= 2024, '>=2024', D.yr.astype(str)))
D['modeA'] = D.err_modes.fillna('').str.contains('A_FR1') & D.err_first
res = []
for lab, d in [('all', D), ('noFR1', D[~D.modeA])]:
    d = d.copy()
    d['w'] = 1 / d.groupby(['cluster', 'y']).pdb.transform('size')
    # rescale weights so that total weight equals n of papers per class (keeps SEs on paper scale)
    m = smf.glm('y ~ x + C(ybin) + res', data=d, family=sm.families.Binomial(), var_weights=d.w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.cluster)[0]})
    res.append(dict(test=f'paper-weighted GLM ({lab})', OR=np.exp(m.params['x']), lo=np.exp(m.conf_int().loc['x', 0]), hi=np.exp(m.conf_int().loc['x', 1]), p=m.pvalues['x'],
                    err_cis_paperw=np.average(d[d.y == 1].x, weights=d[d.y == 1].w), ctrl_cis_paperw=np.average(d[d.y == 0].x, weights=d[d.y == 0].w)))
    # cluster bootstrap
    rng = np.random.default_rng(7)
    cl = d.cluster.unique(); g = {c: d[d.cluster == c] for c in cl}
    errcl = d[d.y == 1].cluster.unique(); ctlcl = [c for c in cl if c not in set(errcl)]
    bs = []
    for b in range(1000):
        s = pd.concat([g[c] for c in rng.choice(errcl, len(errcl))] + [g[c] for c in rng.choice(ctlcl, len(ctlcl))])
        try:
            mm = smf.glm('y ~ x + C(ybin) + res', data=s, family=sm.families.Binomial()).fit()
            bs.append(mm.params['x'])
        except Exception:
            pass
    bs = np.array(bs); bs = bs[np.isfinite(bs) & (np.abs(bs) < 20)]
    res.append(dict(test=f'paper cluster bootstrap, stratified by error/clean papers ({lab}), n={len(bs)}', OR=np.exp(np.median(bs)), lo=np.exp(np.percentile(bs, 2.5)), hi=np.exp(np.percentile(bs, 97.5)),
                    p=2 * min((bs >= 0).mean(), (bs < 0).mean()) if len(bs) else np.nan))
    big = d[d.y == 1].cluster.value_counts().index[:2]
    dd = d[~d.cluster.isin(big)]
    m = smf.glm('y ~ x + C(ybin) + res', data=dd, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.cluster)[0]})
    res.append(dict(test=f'drop two largest error papers ({", ".join(big)}) ({lab})', OR=np.exp(m.params['x']), p=m.pvalues['x'], n_err=int(dd.y.sum())))
# within-paper sign test (paper = unit): error papers with clean kappa chains; compare error-chain cis fraction to clean-chain cis fraction
W = D[D.cluster.isin(D[D.y == 1].cluster)]
rows = []
for c, g in W.groupby('cluster'):
    if g.y.nunique() < 2:
        continue
    rows.append(dict(cluster=c, cis_err=g[g.y == 1].x.mean(), cis_clean=g[g.y == 0].x.mean()))
P = pd.DataFrame(rows)
lower = (P.cis_err < P.cis_clean).sum(); higher = (P.cis_err > P.cis_clean).sum(); tie = (P.cis_err == P.cis_clean).sum()
res.append(dict(test=f'within-paper sign test: error chains less cis than same-paper clean chains (papers lower/higher/tie = {lower}/{higher}/{tie})',
                p=stats.binomtest(lower, lower + higher).pvalue))
tie_trans = ((P.cis_err == 0) & (P.cis_clean == 0)).sum()
res.append(dict(test=f'error papers with clean kappa chains where ALL kappa chains are trans: {tie_trans}/{len(P)}'))
R = pd.DataFrame(res)
R.to_csv(f'{OUT}/paper_weighted.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 140)
print(R.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
