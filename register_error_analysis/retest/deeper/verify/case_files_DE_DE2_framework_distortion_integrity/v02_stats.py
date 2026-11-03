"""Re-test DE2_framework_distortion with the recomputed framework RMSDs (v01) and robustness checks.

Unit: one chain per (group, unit), EM only, chosen as the investigator did (sort by pdb, chain; first per unit).
Error set 'confirmed' = D/E chains whose independent superposition check is not 'not confirmed' (investigator's err_struct.pkl).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_integrity'
MODES = ['Dplus_CDR3short', 'Dminus_CDR3long', 'E_FR4bulge']
F = pd.read_csv(f'{OUT}/fw_recomputed.csv')
G = pd.read_csv(f'{OUT}/sample_groups.csv')
R = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
ER = pd.read_pickle(f'{SP}/deeper/cache/case_files_DE/err_struct.pkl')[['pdb', 'chain', 'confirm']]
X = G.merge(F, on=['pdb', 'chain'], how='left').merge(ER, on=['pdb', 'chain'], how='left')
# D/E mode per chain (first mode in the mode order used by investigator: their X_struct had one mode per chain)
dm = R[R['mode'].isin(MODES)].sort_values('imgt0').groupby(['pdb', 'chain'])['mode'].first()
X['mode'] = [dm.get((p, c)) if g == 'error' else None for p, c, g in zip(X.pdb, X.chain, X.group)]
X['yb'] = pd.cut(X.year, [0, 2019, 2020, 2021, 2023, 2030], labels=['<=2019', '2020', '2021', '2022-23', '>=2024']).astype(str)
X['ct'] = np.where(X.vhh == 1, 'VHH', X.ctype)
X['sp'] = np.where(X.species.fillna('').str.contains('sapiens|human', case=False), 'human',
                   np.where(X.species.fillna('').str.contains('musculus|mouse', case=False), 'mouse', 'other'))
X = X.sort_values(['group', 'pdb', 'chain']).drop_duplicates(['group', 'unit'])
X.to_csv(f'{OUT}/analysis_table.csv', index=False)

def err_set(which='confirmed', modes=MODES):
    e = X[(X.group == 'error') & X['mode'].isin(modes)]
    if which == 'confirmed': e = e[e.confirm != 'not confirmed']
    return e

def logit_matched(e, c, f, extra='', drop_papers=()):
    d = pd.concat([e.assign(err=1), c.assign(err=0)]).dropna(subset=[f])
    d = d[~d.paper.isin(drop_papers)]
    d['x'] = (d[f] - d[f].mean()) / d[f].std()
    m = smf.logit(f'err ~ x + C(yb) + res + C(ct){extra}', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
    return float(np.exp(m.params['x'])), float(m.pvalues['x']), int(d.err.sum()), int((d.err == 0).sum()), float(d[f].std())

def clogit(e, c, f, strata='paper', drop_papers=()):
    d = pd.concat([e.assign(err=1), c.assign(err=0)]).dropna(subset=[f])
    d = d[~d.paper.isin(drop_papers)]
    d['s'] = d[strata] if strata == 'paper' else d.paper + '|' + d.ct
    ok = d.groupby('s').err.agg(['min', 'max']); ok = ok[(ok['min'] == 0) & (ok['max'] == 1)].index
    z = d[d.s.isin(ok)].copy()
    z['x'] = (z[f] - z[f].mean()) / z[f].std()
    m = ConditionalLogit(z.err.values, z[['x']].values, groups=pd.factorize(z.s)[0]).fit(disp=0)
    return float(np.exp(m.params[0])), float(m.pvalues[0]), int(z.err.sum()), int((z.err == 0).sum()), z.s.nunique(), float(z[f].std()), z

rows = []
def add(test, **k): rows.append(dict(test=test, **k))

mc = X[X.group == 'matched_clean']
wp = X[X.group == 'within_paper_clean']
oth = X[X.group == 'other_mode_error']
e = err_set('confirmed')
ea = err_set('all')

# --- 1. exact replication of the headline numbers
for nm, es in [('confirmed', e), ('all', ea)]:
    OR, p, ne, nc, sd = logit_matched(es, mc, 'rep_rmsd')
    add(f'replicate matched logit ({nm})', feature='rep_rmsd', n_err=ne, n_ctl=nc, err_mean=es.rep_rmsd.mean(), ctl_mean=mc.rep_rmsd.mean(),
        err_median=es.rep_rmsd.median(), ctl_median=mc.rep_rmsd.median(), OR=OR, p=p, sd=sd)
    w = wp[wp.paper.isin(es.paper)]
    OR, p, ne, nc, ns, sd, z = clogit(es, w, 'rep_rmsd')
    add(f'replicate within-paper clogit ({nm})', feature='rep_rmsd', n_err=ne, n_ctl=nc, n_strata=ns, err_mean=z[z.err == 1].rep_rmsd.mean(),
        ctl_mean=z[z.err == 0].rep_rmsd.mean(), err_median=z[z.err == 1].rep_rmsd.median(), ctl_median=z[z.err == 0].rep_rmsd.median(), OR=OR, p=p, sd=sd,
        note=f'all within-paper ctl mean {w.rep_rmsd.mean():.3f} (n={len(w)})')

# --- 2. rank-based and transformed
add('MWU confirmed vs matched', feature='rep_rmsd', n_err=len(e), n_ctl=len(mc), err_median=e.rep_rmsd.median(), ctl_median=mc.rep_rmsd.median(),
    p=mannwhitneyu(e.rep_rmsd, mc.rep_rmsd).pvalue, AUC=mannwhitneyu(e.rep_rmsd, mc.rep_rmsd).statistic / len(e) / len(mc))
X['log_rmsd'] = np.log(X.rep_rmsd); e = err_set('confirmed'); mc = X[X.group == 'matched_clean']; wp = X[X.group == 'within_paper_clean']
OR, p, ne, nc, sd = logit_matched(e, mc, 'log_rmsd'); add('matched logit, log RMSD', feature='log_rmsd', n_err=ne, n_ctl=nc, OR=OR, p=p, sd=sd)
for k in [1, 3, 5]:
    top = e.nlargest(k, 'rep_rmsd')
    OR, p, ne, nc, sd = logit_matched(e.drop(top.index), mc, 'rep_rmsd')
    add(f'matched logit, drop top {k} error outliers', feature='rep_rmsd', n_err=ne, n_ctl=nc, err_mean=e.drop(top.index).rep_rmsd.mean(), OR=OR, p=p, sd=sd,
        note=','.join(top.pdb + top.chain))
# winsorise every chain at 2.5 A
X['rmsd_w'] = X.rep_rmsd.clip(upper=2.5); e = err_set('confirmed'); mc = X[X.group == 'matched_clean']; wp = X[X.group == 'within_paper_clean']; oth = X[X.group == 'other_mode_error']
OR, p, ne, nc, sd = logit_matched(e, mc, 'rmsd_w'); add('matched logit, winsorised at 2.5 A', feature='rmsd_w', n_err=ne, n_ctl=nc, OR=OR, p=p, sd=sd)

# --- 3. extra covariates (antibody type)
for extra, lab in [(' + C(sp)', 'species'), (' + v_id', 'germline identity'), (' + L3', 'CDR3 length'), (' + C(sp) + v_id + L3 + sars2', 'species+vid+L3+sars2')]:
    try:
        OR, p, ne, nc, sd = logit_matched(e.dropna(subset=['v_id', 'L3']), mc.dropna(subset=['v_id', 'L3']), 'rep_rmsd', extra)
        add(f'matched logit + {lab}', feature='rep_rmsd', n_err=ne, n_ctl=nc, OR=OR, p=p, sd=sd)
    except Exception as ex: add(f'matched logit + {lab}', note=f'fail {ex}')

# --- 4. paper-level aggregation
pe = e.groupby('paper').rep_rmsd.mean(); pc = mc.groupby('paper').rep_rmsd.mean()
add('paper-level means, errors vs matched-control papers (MWU)', feature='rep_rmsd', n_err=len(pe), n_ctl=len(pc), err_median=pe.median(), ctl_median=pc.median(),
    p=mannwhitneyu(pe, pc).pvalue)

# --- 5. leave-one-paper-out (matched and within paper)
lo_m, lo_w = [], []
w = wp[wp.paper.isin(e.paper)]
for pp in e.paper.unique():
    lo_m.append((pp,) + logit_matched(e, mc, 'rep_rmsd', drop_papers=(pp,))[:2])
    try: lo_w.append((pp,) + clogit(e, w, 'rep_rmsd', drop_papers=(pp,))[:2])
    except Exception: pass
lm = pd.DataFrame(lo_m, columns=['paper', 'OR', 'p']); lw = pd.DataFrame(lo_w, columns=['paper', 'OR', 'p'])
add('LOPO matched: min OR / max p', feature='rep_rmsd', OR=lm.OR.min(), p=lm.p.max(), note=f'paper giving max p: {lm.loc[lm.p.idxmax(), "paper"]}')
add('LOPO within-paper: min OR / max p', feature='rep_rmsd', OR=lw.OR.min(), p=lw.p.max(), note=f'paper giving max p: {lw.loc[lw.p.idxmax(), "paper"]}; OR range {lw.OR.min():.2f}-{lw.OR.max():.2f}')

# --- 6. within paper: same chain type strata, paired paper-level differences
OR, p, ne, nc, ns, sd, z = clogit(e, w, 'rep_rmsd', strata='paper_ct')
add('within paper x chain type clogit', feature='rep_rmsd', n_err=ne, n_ctl=nc, n_strata=ns, OR=OR, p=p, sd=sd,
    err_mean=z[z.err == 1].rep_rmsd.mean(), ctl_mean=z[z.err == 0].rep_rmsd.mean())
OR, p, ne, nc, ns, sd, z = clogit(e, w, 'rep_rmsd')
dif = z.groupby(['paper', 'err']).rep_rmsd.mean().unstack()
dif['d'] = dif[1] - dif[0]
add('within paper: paper-level paired difference (Wilcoxon)', feature='rep_rmsd', n_strata=len(dif), err_mean=dif[1].mean(), ctl_mean=dif[0].mean(),
    p=wilcoxon(dif.d).pvalue, note=f'papers with err>ctl: {(dif.d > 0).sum()}/{len(dif)}; median diff {dif.d.median():.3f} A')
dif.to_csv(f'{OUT}/within_paper_pairs.csv')
OR, p, ne, nc, ns, sd, z = clogit(e, w, 'rank_rmsd' if False else 'log_rmsd')
add('within paper clogit, log RMSD', feature='log_rmsd', n_err=ne, n_ctl=nc, n_strata=ns, OR=OR, p=p, sd=sd)
OR, p, ne, nc, ns, sd, z = clogit(e, w, 'rmsd_w')
add('within paper clogit, winsorised 2.5 A', feature='rmsd_w', n_err=ne, n_ctl=nc, n_strata=ns, OR=OR, p=p, sd=sd)

# --- 7. alternative metrics: away from the slip, reference panel
for f in ['strict_rmsd', 'fr12_rmsd', 'fr13a_rmsd', 'reg_FR1', 'reg_FR2', 'reg_FR3a', 'reg_FR3b', 'panel_mean', 'panel_min']:
    OR, p, ne, nc, sd = logit_matched(e, mc, f)
    add('matched logit (alt metric)', feature=f, n_err=ne, n_ctl=nc, err_mean=e[f].mean(), ctl_mean=mc[f].mean(), err_median=e[f].median(), ctl_median=mc[f].median(), OR=OR, p=p, sd=sd)
    OR, p, ne, nc, ns, sd, z = clogit(e, w, f)
    add('within paper clogit (alt metric)', feature=f, n_err=ne, n_ctl=nc, n_strata=ns, err_mean=z[z.err == 1][f].mean(), ctl_mean=z[z.err == 0][f].mean(), OR=OR, p=p, sd=sd)

# --- 8. by mode (matched)
for mname, ms in {'D+': ['Dplus_CDR3short'], 'D-': ['Dminus_CDR3long'], 'E': ['E_FR4bulge']}.items():
    em_ = err_set('confirmed', ms)
    for f in ['rep_rmsd', 'fr12_rmsd', 'panel_min']:
        OR, p, ne, nc, sd = logit_matched(em_, mc, f)
        add(f'matched logit mode {mname}', feature=f, n_err=ne, n_ctl=nc, err_mean=em_[f].mean(), err_median=em_[f].median(), ctl_median=mc[f].median(),
            OR=OR, p=p, sd=sd, AUC=mannwhitneyu(em_[f].dropna(), mc[f].dropna()).statistic / em_[f].notna().sum() / mc[f].notna().sum())

# --- 9. specificity: other first-occurrence error modes (EM) vs matched clean
for f in ['rep_rmsd', 'fr12_rmsd', 'panel_min']:
    o = oth.dropna(subset=[f])
    OR, p, ne, nc, sd = logit_matched(o, mc, f)
    add('OTHER error modes vs matched clean', feature=f, n_err=ne, n_ctl=nc, err_mean=o[f].mean(), ctl_mean=mc[f].mean(), err_median=o[f].median(),
        ctl_median=mc[f].median(), OR=OR, p=p, sd=sd)
    OR, p, ne, nc, sd = logit_matched(e, o, f)
    add('D/E confirmed vs OTHER error modes', feature=f, n_err=ne, n_ctl=nc, err_mean=e[f].mean(), ctl_mean=o[f].mean(), err_median=e[f].median(),
        ctl_median=o[f].median(), OR=OR, p=p, sd=sd)

# --- 10. selection by the 'confirmed' filter: does it pick high-RMSD chains?
ee = err_set('all')
nc_ = ee[ee.confirm == 'not confirmed']; cf = ee[ee.confirm != 'not confirmed']
add('confirmed vs not-confirmed D/E (MWU)', feature='rep_rmsd', n_err=len(cf), n_ctl=len(nc_), err_median=cf.rep_rmsd.median(), ctl_median=nc_.rep_rmsd.median(),
    err_mean=cf.rep_rmsd.mean(), ctl_mean=nc_.rep_rmsd.mean(), p=mannwhitneyu(cf.rep_rmsd, nc_.rep_rmsd).pvalue)
# resolution dependence in controls
r_ = spearmanr(mc.res, mc.rep_rmsd)
add('resolution vs RMSD in matched controls (Spearman)', feature='rep_rmsd', n_ctl=len(mc), OR=r_.correlation, p=r_.pvalue, note='OR column = rho')

T = pd.DataFrame(rows)
ok = T.p.notna() & ~T.test.str.startswith('LOPO') & ~T.test.str.startswith('resolution')
T.loc[ok, 'q_BH'] = multipletests(T.loc[ok, 'p'], method='fdr_bh')[1]
T.to_csv(f'{OUT}/verify_tests.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200); pd.set_option('display.max_colwidth', 70)
print(T.round(4).to_string())
lw.to_csv(f'{OUT}/lopo_within_paper.csv', index=False); lm.to_csv(f'{OUT}/lopo_matched.csv', index=False)
