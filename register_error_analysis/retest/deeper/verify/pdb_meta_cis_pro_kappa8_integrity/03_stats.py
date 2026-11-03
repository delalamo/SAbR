"""Statistical robustness of 'error kappa chains lack cis-Pro8' (EM, non-amyloid, Pro modelled at IMGT 8).
1. reproduce adjusted OR (year bin + resolution, cluster by paper)
2. + germline family, species, chain-level covariates
3. leave-one-paper-out (drop each error paper's chains, cases AND controls)
4. paper-level and entry-level collapsing (one vote per paper / entry)
5. within-paper and within-entry contrasts (error vs clean kappa chains in the SAME paper / entry)
6. resolution and period strata
7. cases restricted to non-FR1 errors, also excluding chains with ANY FR1 disagreement run (runs_all)
"""
import sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_integrity'
D = pd.read_csv(f'{OUT}/frame.csv.gz')
D = D[(D.method == 'EM') & (~D.amyloid) & (D.err_first | ~D.err_any) & (D.pro8 == True)].copy()
D['y'] = D.err_first.astype(int)
D['x'] = D.cis8.astype(int)
D['cluster'] = D.paper.fillna(D.pdb)
D['yr'] = pd.to_datetime(D.deposit).dt.year
D['ybin'] = np.where(D.yr <= 2018, '<=2018', np.where(D.yr >= 2024, '>=2024', D.yr.astype(str)))
D['fam'] = D.v_gene.fillna('NA').str.extract(r'^(\w+?V\d+)')[0].fillna('NA')
D['fam'] = np.where(D.groupby('fam').pdb.transform('size') < 30, 'other', D.fam)
modes = D.err_modes.fillna('')
D['modeA'] = modes.str.contains('A_FR1') & D.err_first
runs = pd.read_csv(f'{SP}/deeper/runs_all.csv')
fr1 = runs[(runs.imgt0 <= 26)]
fr1keys = set(zip(fr1.pdb, fr1.chain))
D['any_fr1_run'] = [(p, c) in fr1keys for p, c in zip(D.pdb, D.chain)]
out = []


def glm(d, covs, label, x='x'):
    d = d.dropna(subset=['y', x] + [c.replace('C(', '').replace(')', '') for c in covs])
    f = f'y ~ {x}' + ''.join(' + ' + c for c in covs)
    try:
        m = smf.glm(f, data=d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.cluster)[0]})
        b, se, p = m.params[x], m.bse[x], m.pvalues[x]
        r = dict(test=label, n_err=int(d.y.sum()), n_ctrl=int((1 - d.y).sum()), err_cis=d[d.y == 1][x].mean(), ctrl_cis=d[d.y == 0][x].mean(),
                 OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=p, n_err_clusters=d[d.y == 1].cluster.nunique())
    except Exception as ex:
        r = dict(test=label, note=str(ex)[:80])
    out.append(r)
    return r


base = ['C(ybin)', 'res']
glm(D, base, 'reproduce: all error chains, year bin + res')
noA = D[~D.modeA]
glm(noA, base, 'reproduce: excluding FR1-mode error chains')
noA2 = D[~(D.err_first & D.any_fr1_run)]
glm(noA2, base, 'excluding error chains with ANY FR1 disagreement run (incl. copied/artifact)')
glm(D, base + ['C(fam)', 'C(species)'], 'all errors + germline family + species')
glm(noA, base + ['C(fam)', 'C(species)'], 'non-FR1 errors + germline family + species')
D['l2papern'] = np.log2(D.entries_per_paper.fillna(1))
glm(noA.assign(l2papern=np.log2(noA.entries_per_paper.fillna(1))), base + ['C(fam)', 'C(species)', 'l2papern', 'sars2'], 'non-FR1 + family + species + entries/paper + SARS2')
# Firth-free exact: unadjusted Fisher
for lab, d in [('all', D), ('noFR1', noA)]:
    t = pd.crosstab(d.y, d.x).reindex(index=[1, 0], columns=[1, 0]).values
    out.append(dict(test=f'Fisher unadjusted ({lab})', OR=stats.fisher_exact(t)[0], p=stats.fisher_exact(t)[1], n_err=int(d.y.sum()), n_ctrl=int((1 - d.y).sum())))
# strata
for per, lo, hi in [('<=2019', 0, 2019), ('2020-21', 2020, 2021), ('>=2022', 2022, 2100)]:
    glm(noA[(noA.yr >= lo) & (noA.yr <= hi)], ['res'], f'non-FR1, period {per}')
for lab, lo, hi in [('res<3.5', 0, 3.5), ('res 3.5-4', 3.5, 4.0), ('res>=4', 4.0, 9)]:
    glm(noA[(noA.res >= lo) & (noA.res < hi)], ['C(ybin)'], f'non-FR1, {lab}')
# leave-one-paper-out over error papers (drop all chains of that paper)
lopo = []
for cl in D[D.y == 1].cluster.unique():
    for lab, d in [('all', D), ('noFR1', noA)]:
        if cl not in set(d[d.y == 1].cluster):
            continue
        dd = d[d.cluster != cl]
        try:
            m = smf.glm('y ~ x + C(ybin) + res', data=dd, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.cluster)[0]})
            lopo.append(dict(set=lab, dropped=cl, n_err_dropped=int(d[(d.cluster == cl)].y.sum()), OR=np.exp(m.params['x']), p=m.pvalues['x']))
        except Exception as ex:
            lopo.append(dict(set=lab, dropped=cl, note=str(ex)[:60]))
L = pd.DataFrame(lopo)
L.to_csv(f'{OUT}/leave_one_paper_out.csv', index=False)
for lab in ['all', 'noFR1']:
    l = L[L.set == lab]
    out.append(dict(test=f'leave-one-error-paper-out ({lab}): worst case', OR=l.OR.max(), p=l.p.max(), n_err=len(l)))
# paper-level collapse: each paper -> error status (has error kappa chain) and cis fraction of its kappa chains
P = D.groupby('cluster').agg(y=('y', 'max'), cis_frac=('x', 'mean'), yr=('yr', 'median'), res=('res', 'median'), n=('x', 'size')).reset_index()
P['ybin'] = np.where(P.yr <= 2018, '<=2018', np.where(P.yr >= 2024, '>=2024', P.yr.astype(int).astype(str)))
P['any_cis'] = (P.cis_frac > 0).astype(int)
m = smf.glm('y ~ any_cis + C(ybin) + res', data=P, family=sm.families.Binomial()).fit()
out.append(dict(test='PAPER level: paper has >=1 cis-Pro8 kappa chain (all chains incl. clean ones)', n_err=int(P.y.sum()), n_ctrl=int((1 - P.y).sum()),
                err_cis=P[P.y == 1].any_cis.mean(), ctrl_cis=P[P.y == 0].any_cis.mean(), OR=np.exp(m.params['any_cis']), p=m.pvalues['any_cis']))
# paper level using only error chains for error papers vs clean papers' chains (one vote per paper)
PE = D[D.y == 1].groupby('cluster').x.mean()
PC = D[(D.y == 0) & (~D.cluster.isin(PE.index))].groupby('cluster').x.mean()
u = stats.mannwhitneyu(PE, PC)
out.append(dict(test='PAPER level: mean cis8 of error chains per error paper vs clean papers (MWU)', n_err=len(PE), n_ctrl=len(PC),
                err_cis=PE.mean(), ctrl_cis=PC.mean(), p=u.pvalue))
# within paper: error papers that also have clean kappa chains
W = D[D.cluster.isin(PE.index)]
wp = W.groupby('cluster').apply(lambda g: pd.Series({'n_err': g.y.sum(), 'n_clean': (1 - g.y).sum(), 'cis_err': g[g.y == 1].x.mean(), 'cis_clean': g[g.y == 0].x.mean() if (g.y == 0).any() else np.nan}))
wp.to_csv(f'{OUT}/within_paper.csv')
both = wp.dropna()
# Mantel-Haenszel across papers with both
from statsmodels.stats.contingency_tables import StratifiedTable
tabs = []
for cl, g in W.groupby('cluster'):
    if g.y.nunique() < 2:
        continue
    t = np.array([[((g.y == 1) & (g.x == 1)).sum(), ((g.y == 1) & (g.x == 0)).sum()], [((g.y == 0) & (g.x == 1)).sum(), ((g.y == 0) & (g.x == 0)).sum()]], float)
    tabs.append(t)
st = StratifiedTable(tabs)
out.append(dict(test='WITHIN PAPER (MH across error papers with clean kappa chains)', n_err=int(sum(t[0].sum() for t in tabs)), n_ctrl=int(sum(t[1].sum() for t in tabs)),
                err_cis=sum(t[0, 0] for t in tabs) / sum(t[0].sum() for t in tabs), ctrl_cis=sum(t[1, 0] for t in tabs) / sum(t[1].sum() for t in tabs),
                OR=st.oddsratio_pooled, p=st.test_null_odds().pvalue, n_err_clusters=len(tabs)))
# within entry
tabs = []
for pdb, g in D.groupby('pdb'):
    if g.y.nunique() < 2:
        continue
    t = np.array([[((g.y == 1) & (g.x == 1)).sum(), ((g.y == 1) & (g.x == 0)).sum()], [((g.y == 0) & (g.x == 1)).sum(), ((g.y == 0) & (g.x == 0)).sum()]], float)
    tabs.append(t)
if tabs:
    st = StratifiedTable(tabs)
    out.append(dict(test='WITHIN ENTRY (MH across entries with error and clean kappa chains)', n_err=int(sum(t[0].sum() for t in tabs)), n_ctrl=int(sum(t[1].sum() for t in tabs)),
                    err_cis=sum(t[0, 0] for t in tabs) / sum(t[0].sum() for t in tabs), ctrl_cis=sum(t[1, 0] for t in tabs) / sum(t[1].sum() for t in tabs),
                    OR=st.oddsratio_pooled if len(tabs) > 1 else np.nan, p=st.test_null_odds().pvalue if len(tabs) > 1 else np.nan, n_err_clusters=len(tabs)))
# entry level
EN = D.groupby('pdb').agg(y=('y', 'max'), x=('x', 'mean'), yr=('yr', 'first'), res=('res', 'first'), cluster=('cluster', 'first')).reset_index()
EN['ybin'] = np.where(EN.yr <= 2018, '<=2018', np.where(EN.yr >= 2024, '>=2024', EN.yr.astype(str)))
EN['x'] = (EN.x > 0.5).astype(int)
glm(EN, base, 'ENTRY level: majority of kappa Pro8 cis (one row per entry)')
# unit level (antibody-in-paper)
U = D.groupby('unit').agg(y=('y', 'max'), x=('x', 'mean'), yr=('yr', 'median'), res=('res', 'median'), cluster=('cluster', 'first')).reset_index()
U['ybin'] = np.where(U.yr <= 2018, '<=2018', np.where(U.yr >= 2024, '>=2024', U.yr.astype(int).astype(str)))
U['x'] = (U.x > 0.5).astype(int)
glm(U, base, 'UNIT level (antibody-in-paper): majority cis')
R = pd.DataFrame(out)
ok = R.p.notna()
R.loc[ok, 'q_BH'] = multipletests(R.loc[ok, 'p'], method='fdr_bh')[1]
R.to_csv(f'{OUT}/stats.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 75)
print(R.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print(L.groupby('set')[['OR', 'p']].describe().T)
print(both)
