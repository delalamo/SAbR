"""Own recomputation of the most consequential numbers of the antibody_ordinary re-test.
Inputs: deeper/chains.csv.gz, my re-derived case set (v01), seq_novelty and literature_dean unit tables (raw features,
not the investigator's units_frame), struct_fv_frame.csv / struct_site_err.csv (per-Fv measurements only; contrasts
recomputed here)."""
import json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DEEP = '/home/user/SAbR/register_error_analysis/retest/deeper'
AO = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/antibody_ordinary/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_antibody_ordinary/tables'
rng = np.random.default_rng(7)
def wald(a, n1, c, n0):
    t = np.array([[a, n1 - a], [c, n0 - c]]) + (0.5 if min(a, n1 - a, c, n0 - c) == 0 else 0)
    lo = np.log(t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0])); se = np.sqrt((1 / t).sum())
    p = stats.fisher_exact([[a, n1 - a], [c, n0 - c]])[1]
    return f'{a}/{n1} ({a/n1:.1%}) vs {c}/{n0} ({c/n0:.1%}): OR {np.exp(lo):.2f} ({np.exp(lo-1.96*se):.2f}-{np.exp(lo+1.96*se):.2f}) Fisher p={p:.2g}'

CH = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
V = pd.read_csv(f'{OUT}/v01_paper_errors_rederived.csv')
strict = set(map(tuple, V[V.status == 'confirmed first occurrence'][['pdb', 'chain']].values))
incl = set(map(tuple, V[V.status.isin(['confirmed first occurrence', 'confirmed, loose-ledger doubt', 'copy of an earlier model', 'near copy (ledger)'])][['pdb', 'chain']].values))
CH['strict'] = [k in strict for k in zip(CH.pdb, CH.chain)]
CH['incl'] = [k in incl for k in zip(CH.pdb, CH.chain)]
T = pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx')
s1_all = set(T.PDB_CHAIN_ID.str.split(':').str[0].str.lower())
s1_v = set(V.pdb)
CH['win'] = CH.release.astype(str) <= '2022-04-05'
CH['ctrl_inv'] = CH.win & ~CH.pdb.isin(s1_v) & ~CH.err_any.astype(bool)      # as implemented in build_cases.py
CH['ctrl_any'] = CH.win & ~CH.pdb.isin(s1_all) & ~CH.err_any.astype(bool)    # as described: entry nowhere in Table S1
g = CH.groupby('unit')
U = pd.DataFrame({'paper': g.paper.first(), 'strict': g.strict.any(), 'incl': g.incl.any(),
                  'ctrl_inv': g.ctrl_inv.all(), 'ctrl_any': g.ctrl_any.all(),
                  'anyEM': g.method.agg(lambda s: (s == 'EM').any()),
                  'minres_x': g.apply(lambda d: d.loc[d.method != 'EM', 'res'].min())})
U['EM_case'] = CH[CH.strict].groupby('unit').method.agg(lambda s: (s == 'EM').any())
U['EM'] = U.EM_case.fillna(U.anyEM).astype(bool)
SN = pd.read_csv(f'{DEEP}/seq_novelty/tables/units.csv.gz', low_memory=False).set_index('unit')
LD = pd.read_csv(f'{DEEP}/literature_dean/tables/units_literature.csv.gz', low_memory=False).set_index('unit')
U = U.join(SN[['q_species_all', 'vhh_u', 'dep_year', 'method_u']]).join(LD[['rare_any', 'has_hq']])
print('units strict', int(U.strict.sum()), 'papers', U[U.strict].paper.nunique(), 'EM', int(U[U.strict].EM.sum()))
print('controls (implemented def)', int(U.ctrl_inv.sum()), '| controls (entry nowhere in Table S1)', int(U.ctrl_any.sum()))
lost = U[U.ctrl_inv & ~U.ctrl_any]
print('units that are controls only because their Table S1 entry has no >=50%-V error on a scanned chain:', len(lost),
      'EM', int(lost.EM.sum()))
res = {}
for lab, cc in [('ctrl_inv', 'ctrl_inv'), ('ctrl_any', 'ctrl_any')]:
    for feat in ['has_hq', 'rare_any', 'alpaca', 'nonhm']:
        U['alpaca'] = U.q_species_all.eq('alpaca').astype(int)
        U['nonhm'] = (~U.q_species_all.isin(['human', 'mouse'])).astype(int)
        for strat, m in [('all', slice(None)), ('EM', U.EM), ('X-ray', ~U.EM)]:
            d = U[m] if strat != 'all' else U
            ca = d[d.strict]; co = d[d[cc]]
            s = wald(int(ca[feat].astype(bool).sum()), len(ca), int(co[feat].astype(bool).sum()), len(co))
            res[f'{lab} {feat} {strat}'] = s
for k, v in res.items(): print(k, v)

# has_hq: the controls' own structure can make has_hq true when an X-ray chain is at <=3.0 A (scan is 3-5 A)
print('\n--- has_hq and own-structure leakage ---')
co = U[U.ctrl_inv]
print('controls with an X-ray chain at <=3.0 A:', int((co.minres_x <= 3.0).sum()), 'of', len(co),
      '; has_hq among them', round(co[co.minres_x <= 3.0].has_hq.mean(), 3), 'vs others', round(co[~(co.minres_x <= 3.0)].has_hq.mean(), 3))
ca = U[U.strict]
print('cases with an X-ray chain at <=3.0 A:', int((ca.minres_x <= 3.0).sum()))
d = U[~(U.minres_x <= 3.0)]
for strat, m in [('all', slice(None)), ('EM', d.EM), ('X-ray', ~d.EM)]:
    dd = d[m] if strat != 'all' else d
    print('excluding units with X-ray chain <=3.0 A,', strat, wald(int(dd[dd.strict].has_hq.sum()), int(dd.strict.sum()), int(dd[dd.ctrl_inv].has_hq.sum()), int(dd.ctrl_inv.sum())))
# year-adjusted logistic, paper-clustered, excluding <=3.0 A X-ray units
d = U[(U.strict | U.ctrl_inv) & ~(U.minres_x <= 3.0)].copy(); d['y'] = d.strict.astype(int)
d['yb'] = pd.cut(d.dep_year, [0, 2017, 2019, 2020, 2100], labels=['a', 'b', 'c', 'd']).astype(str)
d['hq'] = d.has_hq.astype(int); d['em'] = d.EM.astype(int)
m = smf.gee('y ~ hq + C(yb) + em', groups=pd.factorize(d.paper)[0], data=d, family=sm.families.Binomial()).fit()
print('GEE (excl. <=3.0 A X-ray units) has_hq OR', round(np.exp(m.params.hq), 3), np.round(np.exp(m.conf_int().loc['hq'].values), 3), 'p', round(m.pvalues.hq, 4))
# one vote per paper: per-paper mean among cases
pc = U[U.strict].groupby('paper').has_hq.mean()
print('per-paper case mean has_hq', round(pc.mean(), 3), 'papers with any', int((pc > 0).sum()), 'of', len(pc))
print('cases with has_hq:', U[U.strict & U.has_hq.astype(bool)].index.str[:60].tolist())

# rare length per paper
pc = U[U.strict].groupby('paper').rare_any.mean(); print('\nrare_any per-paper mean among cases', round(pc.mean(), 3), 'n papers', len(pc))
print('rare-length case units:', U[U.strict & (U.rare_any == 1)].index.str[:70].tolist())
print('alpaca case units:', U[U.strict & (U.alpaca == 1)].paper.tolist(), 'vhh_u', U[U.strict & (U.alpaca == 1)].vhh_u.tolist())
print('nonhm case species', U[U.strict & (U.nonhm == 1)].q_species_all.value_counts().to_dict())

# ---------------- orientation outliers (per-Fv measurements from struct_fv_frame)
print('\n--- ABangle outliers ---')
F = pd.read_csv(f'{AO}/struct_fv_frame.csv')
F = F[F.kind == 'Fv']
dom = '10.1038/s41422-021-00487-9'
case = F[(F.role == 'case') & F.strict & (F.method == 'EM')]
ctrl = F[(F.role != 'case') & (F.method == 'EM')]
print('roles', F.role.value_counts().to_dict())
print('EM strict case Fvs', len(case), 'papers', case.paper.nunique(), 'outliers', int(case.outlier99.sum()))
print('EM control Fvs', len(ctrl), 'outliers', int(ctrl.outlier99.sum()))
print(wald(int(case.outlier99.sum()), len(case), int(ctrl.outlier99.sum()), len(ctrl)))
print('case outliers by paper', case[case.outlier99 == 1].groupby('paper').size().to_dict())
o = case[case.outlier99 == 1]
print('distinct (Hseq,Lseq) among outliers', o[['Hseq', 'Lseq']].drop_duplicates().shape[0])
print(o[['pdb', 'H', 'L', 'paper', 'maha_dep', 'core_rms']].to_string())
dd = case[case.paper == dom]
print('dominant paper: Fvs', len(dd), 'distinct Fv seqs', dd[['Hseq', 'Lseq']].drop_duplicates().shape[0], 'entries', dd.pdb.nunique())
print(dd.groupby(['Hseq', 'Lseq']).agg(n=('pdb', 'size'), out=('outlier99', 'sum'), pdbs=('pdb', lambda s: ','.join(sorted(set(s))))).reset_index(drop=True).to_string())
# one-per-distinct-Fv-sequence-per-paper
cu = case.groupby(['paper', 'Hseq', 'Lseq']).outlier99.mean().reset_index()
print('one per distinct Fv sequence per paper: case outlier mean', round(cu.outlier99.mean(), 3), 'n', len(cu), '(any outlier >0.5:', int((cu.outlier99 > .5).sum()), ')')
cp = case.groupby('paper').outlier99.mean()
print('one vote per paper: case outlier mean', round(cp.mean(), 3), 'n papers', len(cp), 'ctrl', round(ctrl.outlier99.mean(), 3))
# paper-level permutation-style test: binomial of papers with any outlier vs control rate
cnt = case.groupby('paper').size()
p_any_ctrl = 1 - (1 - ctrl.outlier99.mean()) ** cnt
print('papers with >=1 outlier', int((case.groupby('paper').outlier99.max() > 0).sum()), 'expected', round(p_any_ctrl.sum(), 2))
# Mahalanobis diff
print('Mahalanobis case-ctrl', round(case.maha_dep.mean() - ctrl.maha_dep.mean(), 3))
# core rms
print('core_rms EM case-ctrl', round(case.core_rms.mean() - ctrl.core_rms.mean(), 3), '; without dominant paper',
      round(case[case.paper != dom].core_rms.mean() - ctrl.core_rms.mean(), 3),
      '; one per distinct Fv per paper', round(case.groupby(['paper', 'Hseq', 'Lseq']).core_rms.mean().mean() - ctrl.core_rms.mean(), 3))
xc = F[(F.role == 'case') & F.strict & (F.method != 'EM')]; xk = F[(F.role != 'case') & (F.method != 'EM')]
print('X-ray strict Fvs', len(xc), 'papers', xc.paper.nunique(), 'outliers', int(xc.outlier99.sum()), '| ctrl', len(xk), int(xk.outlier99.sum()))
# HC2
for p in ['HC2_sabr', 'LC1_sabr']:
    for nm, a, b in [('EM', case, ctrl), ('X-ray', xc, xk)]:
        dlt = a[p].mean() - b[p].mean()
        pa = a.groupby(['paper', 'Hseq', 'Lseq'])[p].mean()
        t = stats.ttest_ind(pa, b[p], equal_var=False)
        print(f'{p} {nm}: diff {dlt:+.2f}; one per distinct Fv per paper {pa.mean()-b[p].mean():+.2f} (n {len(pa)}) Welch p={t.pvalue:.3g}')
# ---------------- site uncertainty
print('\n--- ImmuneBuilder site uncertainty ---')
S = pd.read_csv(f'{AO}/struct_site_err.csv')
s = S[S.strict]
for nm, m in [('EM', s.method == 'EM'), ('X-ray', s.method != 'EM'), ('all', slice(None))]:
    d = s[m]
    pp = d.groupby('paper')['diff'].mean()
    # paper-cluster bootstrap of the site mean diff
    papers = d.paper.unique(); boots = []
    for _ in range(4000):
        smp = rng.choice(papers, len(papers))
        boots.append(pd.concat([d[d.paper == q] for q in smp])['diff'].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dedup = d.drop_duplicates(['paper', 'site_err', 'ctrl_site_err'])
    print(f'{nm}: sites {len(d)} papers {d.paper.nunique()} mean diff {d["diff"].mean():+.3f} cluster-boot ({lo:+.3f},{hi:+.3f}); '
          f'unique sites {len(dedup)} mean {dedup["diff"].mean():+.3f}; per-paper mean {pp.mean():+.3f} t p={stats.ttest_1samp(pp,0).pvalue:.3g}; '
          f'Wilcoxon per paper p={stats.wilcoxon(pp).pvalue if len(pp)>5 else float("nan"):.3g}; papers >0: {(pp>0).sum()}/{len(pp)}')
    if nm == 'EM':
        nd = d[d.paper != dom]; ppn = nd.groupby('paper')['diff'].mean()
        print('   EM without dominant paper per-paper mean', round(ppn.mean(), 3), 'p', round(stats.ttest_1samp(ppn, 0).pvalue, 4))
        print('   EM per-paper diffs', pp.round(2).sort_values().to_dict())
