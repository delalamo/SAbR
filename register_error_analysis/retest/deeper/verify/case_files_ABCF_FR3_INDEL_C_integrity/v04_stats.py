"""Re-test 'atypical FR3 length -> C_FR3' (cryo-EM units) under corrections.
 a. Feature recomputed from a fresh ANARCI run (human+mouse HMMs, vseq) - agreement with the investigator's cache feature.
 b. Control definitions: clean only (investigator) vs all units without a C first-occurrence run.
 c. Case-set corrections, cumulative:
      - confirmed artifacts (9bop, 9e89; investigator)
      - HIV FR3-insertion class 8c8t (3BNC117; same class as 4xnz, which the investigator excluded as a known artifact class;
        all 8/8 EM units with the 85A-D insertion carry a C run, 7 already classed copied/artifact upstream)
      - 8xlk (same V region as 7te9, identical 93-95 +1 run, deposited 2 years later by another lab -> copy, not first occurrence)
      - probable gap-placement artifacts 9yfu (gap 85; investigator's own 'probable artifact') and 7m3l (gap 89)
      - rabbit VH (FR3 37 is the rabbit germline norm, not an indel): 9ml5
 d. Method interaction (EM vs 3-5 A X-ray): if the indel effect were about cryo-EM modelling it should be stronger in EM.
 e. Leave-one-unit-out / leave-one-lab-out ranges, exact (Fisher) and Mantel-Haenszel (year-bin strata) p.
Outputs tables/v04_tests.csv"""
import pickle, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_integrity'
C = pd.read_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/chains_feat.pkl')
FA = pickle.load(open(f'{SP}/deeper/cache/verify_fr3_integrity/anarci_vseq_hm.pkl', 'rb'))
def fr3_fresh(vseq, ct):
    v = FA.get(vseq)
    if not v: return np.nan
    return sum(66 <= p <= 104 for p, i, a in v[1])
C['FR3_fresh'] = [fr3_fresh(v, ct) for v, ct in zip(C.vseq, C.ctype)]
norm = {'H': {38}, 'VHH': {38}, 'K': {36}, 'L': {36, 38}}
C['fr3_atyp_fresh'] = [int(f == f and int(f) not in norm[c]) for f, c in zip(C.FR3_fresh, C.cclass)]
C['rabbit'] = (C.species.astype(str).str.contains('rabbit', case=False)).astype(int)
print('chain-level agreement cache vs fresh FR3 length:', (C.FR3 == C.FR3_fresh).mean().round(4),
      ' fr3_atyp agreement:', (C.fr3_atyp == C.fr3_atyp_fresh).mean().round(4))
print(pd.crosstab(C.fr3_atyp, C.fr3_atyp_fresh))
dis = C[C.fr3_atyp != C.fr3_atyp_fresh][['pdb', 'chain', 'ctype', 'method', 'FR3', 'FR3_fresh', 'err_modes']]
print(dis.head(20).to_string()); dis.to_csv(f'{OUT}/tables/v04_feature_discordant_chains.csv', index=False)
C['ybin'] = pd.cut(C.year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
U = C.groupby('unit').agg(paper=('paper', 'first'), method=('method', 'first'), year=('year', 'first'), ybin=('ybin', 'first'),
                          res=('res', 'mean'), cclass=('cclass', 'first'), err_any=('err_any', 'max'), err_first=('err_first', 'max'),
                          modes=('modes', lambda s: set().union(*s)), fr3_atyp=('fr3_atyp', 'max'), fr3_atyp_fresh=('fr3_atyp_fresh', 'max'),
                          pdb=('pdb', 'first'), lab=('last_author', 'first'), rabbit=('rabbit', 'max'), species=('species', 'first')).reset_index()
U['yC'] = U.modes.map(lambda s: 'C_FR3' in s).astype(int)
E = U[U.method == 'EM'].copy()
rows = []
def test(name, d, x='fr3_atyp', y='yC', adj=True):
    d = d[d.cclass.isin(set(d[d[y] == 1].cclass))].copy()
    a = int(d[d[y] == 1][x].sum()); ne = int(d[y].sum()); c = int(d[d[y] == 0][x].sum()); nc = int((d[y] == 0).sum())
    orf, pf = fisher_exact([[a, ne - a], [c, nc - c]])
    tabs = []
    for _, g in d.groupby('ybin'):
        t = np.array([[((g[y] == 1) & (g[x] == 1)).sum(), ((g[y] == 1) & (g[x] == 0)).sum()], [((g[y] == 0) & (g[x] == 1)).sum(), ((g[y] == 0) & (g[x] == 0)).sum()]], float)
        if t.sum(1).min() > 0: tabs.append(t + 0.0)
    st = StratifiedTable(tabs); mh_or = st.oddsratio_pooled; mh_p = st.test_null_odds(correction=True).pvalue
    o = lo = hi = p = np.nan
    if adj and a >= 3:
        try:
            m = smf.logit(f'{y} ~ {x} + ybin + res + cclass', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
            o, lo, hi, p = np.exp(m.params[x]), *np.exp(m.conf_int().loc[x]), m.pvalues[x]
        except Exception: pass
    r = dict(test=name, err=f'{a}/{ne}', err_pct=round(100 * a / ne, 1), ctl=f'{c}/{nc}', ctl_pct=round(100 * c / nc, 2), OR_fisher=orf, p_fisher=pf,
             OR_MH_year=mh_or, p_MH_year=mh_p, OR_adj=o, lo=lo, hi=hi, p_adj=p)
    rows.append(r); return r
clean = lambda D: D[(D.yC == 1) | (D.err_any == False)]
base = clean(E)
test('a0 reproduce (cache feature, clean controls)', base)
test('a1 fresh-ANARCI feature (human+mouse), clean controls', base, x='fr3_atyp_fresh')
test('b1 controls = all EM units without a C first-occurrence run', E)
test('b2 controls = units without any first-occurrence run (incl. copied/artifact-only)', E[(E.yC == 1) | (E.err_first == False)])
excl = {}
excl['confirmed artifacts (9bop, 9e89)'] = {'9bop', '9e89'}
excl['+ HIV FR3-insertion class (8c8t)'] = {'8c8t'}
excl['+ cross-lab copy of 7te9 (8xlk)'] = {'8xlk'}
excl['+ probable gap-placement artifacts (9yfu, 7m3l)'] = {'9yfu', '7m3l'}
excl['+ rabbit FR3 norm (9ml5)'] = {'9ml5'}
drop = set()
for k, v in excl.items():
    drop |= v
    du = set(E[E.pdb.isin(drop) & (E.yC == 1)].unit)
    test(f'c excl. {k}', base[~base.unit.isin(du)])
# rabbit units: FR3 37 is the germline norm -> drop rabbit units entirely (cases and controls)
test('c* final case set, rabbit units dropped from both arms', base[~base.unit.isin(set(E[E.pdb.isin(drop) & (E.yC == 1)].unit)) & (base.rabbit == 0)])
# d. method interaction, units EM + X-ray 3-5 A, clean controls
D = clean(U).copy(); D['EM'] = (D.method == 'EM').astype(int)
for meth in ['X-ray']:
    test(f'd0 same test in 3-5 A X-ray units', D[D.method == meth], adj=False)
m = smf.logit('yC ~ fr3_atyp * EM + ybin + res', data=D).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(D.paper)[0]})
print(m.summary().tables[1])
inter = dict(test='d1 interaction fr3_atyp x EM (EM vs X-ray), logit + year bin + res, cluster SE', OR_adj=np.exp(m.params['fr3_atyp:EM']),
             lo=np.exp(m.conf_int().loc['fr3_atyp:EM'][0]), hi=np.exp(m.conf_int().loc['fr3_atyp:EM'][1]), p_adj=m.pvalues['fr3_atyp:EM'])
m2 = smf.logit('yC ~ EM + ybin + res', data=D[D.fr3_atyp == 0]).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(D[D.fr3_atyp == 0].paper)[0]})
inter['note'] = f'EM effect among typical-FR3 units OR {np.exp(m2.params["EM"]):.2f} (p={m2.pvalues["EM"]:.3g}); ' \
    f'first-occ C rate typical FR3: EM {D[(D.EM==1)&(D.fr3_atyp==0)].yC.mean():.4f} X-ray {D[(D.EM==0)&(D.fr3_atyp==0)].yC.mean():.4f}; ' \
    f'atypical FR3: EM {D[(D.EM==1)&(D.fr3_atyp==1)].yC.mean():.3f} X-ray {D[(D.EM==0)&(D.fr3_atyp==1)].yC.mean():.3f}'
rows.append(inter); print(inter)
# e. leave-one-out on the investigator's T1-excl set
b = base[~base.unit.isin(set(E[E.pdb.isin({'9bop', '9e89'}) & (E.yC == 1)].unit))]
pos = b[(b.yC == 1) & (b.fr3_atyp == 1)]
lo_p = []
for u in pos.unit:
    d = b[b.unit != u]
    a = int(d[d.yC == 1].fr3_atyp.sum()); ne = int(d.yC.sum()); c = int(d[d.yC == 0].fr3_atyp.sum()); nc = int((d.yC == 0).sum())
    lo_p.append(fisher_exact([[a, ne - a], [c, nc - c]])[1])
print('leave-one-positive-unit-out Fisher p range (T1-excl):', min(lo_p), max(lo_p))
T = pd.DataFrame(rows)
T['q_BH'] = multipletests(T.p_fisher.fillna(T.p_adj).fillna(1), method='fdr_bh')[1]
T.to_csv(f'{OUT}/tables/v04_tests.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 90)
print(T.drop(columns=['note'], errors='ignore').round(5).to_string())
# rabbit composition
print('rabbit among EM FR3-atyp clean controls:', int(base[(base.yC == 0) & (base.fr3_atyp == 1)].rabbit.sum()), 'of', int(base[(base.yC == 0)].fr3_atyp.sum()))
print(base[(base.yC == 0) & (base.fr3_atyp == 1)].species.value_counts().to_string())
