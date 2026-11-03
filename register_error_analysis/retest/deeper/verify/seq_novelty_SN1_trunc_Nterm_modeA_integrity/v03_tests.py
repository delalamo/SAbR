"""Re-test SN1 with an independently built unit table (detector's own ANARCI numbering of SEQRES).

Unit = chains.csv 'unit' (paper x V region). EM unit = any EM chain. Case = unit with a first-occurrence
mode A run (runs_first_occurrence.csv) in an EM chain. Control = EM unit with no disagreement run of any kind
(err_any False in all chains), same chain types.
Unit features are taken from the case chain (cases) or the earliest-deposited chain (controls).
"""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_integrity'
T = pd.read_pickle(f'{SP}/deeper/cache/verify_SN1/chain_nterm.pkl')
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
ch = ch.merge(T, on=['pdb', 'chain'], how='left')
# covariate only: all-species germline call (investigator's chain table; species in chains.csv is human/mouse only)
sp = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables/chain_novelty.csv.gz', usecols=['pdb', 'chain', 'q_species_all'], low_memory=False)
ch = ch.merge(sp, on=['pdb', 'chain'], how='left')
r = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
ra = pd.read_csv(f'{SP}/deeper/runs_all.csv')
a = r[(r['mode'] == 'A_FR1Nterm')][['pdb', 'chain']].drop_duplicates().assign(modeA=True)
aall = ra[(ra['mode'] == 'A_FR1Nterm')][['pdb', 'chain']].drop_duplicates().assign(modeA_any=True)
ch = ch.merge(a, on=['pdb', 'chain'], how='left').merge(aall, on=['pdb', 'chain'], how='left')
ch['modeA'] = ch.modeA.fillna(False).astype(bool); ch['modeA_any'] = ch.modeA_any.fillna(False).astype(bool)
ch['deposit'] = pd.to_datetime(ch.deposit)
ch['isEM'] = ch.method == 'EM'
ch['caseA'] = ch.modeA & ch.isEM
ch = ch.sort_values(['unit', 'caseA', 'deposit', 'pdb', 'chain'], ascending=[True, False, True, True, True])
g = ch.groupby('unit')
U = g.first()
U['any_EM'] = g.isEM.any(); U['err_any_u'] = g.err_any.any(); U['caseA_u'] = g.caseA.any()
U['modeA_any_u'] = g.modeA_any.any(); U['res_med'] = g.res.median(); U['sars2_u'] = g.sars2.any()
U['vhh_u'] = g.vhh.max().fillna(0).astype(int)
U['dep_year'] = U.deposit.dt.year
U['yb'] = pd.cut(U.dep_year, [0, 2018, 2019, 2020, 2021, 2022, 2023, 2100],
                 labels=['<=2018', '2019', '2020', '2021', '2022', '2023', '>=2024']).astype(str)
U['nonhm'] = (~U.q_species_all.isin(['human', 'mouse'])).astype(int)
U['sars_i'] = U.sars2_u.astype(int)
U['trunc'] = (U.seq_first_imgt > 1).astype(int)
U['trunc_2_8'] = U.seq_first_imgt.between(2, 8).astype(int)
U['bgt1'] = (U.built_first_anarci > 1).astype(int)          # N-terminus (IMGT 1) not built, by ANARCI label
U['bgt3'] = (U.built_first_anarci > 3).astype(int)
U['unbuilt_full'] = ((U.trunc == 0) & (U.bgt1 == 1)).astype(int)   # full SEQRES but residue 1 unbuilt
U = U.reset_index()
U.to_pickle(f'{SP}/deeper/cache/verify_SN1/units_verify.pkl')
EM = U.any_EM
case = EM & U.caseA_u
cts = U[case].ctype.unique()
ctrl = EM & ~U.err_any_u & U.ctype.isin(cts)
log = open(f'{OUT}/v03_tests.log', 'w')
def P(*x):
    s = ' '.join(str(i) for i in x); print(s); log.write(s + '\n')
P('cases', case.sum(), 'controls', ctrl.sum())

def fisher(f, cs, ct, label):
    a1 = U.loc[cs, f].sum(); n1 = cs.sum(); a0 = U.loc[ct, f].sum(); n0 = ct.sum()
    orr, p = stats.fisher_exact([[a1, n1 - a1], [a0, n0 - a0]])
    P(f'{label:60s} {f:12s} err {a1}/{n1} ({a1/n1:.3f})  ctrl {a0}/{n0} ({a0/n0:.4f})  OR {orr:.2f}  Fisher p={p:.2g}')
    return dict(test=label, feature=f, err=f'{a1}/{n1}', ctrl=f'{a0}/{n0}', OR_raw=orr, p_raw=p)

def logit(formula, d, term, label):
    try:
        fit = smf.logit(formula, data=d).fit(disp=0, maxiter=300)
        o = np.exp(fit.params[term]); lo, hi = np.exp(fit.conf_int().loc[term]); p = fit.pvalues[term]
        P(f'{label:60s} {term:12s} OR {o:.2f} ({lo:.2f}-{hi:.2f}) p={p:.2g} n={int(d.y.sum())}/{int((d.y==0).sum())}')
        return dict(test=label, feature=term, OR_adj=o, lo=lo, hi=hi, p_adj=p, n_err=int(d.y.sum()), n_ctrl=int((d.y == 0).sum()))
    except Exception as e:
        P(label, term, 'fit failed', e); return dict(test=label, feature=term)

rows = []
D = U[case | ctrl].copy(); D['y'] = case[case | ctrl].astype(int)
M2 = ' + C(yb) + res_med + C(ctype) + vhh_u + nonhm + sars_i'
P('\n== 1. Reproduction')
rows.append(fisher('trunc', case, ctrl, '1a unit, seq_first>1 (detector ANARCI)'))
rows.append(logit('y ~ trunc' + M2, D, 'trunc', '1b M2'))
rows.append(fisher('trunc_2_8', case, ctrl, '1c seq_first 2-8 only'))

P('\n== 2. Overlap with "N-terminus not built" (first built residue, ANARCI label > 1)')
P(pd.crosstab([D.y, D.trunc], D.built_first_anarci.clip(upper=9), margins=True).to_string())
rows.append(fisher('bgt1', case, ctrl, '2a N-term unbuilt (first built IMGT>1), any SEQRES'))
rows.append(fisher('unbuilt_full', case, ctrl, '2b full SEQRES but IMGT1 unbuilt'))
rows.append(logit('y ~ trunc + unbuilt_full' + M2, D, 'trunc', '2c M2 + both: truncated SEQRES (ref: built from 1)'))
rows.append(logit('y ~ trunc + unbuilt_full' + M2, D, 'unbuilt_full', '2c M2 + both: full SEQRES, unbuilt'))
Db = D[D.bgt1 == 1].copy()
rows.append(logit('y ~ trunc' + M2, Db, 'trunc', '2d among N-term-unbuilt only: truncated vs full SEQRES'))
rows.append(logit('y ~ trunc_2_8 + C(fbc)' + M2, Db.assign(fbc=Db.built_first_anarci.clip(upper=5).astype(int).astype(str)), 'trunc_2_8', '2d2 among N-term-unbuilt, + first-built position, trunc 2-8'))
a1 = Db[(Db.y == 1)].trunc.sum(); n1 = (Db.y == 1).sum(); a0 = Db[(Db.y == 0)].trunc.sum(); n0 = (Db.y == 0).sum()
orr, p = stats.fisher_exact([[a1, n1 - a1], [a0, n0 - a0]])
P(f'2d raw among first-built>1: trunc err {a1}/{n1}  ctrl {a0}/{n0}  OR {orr:.2f} p={p:.2g}')
rows.append(dict(test='2d raw among first-built>1', feature='trunc', err=f'{a1}/{n1}', ctrl=f'{a0}/{n0}', OR_raw=orr, p_raw=p))
# error rate by category among all EM units of these chain types (cases+controls)
D['cat'] = np.where(D.trunc == 1, 'truncated SEQRES', np.where(D.bgt1 == 1, 'full SEQRES, IMGT1 unbuilt', 'full SEQRES, built from 1'))
P(D.groupby('cat').y.agg(['sum', 'size', 'mean']).to_string())
# stratified by first built position
D['fb'] = D.built_first_anarci.clip(upper=5)
for k, gg in D[D.fb >= 2].groupby('fb'):
    t = pd.crosstab(gg.trunc, gg.y).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    P(f'   first built IMGT={k}{"+" if k==5 else ""}: full SEQRES err {t.loc[0,1]}/{t.loc[0].sum()}  truncated err {t.loc[1,1]}/{t.loc[1].sum()}')
# Mantel-Haenszel across first-built strata 2,3,4,5+
from statsmodels.stats.contingency_tables import StratifiedTable
tabs = []
for k, gg in D[D.fb >= 2].groupby('fb'):
    t = pd.crosstab(gg.trunc, gg.y).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values.astype(float)
    if (t.sum(1) > 0).all(): tabs.append(t + 0.5 * (t == 0).any())
st = StratifiedTable(tabs)
P(f'2e MH OR truncated vs full SEQRES within first-built strata: {st.oddsratio_pooled:.2f} p={st.test_null_odds().pvalue:.2g}')
rows.append(dict(test='2e MH within first-built-position strata (2,3,4,5+)', feature='trunc', OR_adj=st.oddsratio_pooled, p_adj=st.test_null_odds().pvalue))

P('\n== 3. Alternative controls')
ctrl2 = EM & ~U.caseA_u & ~U.modeA_any_u & U.ctype.isin(cts)
rows.append(fisher('trunc', case, ctrl2, '3a ctrl = EM units without any mode A run (other errors allowed)'))
D2 = U[case | ctrl2].copy(); D2['y'] = case[case | ctrl2].astype(int)
rows.append(logit('y ~ trunc' + M2, D2, 'trunc', '3a M2'))
ctrl3 = ctrl & (U.built_first_anarci <= 8)   # controls with a built N-terminal strand (first built <= IMGT 8)
case3 = case & (U.built_first_anarci <= 8)
rows.append(fisher('trunc', case3, ctrl3, '3b both restricted to first built residue <= IMGT 8'))

P('\n== 4. De-duplication and influence')
# collapse identical V sequences within the same last author (repeat depositions w/o paper become separate units)
U['labab'] = U.last_author.fillna('?') + '|' + U.vseq.fillna('?')
dd = U[case | ctrl].copy(); dd['y'] = case[case | ctrl].astype(int)
dd = dd.sort_values('y', ascending=False).drop_duplicates('labab')
P('dedup by lab x V sequence: cases', dd.y.sum(), 'controls', (dd.y == 0).sum())
a1 = dd[dd.y == 1].trunc.sum(); n1 = dd.y.sum(); a0 = dd[dd.y == 0].trunc.sum(); n0 = (dd.y == 0).sum()
orr, p = stats.fisher_exact([[a1, n1 - a1], [a0, n0 - a0]])
P(f'4a dedup Fisher: err {a1}/{n1} ctrl {a0}/{n0} OR {orr:.2f} p={p:.2g}')
rows.append(dict(test='4a dedup lab x V seq', feature='trunc', err=f'{a1}/{n1}', ctrl=f'{a0}/{n0}', OR_raw=orr, p_raw=p))
rows.append(logit('y ~ trunc' + M2, dd, 'trunc', '4a dedup M2'))
# leave-one-lab-out (case labs)
res = []
for lab in D[D.y == 1].last_author.unique():
    d = D[D.last_author != lab]
    try:
        f = smf.logit('y ~ trunc' + M2, data=d).fit(disp=0, maxiter=300)
        res.append((lab, np.exp(f.params['trunc']), f.pvalues['trunc']))
    except Exception:
        pass
R = pd.DataFrame(res, columns=['lab', 'OR', 'p'])
P(f'4b leave-one-lab-out M2 (n={len(R)}): OR min {R.OR.min():.2f} max {R.OR.max():.2f}; p max {R.p.max():.2g}')
rows.append(dict(test='4b leave-one-lab-out M2', feature='trunc', OR_adj=R.OR.min(), p_adj=R.p.max()))
# drop the truncated cases that are not N-terminal slips (insertion-convention 22ge, X-residue record 9nyx, whole-chain shift 3jcx)
odd = {'22ge', '9nyx', '3jcx'}
cs4 = case & ~U.pdb.isin(odd)
rows.append(fisher('trunc', cs4, ctrl, '4c drop 3 truncated cases that are not N-terminal slips'))

P('\n== 5. Period and resolution structure of truncation among controls')
P(pd.crosstab(D[D.y == 0].yb, D[D.y == 0].trunc, normalize='index').round(3).to_string())
P(pd.crosstab(D[D.y == 1].yb, D[D.y == 1].trunc).to_string())

P('\n== 6. X-ray units (same definition)')
X = ~U.any_EM
cx = X & U.caseA_u.astype(bool)
# caseA is EM-only; recompute for X-ray
xa = ch[ch.modeA & ~ch.isEM].unit.unique()
cx = X & U.unit.isin(xa); ctx = X & ~U.err_any_u & U.ctype.isin(cts)
rows.append(fisher('trunc', cx, ctx, '6a X-ray units (3-5 A)'))

Rt = pd.DataFrame(rows)
Rt.to_csv(f'{OUT}/v03_tests.csv', index=False)
