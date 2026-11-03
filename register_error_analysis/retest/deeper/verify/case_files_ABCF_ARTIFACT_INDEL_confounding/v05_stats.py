"""Statistics for the ARTIFACT_INDEL verification (all exact tests; BH across the tests in this file).

T1  High-resolution X-ray (<=2.75 A) entries carrying a claimed artifact pattern (FR3 gap 84, 84+85, 80+84; FR1 gap 26)
    vs the FR3-gap-81 control: SAbR/ANARCI disagreement (>=3 residues) in the window (v03, scan-default ANARCI).
    Entry level and one-per-antibody level (antibody = heavy CDR3 + FR3 string).
T2  3-5 A scan chains (existing SAbR output, v02), one per antibody, claimed patterns vs FR3-gap-81, overall and
    stratified by method (EM / X-ray) and by period (<=2019, 2020-21, >=2022); Mantel-Haenszel test across strata.
T3  Map-free predicted models (v04): claimed-artifact run windows vs all other C_FR3/F_CDR1 first-occurrence windows,
    reproduced (>=3 disagreements in window) by IgFold / ImmuneBuilder / both.
T4  ANARCI species configuration: the same 20 FR3-gap-81 high-res X-ray controls renumbered with ANARCI all-species
    HMMs (v03b) vs the scan default (v03): paired, exact McNemar.
Output: tables/v05_tests.csv
"""
import pickle, json, numpy as np, pandas as pd
from scipy.stats import fisher_exact, binomtest
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_confounding'
CLAIM = {'FR3_84', 'FR3_84_85', 'FR3_80_84', 'FR1_26'}
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
CEN = pd.read_csv(f'{OUT}/tables/v01_pattern_census.csv')
# antibody key: CDR3 (IMGT 105-117) + FR3 (IMGT 66-104) from the cached heavy numbering
abkey = {}
for seq, doms in AN.items():
    for d in doms or []:
        if d[0] != 'H': continue
        if d[4]:
            abkey[''.join(a for p, i, a in d[4] if a != '-')] = ''.join(a for p, i, a in d[4] if a != '-' and 66 <= int(p) <= 117)
        break
CEN['ab'] = CEN.vseq.map(abkey)
ab = dict(zip(zip(CEN.pdb, CEN.chain), CEN.ab))
rows = []
def add(test, a1, n1, a0, n0, note='', strata=None):
    t = [[a1, n1 - a1], [a0, n0 - a0]]
    orr, p = fisher_exact(t)
    r = dict(test=test, claimed=f'{a1}/{n1}', control=f'{a0}/{n0}', OR=orr, p=p, note=note)
    rows.append(r)
# ---- T1
H = pd.read_csv(f'{OUT}/tables/v03_hires_sabr.csv')
H = H[H.res <= 2.75].copy(); H['dis3'] = H.n_dis >= 3; H['claim'] = H.pattern.isin(CLAIM)
H['ab'] = [ab.get((p, c)) for p, c in zip(H.pdb, H.chain)]
e1 = H[H.claim]; e0 = H[H.pattern == 'FR3_81']
add('T1 hi-res X-ray <=2.75 A, entries: disagreement in window', e1.dis3.sum(), len(e1), e0.dis3.sum(), len(e0),
    'claimed: ' + ' '.join(f'{p}:{int(d)}' for p, d in zip(e1.pdb, e1.n_dis)))
a1 = e1.groupby('ab').dis3.max(); a0 = e0.groupby('ab').dis3.max()
add('T1b hi-res X-ray <=2.75 A, one per antibody', int(a1.sum()), len(a1), int(a0.sum()), len(a0), 'exception: 7kbk (alpaca VHH, 2.09 A)')
# ---- T2
S = pd.read_csv(f'{OUT}/tables/v02_scan_chains.csv')
S = S[(S.n_win > 0) & S.pattern.isin(CLAIM | {'FR3_81'})].copy()
S['claim'] = S.pattern.isin(CLAIM); S['ab'] = [ab.get((p, c)) for p, c in zip(S.pdb, S.chain)]
S['EM'] = S.method_idx.str.contains('ELECTRON'); S['period'] = pd.cut(S.year, [0, 2019, 2021, 2100], labels=['<=2019', '2020-21', '>=2022']).astype(str)
U = S.groupby(['ab', 'claim']).agg(dis3=('dis3', 'max'), EM=('EM', 'max'), period=('period', 'first'), year=('year', 'min')).reset_index()
add('T2 3-5 A scan, one per antibody', int(U[U.claim].dis3.sum()), int(U.claim.sum()), int(U[~U.claim].dis3.sum()), int((~U.claim).sum()),
    'claimed exception: 9ml5 (rabbit EM, run at 85-95 instead)')
for strat in ['EM', 'period']:
    tabs = []
    for k, g in U.groupby(strat):
        x1, n1 = int(g[g.claim].dis3.sum()), int(g.claim.sum()); x0, n0 = int(g[~g.claim].dis3.sum()), int((~g.claim).sum())
        if n1 and n0:
            add(f'T2 stratum {strat}={k}', x1, n1, x0, n0); tabs.append(np.array([[x1, n1 - x1], [x0, n0 - x0]]) + 0.5)
    st = StratifiedTable(tabs)
    rows.append(dict(test=f'T2 Mantel-Haenszel across {strat} strata (0.5 added)', OR=st.oddsratio_pooled, p=st.test_null_odds(correction=True).pvalue,
                     note=f'{len(tabs)} strata'))
# ---- T3
P = pd.read_csv(f'{OUT}/tables/v04_predicted_models_by_run.csv')
for c, lab in [('ig3', 'IgFold'), ('ib3', 'ImmuneBuilder'), ('both', 'both')]:
    add(f'T3 predicted model reproduces run window ({lab})', int(P[P.claimed_artifact][c].sum()), int(P.claimed_artifact.sum()),
        int(P[~P.claimed_artifact][c].sum()), int((~P.claimed_artifact).sum()),
        'non-claimed reproduced: ' + ','.join(P[(~P.claimed_artifact) & P[c]].pdbs))
# ---- T4
Hb = pd.read_csv(f'{OUT}/tables/v03b_hires_sabr_allspeciesHMM.csv')
m = pd.read_csv(f'{OUT}/tables/v03_hires_sabr.csv').query("pattern=='FR3_81'")[['pdb', 'n_dis']].merge(Hb[['pdb', 'n_dis']], on='pdb', suffixes=('_default', '_allsp'))
b = int(((m.n_dis_default < 3) & (m.n_dis_allsp >= 3)).sum()); c = int(((m.n_dis_default >= 3) & (m.n_dis_allsp < 3)).sum())
rows.append(dict(test='T4 FR3-gap-81 rabbit X-ray controls: disagree with ANARCI all-species vs default HMMs (paired)',
                 claimed=f'{int((m.n_dis_allsp>=3).sum())}/{len(m)} (all-species)', control=f'{int((m.n_dis_default>=3).sum())}/{len(m)} (default)',
                 p=binomtest(b, b + c, 0.5).pvalue if b + c else 1.0, note='exact McNemar; ANARCI rabbit HMM moves the gap 81->84 in every rabbit VH'))
R = pd.DataFrame(rows)
R['q_BH'] = multipletests(R.p.fillna(1), method='fdr_bh')[1]
R.to_csv(f'{OUT}/tables/v05_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(R.to_string())
print(U.groupby(['claim', 'EM', 'period']).agg(n=('dis3', 'size'), dis=('dis3', 'sum')))
