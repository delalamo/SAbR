"""Confounding / definition-overlap re-tests of FR3_INDEL_C.

Unit = antibody V region in one paper (chains.csv 'unit'), cryo-EM, AL amyloid excluded (as investigator).
Outcome C_first = some chain of the unit carries a first-occurrence C_FR3 run. Exposure fr3_atyp from SEQRES ANARCI.

R0  investigator frame reproduced: controls = units with no disagreement run of any kind (err_any False);
    logit + ybin + res + cclass, cluster by paper
R1  + sars2 ; cluster by lab (last author) instead of paper; one unit per paper (first unit)
R2  Mantel-Haenszel over ybin x cclass strata; exact Fisher (crude)
R3  control = every EM unit without a first-occurrence C run (no conditioning on err_any) -> tests control depletion
R4  exclude from BOTH arms units whose FR3 indel pattern is convention-prone: SAbR/ANARCI disagree (>=3 FR3 residues)
    in >=50% of OTHER papers' chains with that pattern (leave-own-paper-out, from tables/fr3_pattern_chains)
    (a) unknown patterns kept; (b) unknown patterns also dropped
R5  outcome restricted to "remote" C runs: run does not touch the ANARCI indel (both ends > 2 residues from every
    ANARCI gap/insertion position in 66-104) - i.e. not the gap-placement window
R6  location specificity: fr3_atyp in error units of every other mode vs the same clean controls
R7  X-ray 3-5 A units: same contrast (if numbers allow)
R8  rate view: P(C_first | fr3_atyp) vs P(C_first | typical) in EM, by period
Descriptives: run length, |shift|, next-to-unbuilt and local map fit (Q-score percentile) of the C runs, FR3-atypical
vs FR3-typical C units.
Outputs: tables/tests.csv, tables/c_units_detail.csv
"""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DE}/verify/case_files_ABCF_FR3_INDEL_C_confounding'
U = pd.read_csv(f'{OUT}/tables/units_em.csv.gz')
CF = pd.read_csv(f'{OUT}/tables/chain_features.csv.gz', low_memory=False)
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
amy = set(pd.read_csv(f'{DE}/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
RF = RF[~RF.pdb.isin(amy)]
unit_of = dict(zip(zip(CF.pdb, CF.chain), CF.unit))
paper_of = dict(zip(zip(CF.pdb, CF.chain), CF.paper))
RF['unit'] = [unit_of.get(k) for k in zip(RF.pdb, RF.chain)]
U['first_modes'] = U.first_modes.fillna('')
U['sars2'] = U.sars2.fillna(0).astype(int)
U['lab'] = U.lab.fillna('NA')
rows = []


def fit(d, name, x='fr3_atyp', y='y', extra='', groups='paper', note=''):
    ne, nc = int(d[y].sum()), int((d[y] == 0).sum())
    a = int(d[(d[y] == 1)][x].sum()); b = ne - a; c = int(d[(d[y] == 0)][x].sum()); dd = nc - c
    orc, pf = fisher_exact([[a, b], [c, dd]])
    r = dict(test=name, n_error=ne, n_control=nc, error_value=f'{a/ne:.1%} ({a}/{ne})' if ne else '', control_value=f'{c/nc:.2%} ({c}/{nc})',
             OR_crude=orc, p_fisher=pf, note=note)
    try:
        if min(a, b, c, dd) == 0: raise ValueError('zero cell')
        m = smf.logit(f'{y} ~ {x} + ybin + res + cclass{extra}', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d[groups])[0]})
        r.update(OR_adj=np.exp(m.params[x]), lo=np.exp(m.conf_int().loc[x, 0]), hi=np.exp(m.conf_int().loc[x, 1]), p_adj=m.pvalues[x])
    except Exception as ex:
        r.update(OR_adj=np.nan, lo=np.nan, hi=np.nan, p_adj=np.nan, note=(note + ' logit n/a: ' + str(ex)[:60]).strip())
    rows.append(r)
    return r


def frame(Ux, mode='C_FR3', ctrl='clean', ysub=None):
    d = Ux.copy()
    d['y'] = d.first_modes.str.contains(mode).astype(int)
    if ysub is not None: d['y'] = (d.y & ysub(d)).astype(int)
    if ctrl == 'clean':
        d = d[(d.y == 1) | (~d.err_any.astype(bool))]
    else:  # every unit without a first-occurrence run of this mode
        d = d[(d.y == 1) | (~d.first_modes.str.contains(mode))]
    d = d[d.cclass.isin(set(d[d.y == 1].cclass))]
    return d


# R0 / R1 / R2
d0 = frame(U)
fit(d0, 'R0 investigator frame (ybin+res+cclass, cluster paper)')
fit(d0, 'R1a + sars2', extra=' + sars2')
fit(d0, 'R1b cluster by lab', groups='lab')
d1 = d0.sort_values('y', ascending=False).drop_duplicates('paper')
fit(d1, 'R1c one unit per paper (error unit kept if any)')
tabs = []
for _, g in d0.groupby(['ybin', 'cclass']):
    t = np.array([[((g.fr3_atyp == 1) & (g.y == 1)).sum(), ((g.fr3_atyp == 1) & (g.y == 0)).sum()],
                  [((g.fr3_atyp == 0) & (g.y == 1)).sum(), ((g.fr3_atyp == 0) & (g.y == 0)).sum()]], dtype=float)
    if (t.sum(0) > 0).all() and (t.sum(1) > 0).all(): tabs.append(t + 0.5 * (t == 0).any())
st = StratifiedTable(tabs)
rows.append(dict(test='R2 Mantel-Haenszel over ybin x cclass', OR_adj=st.oddsratio_pooled, lo=st.oddsratio_pooled_confint()[0],
                 hi=st.oddsratio_pooled_confint()[1], p_adj=st.test_null_odds().pvalue, n_error=int(d0.y.sum()), n_control=int((d0.y == 0).sum())))
# period-restricted
for yb in ['2020-21', '2022-23', '>=2024']:
    fit(d0[d0.ybin == yb], f'R2b period {yb} only')
fit(d0[d0.cclass == 'H'], 'R2c heavy chains only (non-VHH)')
fit(d0[d0.sars2 == 0], 'R2d non-SARS-CoV-2 units only')

# R3 control without conditioning on err_any
d3 = frame(U, ctrl='all')
fit(d3, 'R3 control = all EM units without a first-occurrence C run')

# R4 convention-prone patterns (leave-own-paper-out disagreement rate)
P = pd.read_csv(f'{OUT}/tables/fr3_pattern_chains.csv.gz')
P = P[P.contiguous & P.spanned].copy()
P['light'] = P.ct.isin(['K', 'L'])
P['paper'] = [paper_of.get(k, k[0]) for k in zip(P.pdb, P.chain)]
P['paper'] = P.paper.fillna(P.pdb)
# add investigator's high-resolution SAbR checks (s09) as extra chains
HG = pd.read_csv(f'{DE}/case_files_ABCF/tables/hires_gap_check.csv')
pmap = {'FR3_84': '(84,)', 'FR3_80_84': '(80, 84)', 'FR3_84_85': '(84, 85)', 'FR3_81': '(81,)', 'FR3_96': '(96,)', 'FR3_82': '(82,)'}
add = HG[HG.pattern.isin(pmap)].assign(pattern=lambda x: x.pattern.map(pmap), light=False, paper=lambda x: x.pdb, dis3=lambda x: x.dis_FR3_74_100 >= 3)
P = pd.concat([P[['pdb', 'chain', 'light', 'pattern', 'paper', 'dis3']], add[['pdb', 'chain', 'light', 'pattern', 'paper', 'dis3']]]).drop_duplicates(['pdb', 'chain'])
U['light'] = U.cclass.isin(['K', 'L'])


def loo(r):
    if not r.fr3_atyp: return np.nan
    g = P[(P.pattern == r.fr3_pattern) & (P.light == r.light) & (P.paper != r.paper)]
    return g.dis3.mean() if len(g) else np.nan
U['pat_loo_dis'] = [loo(r) for r in U.itertuples()]
U['pat_loo_n'] = [len(P[(P.pattern == r.fr3_pattern) & (P.light == r.light) & (P.paper != r.paper)]) if r.fr3_atyp else 0 for r in U.itertuples()]
U['conv_prone'] = (U.pat_loo_dis >= 0.5).astype(int)
U['conv_unknown'] = (U.fr3_atyp.astype(bool) & U.pat_loo_dis.isna()).astype(int)
fit(frame(U[U.conv_prone == 0]), 'R4a drop convention-prone FR3 patterns from both arms (unknown kept)')
fit(frame(U[(U.conv_prone == 0) & (U.conv_unknown == 0)]), 'R4b drop convention-prone and unknown-behaviour FR3 patterns')

# R5 remote C runs
def gap_positions(pattern):
    if not isinstance(pattern, str) or not pattern: return []
    g, _, ins = pattern.partition('+')
    pos = [int(t) for t in g.strip('()').replace(' ', '').split(',') if t]
    pos += [int(''.join(ch for ch in t if ch.isdigit())) for t in ins.split(',') if t]
    return pos
pat_of_unit = dict(zip(U.unit, U.fr3_pattern))
cr = RF[RF['mode'] == 'C_FR3'].copy()
cr['gaps'] = [gap_positions(pat_of_unit.get(u)) for u in cr.unit]
cr['touch_indel'] = [any(r.imgt0 - 2 <= g + 0.5 and g - 0.5 <= r.imgt1 + 2 for g in gs) for r, gs in zip(cr.itertuples(), cr.gaps)]
remote_units = set(cr[~cr.touch_indel].unit)
fit(frame(U, ysub=lambda d: d.unit.isin(remote_units).astype(int)), 'R5 outcome = C runs not touching the ANARCI indel (+-2)')

# R6 other modes, same clean controls
for m in ['A_FR1Nterm', 'B_CDR2FR3', 'F_CDR1', 'Dplus_CDR3short', 'Dminus_CDR3long', 'E_FR4bulge', 'other']:
    fit(frame(U, mode=m), f'R6 fr3_atyp -> {m} (location control)')
    fit(frame(U, mode=m, ctrl='all'), f'R6 fr3_atyp -> {m}, control = all units without {m}')

# R7 X-ray
C = CF[CF.method == 'X-ray'].copy()
C = C[~C.pdb.isin(amy)]
C['ybin'] = pd.cut(C.year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
C['C_first'] = C.first_modes.astype(str).str.contains('C_FR3')
UX = C.groupby('unit').agg(paper=('paper', 'first'), ybin=('ybin', 'first'), res=('res', 'mean'), cclass=('cclass', 'first'),
                           err_any=('err_any', 'max'), first_modes=('first_modes', lambda s: ';'.join(map(str, s))), fr3_atyp=('fr3_atyp', 'max')).reset_index()
UX['first_modes'] = UX.first_modes.str.replace("'", '')
dx = frame(UX)
rows.append(dict(test='R7 X-ray 3-5 A units (crude)', n_error=int(dx.y.sum()), n_control=int((dx.y == 0).sum()),
                 error_value=f'{dx[dx.y==1].fr3_atyp.mean():.1%} ({int(dx[dx.y==1].fr3_atyp.sum())}/{int(dx.y.sum())})',
                 control_value=f'{dx[dx.y==0].fr3_atyp.mean():.2%} ({int(dx[dx.y==0].fr3_atyp.sum())}/{int((dx.y==0).sum())})',
                 OR_crude=fisher_exact(pd.crosstab(dx.fr3_atyp, dx.y).loc[[1, 0], [1, 0]].values)[0],
                 p_fisher=fisher_exact(pd.crosstab(dx.fr3_atyp, dx.y).loc[[1, 0], [1, 0]].values)[1]))

T = pd.DataFrame(rows)
T['p_use'] = T.p_adj.fillna(T.p_fisher)
T['q_BH'] = multipletests(T.p_use.fillna(1), method='fdr_bh')[1]
T.to_csv(f'{OUT}/tables/tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 75); pd.set_option('display.max_rows', 100)
print(T[['test', 'n_error', 'n_control', 'error_value', 'control_value', 'OR_crude', 'OR_adj', 'lo', 'hi', 'p_use', 'q_BH', 'note']].round(4).to_string())

# R8 rates
d = frame(U, ctrl='all')
print('\nR8 P(C_first) by fr3_atyp x period (all EM units):')
print(d.groupby(['ybin', 'fr3_atyp']).y.agg(['sum', 'count', 'mean']).round(4))
print('all EM units with fr3_atyp, by status:')
print(pd.crosstab(U.fr3_atyp, np.select([U.first_modes.str.contains('C_FR3'), U.err_first.astype(bool), U.err_any.astype(bool)], ['C_first', 'other_first', 'copied/artifact only'], 'clean')))

# ---------- descriptives of the C units: run length, shift, unbuilt neighbour, local fit
LQ = {}
need = set(cr.pdb)
for line in open(f'{SP}/retest/local_all.jsonl'):
    x = json.loads(line)
    if x['pdb'] not in need: continue
    f = x.get('feats') or {}; key = 'Q_SCORE' if 'Q_SCORE' in f else ('RSCC' if 'RSCC' in f else None)
    if key is None: continue
    am = x['auth_map']
    LQ[(x['pdb'], x['chain'])] = {am[int(k) - 1]: v for k, v in f[key].items() if v is not None and 0 < int(k) <= len(am)}
REC = {}
for fn in ['new.jsonl', 'snapshot_rest.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        if json.loads(line)['pdb'] not in need: continue
        x = json.loads(line)
        if x.get('residues'): REC[(x['pdb'], x['chain'])] = x['residues']
det = []
for r in cr[cr.method == 'EM'].itertuples():
    rec = REC.get((r.pdb, r.chain)); q = LQ.get((r.pdb, r.chain))
    qwin = qpct = np.nan
    if rec and q:
        vres = [x for x in rec if x[4] is not None and 1 <= x[4] <= 128]
        allq = [q.get(str(x[0]) + (x[1] or '')) for x in vres]
        allq = [v for v in allq if v is not None]
        win = [q.get(str(x[0]) + (x[1] or '')) for x in vres if r.imgt0 <= x[4] <= r.imgt1]
        win = [v for v in win if v is not None]
        if win and allq:
            qwin = float(np.mean(win)); qpct = float(np.mean([v < qwin for v in allq]))
    det.append(dict(pdb=r.pdb, chain=r.chain, unit=r.unit, imgt0=r.imgt0, imgt1=r.imgt1, delta=r.delta, length=r.length,
                    next_to_unbuilt=r.next_to_gap_origin, touch_indel=r.touch_indel, fr3_pattern=pat_of_unit.get(r.unit),
                    fr3_atyp=int(U.set_index('unit').fr3_atyp.get(r.unit, 0)), conv_prone=int(U.set_index('unit').conv_prone.get(r.unit, 0)),
                    pat_loo_dis=U.set_index('unit').pat_loo_dis.get(r.unit), pat_loo_n=U.set_index('unit').pat_loo_n.get(r.unit),
                    q_win=qwin, q_win_pct_in_domain=qpct))
DT = pd.DataFrame(det)
DT.to_csv(f'{OUT}/tables/c_units_detail.csv', index=False)
du = DT.sort_values('length', ascending=False).drop_duplicates('unit')
print('\nC units (EM), one run per unit:')
print(du.sort_values(['fr3_atyp', 'length']).to_string())
a, b = du[du.fr3_atyp == 1], du[du.fr3_atyp == 0]
print('run length median atyp/typ', a.length.median(), b.length.median(), 'MWU p', mannwhitneyu(a.length, b.length).pvalue)
print('|delta| median atyp/typ', a.delta.abs().median(), b.delta.abs().median())
print('next to unbuilt atyp/typ', a.next_to_unbuilt.mean(), b.next_to_unbuilt.mean(), fisher_exact([[a.next_to_unbuilt.sum(), (~a.next_to_unbuilt.astype(bool)).sum()], [b.next_to_unbuilt.sum(), (~b.next_to_unbuilt.astype(bool)).sum()]]))
print('touch indel among atyp', a.touch_indel.sum(), '/', len(a))
print('Q pct-in-domain median atyp/typ', a.q_win_pct_in_domain.median(), b.q_win_pct_in_domain.median(),
      'MWU p', mannwhitneyu(a.q_win_pct_in_domain.dropna(), b.q_win_pct_in_domain.dropna()).pvalue if a.q_win_pct_in_domain.notna().sum() and b.q_win_pct_in_domain.notna().sum() else None)
U.to_csv(f'{OUT}/tables/units_em_flags.csv.gz', index=False)
