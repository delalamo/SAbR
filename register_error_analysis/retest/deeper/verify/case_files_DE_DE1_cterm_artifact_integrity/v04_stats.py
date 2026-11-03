"""V04: statistics for the DE1 verification.
(a) reverse-engineer and re-apply the investigator's confirmation rule to their slot shifts (rule not in their scripts);
(b) the claimed 2x2 (D+ model ends <=121 vs later, not confirmed) + alternative cut-offs, indeterminate geometry,
    leave-one-paper-out; (c) my independent geometry verdicts (v02); (d) truncation experiment rates (v01) with exact
    binomial CIs and Fisher tests; BH q over all tests here.
Output: tables/v04_tests.csv
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, binomtest
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE1_cterm_artifact_integrity'
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables/independent_check_DE_units.csv')
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'paper'])
T = T.merge(C, on=['pdb', 'chain'], how='left')


def rule(r):
    lo, hi = max(116, int(r.imgt0)), min(124, int(r.imgt1))
    v = [r[f'slotshift_{n}'] for n in range(lo, hi + 1) if pd.notna(r[f'slotshift_{n}'])]
    if not v: return 'not testable (run outside 116-124 or unbuilt)'
    k = sum(np.sign(x) == np.sign(r.delta) for x in v)
    return 'confirmed' if k >= max(1, len(v) / 2) else 'not confirmed'
T['rule_re'] = T.apply(rule, axis=1)
print('reverse-engineered rule reproduces investigator verdicts:', int((T.rule_re == T.confirm).sum()), '/', len(T))
print(T[T.rule_re != T.confirm][['pdb', 'chain', 'mode', 'confirm', 'rule_re']])
tests = []


def fisher(name, a, b, c, d, note=''):
    o, p = fisher_exact([[a, b], [c, d]])
    tests.append(dict(test=name, err=f'{a}/{a+b}', ctl=f'{c}/{c+d}', OR=o, p=p, note=note))

dp = T[T['mode'] == 'Dplus_CDR3short'].copy()
nc = dp.confirm == 'not confirmed'
for cut in [118, 119, 121]:
    s = dp.last_built_imgt <= cut
    fisher(f'investigator verdicts: D+ not confirmed, model ends <=IMGT{cut} vs later', int((s & nc).sum()), int((s & ~nc).sum()),
           int((~s & nc).sum()), int((~s & ~nc).sum()))
# leave-one-paper-out on the claimed <=121 table
s = dp.last_built_imgt <= 121
lopo = []
for pp in dp.paper.unique():
    z = dp[dp.paper != pp]; zs = z.last_built_imgt <= 121; zn = z.confirm == 'not confirmed'
    lopo.append(fisher_exact([[int((zs & zn).sum()), int((zs & ~zn).sum())], [int((~zs & zn).sum()), int((~zs & ~zn).sum())]])[1])
print('leave-one-paper-out max p for <=121 table:', max(lopo), 'papers', dp.paper.nunique())
# one unit per paper (first unit) -- independence
dpp = dp.sort_values(['paper', 'pdb', 'chain']).drop_duplicates('paper')
s1 = dpp.last_built_imgt <= 121; n1 = dpp.confirm == 'not confirmed'
fisher('investigator verdicts, one D+ unit per paper, <=121', int((s1 & n1).sum()), int((s1 & ~n1).sum()), int((~s1 & n1).sum()), int((~s1 & ~n1).sum()))
# my geometry verdicts
G = pd.read_csv(f'{OUT}/tables/v02_geometry_units.csv')
for core in ['global', 'local']:
    g = G[G['mode'] == 'Dplus_CDR3short']
    s = g.last_built_imgt <= 121; n = g[f'v_{core}'] == 'not confirmed'
    fisher(f'v02 {core} verdicts: D+ not confirmed, ends <=121 vs later', int((s & n).sum()), int((s & ~n).sum()), int((~s & n).sum()), int((~s & ~n).sum()),
           'indeterminate counted as not-not-confirmed')
    det = g[g[f'v_{core}'].isin(['confirmed', 'not confirmed'])]
    s = det.last_built_imgt <= 121; n = det[f'v_{core}'] == 'not confirmed'
    fisher(f'v02 {core} verdicts, determinate only: ends <=121 vs later', int((s & n).sum()), int((s & ~n).sum()), int((~s & n).sum()), int((~s & ~n).sum()))
    s = det.last_built_imgt <= 119
    fisher(f'v02 {core} verdicts, determinate only: ends <=119 vs later', int((s & n).sum()), int((s & ~n).sum()), int((~s & n).sum()), int((~s & ~n).sum()))
agree = pd.crosstab(G.confirm, G.v_global)
print(agree)
# truncation experiment
V = pd.read_csv(f'{OUT}/tables/v01_sabr_runs.csv')
t = V[V.label == 'T clean']
full = t[t.exp == 'trunc_full']
for e in ['trunc_117', 'trunc_118', 'trunc_118nometa', 'trunc_119', 'trunc_120', 'trunc_121', 'trunc_122', 'trunc_123', 'trunc_125']:
    x = t[t.exp == e]
    a = int(x.cterm_run_pos.sum()); n = len(x)
    ci = binomtest(a, n).proportion_ci()
    tests.append(dict(test=f'SAbR on clean chains truncated {e[6:]}: D+-like run at the C-terminus', err=f'{a}/{n}', ctl=f'{int(full.cterm_run_pos.sum())}/{len(full)} untruncated',
                      OR=np.nan, p=fisher_exact([[a, n - a], [int(full.cterm_run_pos.sum()), len(full) - int(full.cterm_run_pos.sum())]])[1],
                      note=f'95% CI {ci.low:.2f}-{ci.high:.2f}'))
P = V[V.label == 'P']
for e in ['pred_full', 'pred_same_end', 'pred_same_built']:
    x = P[P.exp == e]
    tests.append(dict(test=f'ImmuneBuilder model of the short-ended D+ antibodies, {e}: D+-like C-terminal run', err=f'{int(x.cterm_run_pos.sum())}/{len(x)}',
                      ctl='', OR=np.nan, p=np.nan, note=';'.join(f'{a}{b}:{r}' for a, b, r in zip(x.pdb, x.chain, x.runs.fillna('none')))))
L = V[(V.label == 'L long D+')]
for e in ['trunc_full', 'trunc_118']:
    x = L[L.exp == e]
    tests.append(dict(test=f'long-ended D+ models, {e}: SAbR C-terminal + run', err=f'{int(x.cterm_run_pos.sum())}/{len(x)}', ctl='', OR=np.nan, p=np.nan))
O = pd.DataFrame(tests)
ok = O.p.notna(); O.loc[ok, 'q_BH'] = multipletests(O.loc[ok, 'p'], method='fdr_bh')[1]
O.to_csv(f'{OUT}/tables/v04_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(O.to_string())
