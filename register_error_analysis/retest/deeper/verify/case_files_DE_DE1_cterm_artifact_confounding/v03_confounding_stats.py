"""V03: does the association 'D+ model ends at <= IMGT 121' -> 'run not confirmed by framework superposition' survive
confounding control, and is non-confirmation better explained by a simpler factor?
Inputs: the investigator's independent_check_DE_units.csv (80 D/E units; slot shifts at ANARCI 116-124), chains.csv,
local_all.jsonl (per-residue Q-score / RSCC), SAbR records.
Analyses
 A. reproduce the Fisher test; strata: EM only, non-SARS-CoV-2 only, SARS-CoV-2 only, heavy/VHH vs light, year bin,
    resolution half; exact stratified (conditional) permutation tests; one unit per paper; one unit per last author;
    Firth-penalised logit with year, resolution, SARS-CoV-2, method.
 B. 'equal information' check: recompute the investigator's confirmation rule (k >= max(1, n/2) of run residues in the
    window sitting >=1 slot away in the direction of the SAbR-ANARCI shift) using ONLY IMGT 116-118 for every unit,
    i.e. the window available to a model that stops at W/F118.
 C. simpler factors for non-confirmation across all testable D/E units: run touches the chain end, unbuilt FR4 residues
    after 118, local map fit at 116-118 (Q-score, EM only), resolution, year, method.
Outputs: tables/v03_*.csv
"""
import json, numpy as np, pandas as pd, warnings
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE1_cterm_artifact_confounding'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables'
rng = np.random.default_rng(2026)

I = pd.read_csv(f'{INV}/independent_check_DE_units.csv')
CH = pd.read_csv(f'{SP}/deeper/chains.csv.gz')
I = I.merge(CH[['pdb', 'chain', 'res', 'sars2', 'paper', 'last_author', 'vhh', 'deposit']], on=['pdb', 'chain'], how='left')
I['nc'] = (I.confirm == 'not confirmed').astype(int)
I['testable'] = I.confirm.isin(['confirmed', 'not confirmed'])
I['le121'] = I.model_ends_le121.astype(bool).astype(int)
I['yb'] = pd.cut(I.year, [0, 2020, 2022, 2030], labels=['<=2020', '2021-22', '>=2023']).astype(str)
I['ct2'] = np.where((I.ctype == 'H'), 'H/VHH', 'light')
I['em'] = (I.method == 'EM').astype(int)
I['sars2'] = I.sars2.astype(bool).astype(int)


def fisher(d, x='le121', y='nc'):
    a = int(((d[x] == 1) & (d[y] == 1)).sum()); b = int(((d[x] == 1) & (d[y] == 0)).sum())
    c = int(((d[x] == 0) & (d[y] == 1)).sum()); e = int(((d[x] == 0) & (d[y] == 0)).sum())
    orr, p = fisher_exact([[a, b], [c, e]])
    orh = (a + .5) * (e + .5) / ((b + .5) * (c + .5))
    return dict(nc_in_exposed=f'{a}/{a + b}', nc_in_unexposed=f'{c}/{c + e}', OR_fisher=orr, OR_haldane=orh, p=p, n=len(d))


def strat_perm(d, strata, x='le121', y='nc', B=20000):
    """exact conditional test: permute exposure within strata, statistic = # non-confirmed among exposed"""
    obs = int(((d[x] == 1) & (d[y] == 1)).sum())
    groups = [g for _, g in d.groupby(strata)]
    cnt = 0
    for _ in range(B):
        s = 0
        for g in groups:
            xx = rng.permutation(g[x].values); s += int(((xx == 1) & (g[y].values == 1)).sum())
        cnt += s >= obs
    # informative strata = those with both exposure levels and both outcomes
    inf = sum(1 for g in groups if g[x].nunique() == 2 and g[y].nunique() == 2)
    return obs, (cnt + 1) / (B + 1), inf


def firth_logit(X, y, iters=100):
    """Firth-penalised logistic regression (Heinze & Schemper); returns beta, se, profile-free Wald p."""
    from scipy.stats import norm
    X = np.column_stack([np.ones(len(y)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        XtW = X.T * W; Iinf = XtW @ X; Iinv = np.linalg.inv(Iinf)
        h = np.einsum('ij,jk,ik->i', X * W[:, None] ** .5, Iinv, X * W[:, None] ** .5)
        U = X.T @ (y - p + h * (0.5 - p))
        step = Iinv @ U; b = b + step
        if np.abs(step).max() < 1e-8: break
    se = np.sqrt(np.diag(Iinv)); z = b / se
    return b, se, 2 * norm.sf(np.abs(z))


rows = []
dp = I[(I['mode'] == 'Dplus_CDR3short') & I.testable].copy()
rows.append(dict(analysis='A1 reproduce: all D+ units', **fisher(dp)))
rows.append(dict(analysis='A2 EM only', **fisher(dp[dp.em == 1])))
rows.append(dict(analysis='A3 non-SARS-CoV-2 only', **fisher(dp[dp.sars2 == 0])))
rows.append(dict(analysis='A4 SARS-CoV-2 only', **fisher(dp[dp.sars2 == 1])))
rows.append(dict(analysis='A5 heavy/VHH only', **fisher(dp[dp.ct2 == 'H/VHH'])))
rows.append(dict(analysis='A6 light only', **fisher(dp[dp.ct2 == 'light'])))
rows.append(dict(analysis='A7 EM, deposited >= 2020', **fisher(dp[(dp.em == 1) & (dp.year >= 2020)])))
med = dp[dp.em == 1].res.median()
rows.append(dict(analysis=f'A8 EM, res <= {med:.2f}', **fisher(dp[(dp.em == 1) & (dp.res <= med)])))
rows.append(dict(analysis=f'A9 EM, res > {med:.2f}', **fisher(dp[(dp.em == 1) & (dp.res > med)])))
# one unit per paper / per last author (worst case: keep the unit that is least favourable to the claim)
for lab, col in [('A10 one unit per paper', 'paper'), ('A11 one unit per last author', 'last_author')]:
    g = dp.groupby(col).agg(le121=('le121', 'max'), nc=('nc', 'max'), em=('em', 'max')).reset_index()
    rows.append(dict(analysis=lab, **fisher(g)))
    rows.append(dict(analysis=lab + ', EM', **fisher(g[g.em == 1])))
# stratified exact tests
for lab, st in [('A12 exact, stratified by year bin', ['yb']), ('A13 exact, stratified by method x SARS-CoV-2', ['em', 'sars2']),
                ('A14 exact, stratified by year bin x method', ['yb', 'em']), ('A15 exact, stratified by chain type x method', ['ct2', 'em'])]:
    obs, p, inf = strat_perm(dp, st)
    rows.append(dict(analysis=lab, nc_in_exposed=f'{obs}/{int(dp.le121.sum())}', p=p, n=len(dp), note=f'{inf} informative strata'))
# resolution-adjusted (continuous) via Firth
for lab, cov in [('A16 Firth: le121 + year', ['year']), ('A17 Firth: le121 + res', ['res']), ('A18 Firth: le121 + sars2', ['sars2']),
                 ('A19 Firth: le121 + em', ['em']), ('A20 Firth: le121 + year + res + sars2', ['year', 'res', 'sars2'])]:
    d = dp.dropna(subset=cov)
    Xm = np.array(d[['le121'] + cov].astype(float).values, copy=True)
    Xm[:, 1:] = (Xm[:, 1:] - Xm[:, 1:].mean(0)) / np.where(Xm[:, 1:].std(0) > 0, Xm[:, 1:].std(0), 1)
    b, se, p = firth_logit(Xm, d.nc.values.astype(float))
    rows.append(dict(analysis=lab, OR_firth=float(np.exp(b[1])), lo=float(np.exp(b[1] - 1.96 * se[1])), hi=float(np.exp(b[1] + 1.96 * se[1])),
                     p=float(p[1]), n=len(d), note='covariates standardised; ' + ', '.join(f'{c}: OR {np.exp(b[i + 2]):.2f} p {p[i + 2]:.2g}' for i, c in enumerate(cov))))

# B. equal-information window (116-118 only) -------------------------------------------------------------
def confirm_window(r, lo=116, hi=124):
    pos = [n for n in range(max(lo, int(r.imgt0)), min(hi, int(r.imgt1)) + 1) if pd.notna(r.get(f'slotshift_{n}'))]
    if not pos: return 'not testable'
    k = sum(np.sign(r[f'slotshift_{n}']) == np.sign(r.delta) for n in pos)
    return 'confirmed' if k >= max(1, len(pos) / 2) else 'not confirmed'


I['confirm_rebuilt'] = I.apply(confirm_window, axis=1)
I['confirm_116_118'] = I.apply(lambda r: confirm_window(r, 116, 118), axis=1)
rep = (I.confirm_rebuilt == I.confirm).mean()
dp = I[(I['mode'] == 'Dplus_CDR3short')].copy()
w = dp[dp.confirm_116_118 != 'not testable'].copy(); w['nc'] = (w.confirm_116_118 == 'not confirmed').astype(int)
rows.append(dict(analysis='B1 window 116-118 only, all D+', **fisher(w), note=f'rule reproduces investigator labels for {rep:.0%} of 80 units'))
rows.append(dict(analysis='B2 window 116-118 only, EM D+', **fisher(w[w.em == 1])))
g = w.groupby('paper').agg(le121=('le121', 'max'), nc=('nc', 'max'), em=('em', 'max')).reset_index()
rows.append(dict(analysis='B3 window 116-118 only, one unit per paper', **fisher(g)))

# C. simpler factors across all testable D/E units ------------------------------------------------------
# local Q-score / RSCC at the run's 116-118 residues vs the chain's V-domain median
want = set(zip(I.pdb, I.chain))
rec = {}
for f in ['validation', 'snapshot_rest', 'new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r = json.loads(l)
        if r.get('status') == 'ok' and (r['pdb'], r['chain']) in want: rec[(r['pdb'], r['chain'])] = r['residues']
loc = {}
for l in open(f'{SP}/retest/local_all.jsonl'):
    r = json.loads(l)
    if (r['pdb'], r['chain']) in want:
        feats = r['feats'] if isinstance(r['feats'], dict) else eval(r['feats'])
        key = 'Q_SCORE' if 'Q_SCORE' in feats else ('RSCC' if 'RSCC' in feats else None)
        if key is None: continue
        amap = r['auth_map']
        loc[(r['pdb'], r['chain'])] = (key, {amap[int(k) - 1]: v for k, v in feats[key].items() if int(k) - 1 < len(amap)})
qq = []
for r in I.itertuples():
    res = rec.get((r.pdb, r.chain)); L = loc.get((r.pdb, r.chain))
    if res is None or L is None: qq.append((np.nan, np.nan, np.nan, None)); continue
    key, m = L
    v = [x for x in res if x[4] is not None]
    allq = [m.get(f'{x[0]}{x[1]}', m.get(str(x[0]))) for x in v]
    allq = [q for q in allq if q is not None]
    site = [m.get(f'{x[0]}{x[1]}', m.get(str(x[0]))) for x in v if 116 <= x[4] <= 118 and not x[5]]
    site = [q for q in site if q is not None]
    lastq = [m.get(f'{x[0]}{x[1]}', m.get(str(x[0]))) for x in v[-3:]]
    lastq = [q for q in lastq if q is not None]
    qq.append((np.mean(site) if site else np.nan, np.median(allq) if allq else np.nan, np.mean(lastq) if lastq else np.nan, key))
I['q_site'], I['q_vmed'], I['q_last3'], I['q_key'] = zip(*qq)
I['q_site_rel'] = I.q_site - I.q_vmed
I['run_at_chain_end'] = (I.imgt1 >= I.last_built_imgt - 1).astype(int)
I['unbuilt_after118'] = (I.fr4_unbuilt_after118_10.fillna(0) >= 1).astype(int)
I['unbuilt5_after118'] = (I.fr4_unbuilt_after118_10.fillna(0) >= 5).astype(int)
I.to_csv(f'{OUT}/tables/v03_units_augmented.csv', index=False)
T = I[I.testable].copy()
for lab, d in [('all testable D/E', T), ('EM testable D/E', T[T.em == 1]), ('EM D- and E only', T[(T.em == 1) & (T['mode'] != 'Dplus_CDR3short')]),
               ('EM D+ only', T[(T.em == 1) & (T['mode'] == 'Dplus_CDR3short')])]:
    for x in ['le121', 'run_at_chain_end', 'unbuilt_after118', 'unbuilt5_after118']:
        rows.append(dict(analysis=f'C {lab}: {x}', **fisher(d, x=x)))
    for x in ['q_site', 'q_site_rel', 'res', 'year']:
        dd = d[d.q_key.eq('Q_SCORE')] if x.startswith('q_') else d
        dd = dd.dropna(subset=[x])
        a, b = dd[dd.nc == 1][x], dd[dd.nc == 0][x]
        if len(a) >= 3 and len(b) >= 3:
            rows.append(dict(analysis=f'C {lab}: {x} (not confirmed vs confirmed; Q-score EM only for q_)', nc_in_exposed=f'median {a.median():.3f} (n={len(a)})',
                             nc_in_unexposed=f'median {b.median():.3f} (n={len(b)})', p=mannwhitneyu(a, b).pvalue, n=len(dd)))
# within units whose model does NOT end <=121: does chain end / unbuilt FR4 still separate?
d = T[(T.em == 1) & (T.le121 == 0)]
for x in ['run_at_chain_end', 'unbuilt_after118', 'unbuilt5_after118']:
    rows.append(dict(analysis=f'C EM D/E with model ending >121: {x}', **fisher(d, x=x)))
# D+ chain-end with full-length FR4 left: is 'le121' explained by 'run at chain end'?
d = T[(T['mode'] == 'Dplus_CDR3short')]
rows.append(dict(analysis='C D+: crosstab le121 x run_at_chain_end', note=str(pd.crosstab(d.le121, d.run_at_chain_end).to_dict())))

R = pd.DataFrame(rows)
ok = R.p.notna()
R.loc[ok, 'q_BH'] = multipletests(R.loc[ok, 'p'], method='fdr_bh')[1]
R.to_csv(f'{OUT}/tables/v03_confounding_tests.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 120)
print(R.round(5).to_string())
print('\nD+ units with model ending <= 121:')
print(I[(I['mode'] == 'Dplus_CDR3short') & (I.le121 == 1)][['pdb', 'chain', 'method', 'res', 'year', 'sars2', 'paper', 'delta', 'imgt0', 'imgt1', 'last_built_imgt',
                                                           'confirm', 'confirm_116_118', 'q_site', 'q_vmed', 'q_site_rel']].to_string())
print('\nD+ long models, window 116-118:')
print(I[(I['mode'] == 'Dplus_CDR3short') & (I.le121 == 0)][['pdb', 'chain', 'method', 'confirm', 'confirm_116_118', 'q_site_rel']].to_string())
