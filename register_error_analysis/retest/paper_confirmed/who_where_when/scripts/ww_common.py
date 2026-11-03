"""Shared loaders and statistics for the paper-confirmed re-test of the 'who, where, when' trends
(people, places, publication, antigen and competition; deeper README 4.3, 4.4, 6).

Unit flags (entry / paper / study level) are derived from the per-chain case flags in
scratchpad/paperconf/chain_flags.csv.gz:
  case   = unit contains a paper_case_strict chain (Table S1 error confirmed by SAbR/ANARCI, first occurrence)
  case_i = unit contains a paper_case_incl_copies chain (sensitivity: + copies, near copies, loose-ledger case)
  ctrl   = every V chain of the unit is a paper_control chain (released <= 2022-04-05, entry not in Table S1,
           no SAbR/ANARCI run of any origin)  -> window controls
  gen    = unit has a ledger-clean first-occurrence run (deeper first-occurrence errors minus synthesis strict ledger)
  rest   = gen & not case_i  (deeper first-occurrence errors that are not paper-confirmed), all years
  rest_w = rest & released in the paper window (in-window errors conkit did not flag or that S1 did not confirm)
  clean  = no run of any origin (all years) -> full-set controls
"""
import numpy as np, pandas as pd, warnings
from scipy.stats import norm, fisher_exact, mannwhitneyu
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/who_where_when'
T = f'{OUT}/tables'
CACHE = f'{SP}/paperconf/who_where_when'   # private (non-pseudonymised) intermediate files
WINDOW = '2022-04-05'
STRICT_LEDGER = {'amyloid_fibril', 'indel_gap_artifact_confirmed', 'DE_chain_end_artifact', 'near_copy', 'not_a_register_shift'}
GROUPS = ['case', 'case_i', 'rest', 'rest_w']


def chains():
    C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
    F = pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz')
    C = C.merge(F, on=['pdb', 'chain'])
    R = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
    L = pd.read_csv(f'{D}/synthesis/artifact_ledger_runs.csv')[['pdb', 'chain', 'imgt0', 'imgt1', 'flag']]
    R = R.merge(L, on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
    g = R[~R.flag.isin(STRICT_LEDGER)]
    C['gen_first'] = [k in set(zip(g.pdb, g.chain)) for k in zip(C.pdb, C.chain)]
    return C


def entry_flags(C=None):
    C = chains() if C is None else C
    E = C.groupby('pdb').agg(method=('method', 'first'), release=('release', 'first'), deposit=('deposit', 'first'), paper=('paper', 'first'),
                             case=('paper_case_strict', 'max'), case_f=('paper_case_first', 'max'), case_i=('paper_case_incl_copies', 'max'),
                             ctrl=('paper_control', 'min'), win=('in_paper_window', 'min'), s1=('paper_flagged_entry', 'max'),
                             gen=('gen_first', 'max'), anyrun=('err_any', 'max'), nV=('chain', 'size')).reset_index()
    for k in ['case', 'case_f', 'case_i', 'ctrl', 'win', 's1', 'gen', 'anyrun']:
        E[k] = E[k].astype(bool)
    E['clean'] = ~E.anyrun
    E['rest'] = E.gen & ~E.case_i
    E['rest_w'] = E.rest & E.win
    E['win_all'] = E.case | E.rest_w   # all in-window first-occurrence errors (paper-confirmed + not), i.e. without conkit's selection
    return E


def unit_flags(E, key_map, key='unit'):
    """Aggregate entry flags to a higher unit. key_map: DataFrame with columns pdb, <key>. Every entry of the unit
    that is in E is used; ctrl requires all entries ctrl, clean requires all clean."""
    m = key_map[['pdb', key]].drop_duplicates().merge(E[['pdb', 'case', 'case_f', 'case_i', 'ctrl', 'clean', 'gen', 'win', 's1']], on='pdb')
    U = m.groupby(key).agg(case=('case', 'max'), case_f=('case_f', 'max'), case_i=('case_i', 'max'), ctrl=('ctrl', 'min'), clean=('clean', 'min'),
                           gen=('gen', 'max'), win=('win', 'min'), s1=('s1', 'max'), n_case_entries=('case', 'sum')).reset_index()
    U['rest'] = U.gen & ~U.case_i
    U['rest_w'] = U.rest & U.win
    U['win_all'] = U.case | U.rest_w
    return U


def ycat3(year):
    y = pd.Series(year).astype(float)
    return np.where(y <= 2019, '<=2019', np.where(y == 2020, '2020', '2021+'))


# ---------------------------------------------------------------- statistics
def logit(d, y, x, rhs='', cl=None, label=''):
    """Logistic regression, cluster-robust SE by `cl` (if given). Returns dict with OR per unit of x."""
    cols = [y, x] + [c for c in d.columns if c in rhs.replace('C(', ' ').replace(')', ' ').replace('+', ' ').split()]
    if cl: cols.append(cl)
    d = d.dropna(subset=list(dict.fromkeys(cols))).copy()
    d[y] = d[y].astype(float)
    if d[x].dtype == bool: d[x] = d[x].astype(float)
    out = dict(n_case=int(d[y].sum()), n_ctrl=int((d[y] == 0).sum()))
    if out['n_case'] < 2 or d[x].nunique() < 2: return out
    if rhs:   # drop covariates that are constant in this subset
        keep = []
        for t in [t.strip() for t in rhs.split('+')]:
            col = t[2:-1] if t.startswith('C(') else t
            if col in d.columns and d[col].nunique() < 2: continue
            keep.append(t)
        rhs = ' + '.join(keep)
    f = f'{y} ~ {x}' + (f' + {rhs}' if rhs else '')
    xb = d[x].dropna().isin([0, 1]).all()
    if xb:
        cells = pd.crosstab(d[y], d[x].astype(int)).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
        if (cells.values == 0).any():   # quasi-separation: fall back to exact / Mantel-Haenszel with 0.5 correction
            st = [s for s in ['yc3', 'yc5'] if s in rhs] if rhs else []
            return mh(d, y, x, st[0]) if st else mh(d.assign(_s=1), y, x, '_s')
    try:
        mod = smf.glm(f, d, family=sm.families.Binomial())
        r = mod.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cl])[0]}) if cl else mod.fit()
        b, se = r.params[x], r.bse[x]
        if not np.isfinite(se) or se > 20: raise ValueError('separation')
        out.update(OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=r.pvalues[x], se_log=se)
    except Exception as e:
        out['err'] = str(e)[:60]
    return out


def mh(d, y, x, strata):
    """Mantel-Haenszel OR for binary x over strata. If any informative stratum has a zero cell, the OR and CI are the
    exact conditional (Fisher) estimate on the pooled table and p is the stratified CMH test (exact Fisher if 1 stratum)."""
    from scipy.stats.contingency import odds_ratio
    tabs = []
    for s, g in d.groupby(strata):
        a = int(((g[y] == 1) & (g[x] == 1)).sum()); b = int(((g[y] == 1) & (g[x] == 0)).sum())
        c = int(((g[y] == 0) & (g[x] == 1)).sum()); dd = int(((g[y] == 0) & (g[x] == 0)).sum())
        if a + b == 0 or c + dd == 0 or a + c == 0 or b + dd == 0: continue
        tabs.append(np.array([[a, b], [c, dd]], float))
    out = dict(n_case=int(d[y].sum()), n_ctrl=int((d[y] == 0).sum()))
    out['pct_case'] = 100 * d.loc[d[y] == 1, x].astype(float).mean(); out['pct_ctrl'] = 100 * d.loc[d[y] == 0, x].astype(float).mean()
    if not tabs: return out
    try:
        pooled = sum(tabs)
        if len(tabs) == 1 or any((t == 0).any() for t in tabs):
            r = odds_ratio(pooled.astype(int), kind='conditional'); ci = r.confidence_interval()
            out['OR'], out['lo'], out['hi'] = r.statistic, ci.low, ci.high
            out['p'] = fisher_exact(pooled)[1] if len(tabs) == 1 else StratifiedTable(tabs).test_null_odds(correction=True).pvalue
            out['method_note'] = 'exact conditional OR (pooled), CMH/Fisher p'
        else:
            st = StratifiedTable(tabs)
            out['OR'] = st.oddsratio_pooled; out['lo'], out['hi'] = st.oddsratio_pooled_confint(); out['p'] = st.test_null_odds(correction=True).pvalue
        if out['lo'] > 0 and np.isfinite(out['hi']):
            out['se_log'] = (np.log(out['hi']) - np.log(out['lo'])) / (2 * 1.96)
    except Exception as e:
        out['err'] = str(e)[:60]
    return out


def min_det_or(n1, p0, n0=None, alpha=0.05, power=0.8, direction=1):
    """Smallest OR (>1 if direction=1, <1 if -1) detectable with 80% power, n1 cases vs n0 controls, control prevalence p0."""
    if n1 is None or n1 < 2 or p0 is None or not (0 < p0 < 1): return np.nan
    n0 = n0 or 10 ** 6
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    for lor in np.linspace(0.001, np.log(200), 3000):
        OR = np.exp(direction * lor)
        p1 = OR * p0 / (1 - p0 + OR * p0)
        pbar = (n1 * p1 + n0 * p0) / (n1 + n0)
        se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n0)); se1 = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        if (abs(p1 - p0) - za * se0) / se1 >= zb: return OR
    return np.inf if direction == 1 else 0.0


def z_het(a, b):
    try:
        z = (np.log(a['OR']) - np.log(b['OR'])) / np.sqrt(a['se_log'] ** 2 + b['se_log'] ** 2)
        return 2 * norm.sf(abs(z))
    except Exception:
        return np.nan


def fmt(r):
    if not isinstance(r, dict) or 'OR' not in r or not np.isfinite(r.get('OR', np.nan)):
        return f"n/a (cases {r.get('n_case', '?')})" if isinstance(r, dict) else 'n/a'
    return f"OR {r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.2g}"


def verdict(full_dir, r, null_full=False, alpha=0.05):
    """full_dir: +1 if full-set OR>1, -1 if <1, 0 if null. r: paper-strict result."""
    if not isinstance(r, dict) or 'OR' not in r or not np.isfinite(r.get('OR', np.nan)): return 'cannot test'
    lor = np.log(r['OR']); sig = r['p'] < alpha
    if null_full or full_dir == 0:
        return 'null still null' if not sig else ('absent or reversed' if False else 'NEW signal (full set null)')
    if np.sign(lor) == full_dir:
        return 'holds' if sig else 'same direction, underpowered'
    return 'absent or reversed' if sig or abs(lor) > 0.1 else 'attenuated'


def run_groups(frame, x, rhs, cl, ctrl_win='ctrl', ctrl_all='clean', extra=None, kind='logit'):
    """Fit x for: case vs window controls, case_i vs window controls, rest (all yrs) vs clean (all yrs),
    rest_w vs window controls, gen (all) vs clean. Returns dict of results keyed by group."""
    res = {}
    for g, ctl in [('case', ctrl_win), ('case_i', ctrl_win), ('rest_w', ctrl_win), ('win_all', ctrl_win), ('rest', ctrl_all), ('gen', ctrl_all)]:
        d = frame[(frame[g]) | (frame[ctl] & ~frame['gen'] & ~frame['case_i'])].copy()
        d['_y'] = d[g].astype(int)
        r_ = (rhs or '').replace('{YC}', 'C(yc3)' if g in ('case', 'case_i', 'rest_w', 'win_all') else 'C(yc5)')
        if kind == 'logit':
            res[g] = logit(d, '_y', x, r_, cl)
        else:
            res[g] = mh(d, '_y', x, 'yc3' if g in ('case', 'case_i', 'rest_w', 'win_all') else 'yc5')
        if kind == 'logit' and 'OR' in res[g]:
            dd = d.dropna(subset=[x])
            if dd[x].dropna().isin([0, 1, True, False]).all():
                res[g]['pct_case'] = 100 * dd.loc[dd._y == 1, x].astype(float).mean(); res[g]['pct_ctrl'] = 100 * dd.loc[dd._y == 0, x].astype(float).mean()
            else:
                res[g]['med_case'] = dd.loc[dd._y == 1, x].median(); res[g]['med_ctrl'] = dd.loc[dd._y == 0, x].median()
    return res


def add_yc(df, ycol):
    y = df[ycol].astype(float)
    df['yc3'] = np.where(y <= 2019, '<=2019', np.where(y == 2020, '2020', '2021+'))
    df['yc5'] = np.where(y <= 2019, '<=2019', np.where(y == 2020, '2020', np.where(y == 2021, '2021', np.where(y <= 2023, '2022-23', '2024+'))))
    return df


ROWS = []


def record(trend, level, method, x, spec, res, full_ref='', note=''):
    """Append one row per group to ROWS and print a compact line."""
    line = [f'{trend[:48]:48s} {level:6s} {method:5s} {x[:22]:22s}']
    for g, r in res.items():
        row = dict(trend=trend, level=level, method=method, x=x, spec=spec, group=g, full_ref=full_ref, note=note)
        row.update({k: v for k, v in r.items()})
        ROWS.append(row)
    for g in ['case', 'case_i', 'rest_w', 'rest']:
        r = res.get(g, {})
        desc = ''
        if 'pct_case' in r: desc = f"{r['pct_case']:.0f}v{r.get('pct_ctrl', r.get('pct_ctrl_std', np.nan)):.0f}% "
        elif 'med_case' in r: desc = f"{r['med_case']:.2g}v{r['med_ctrl']:.2g} "
        line.append(f"{g}[{r.get('n_case', 0)}/{r.get('n_ctrl', 0)}] {desc}{fmt(r)}")
    if 'case' in res and 'rest' in res: line.append(f"zhet(case v rest)={z_het(res['case'], res['rest']):.2g}")
    print(' | '.join(line))


def save_rows(path):
    df = pd.DataFrame(ROWS)
    df.to_csv(path, index=False)
    return df
