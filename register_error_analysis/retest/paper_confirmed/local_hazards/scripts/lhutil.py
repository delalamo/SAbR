"""Shared loaders and statistics for the paper-confirmed local-hazard re-test."""
import numpy as np, pandas as pd, warnings
from scipy.stats import norm, fisher_exact
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = '/home/user/SAbR/register_error_analysis/retest/deeper'
O = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/local_hazards/tables'
MODEMAP = {'A_FR1Nterm': 'A', 'F_CDR1': 'F', 'B_CDR2FR3': 'B', 'C_FR3': 'C', 'Dplus_CDR3short': 'D+', 'Dminus_CDR3long': 'D-', 'E_FR4bulge': 'E'}
STRICT_LEDGER = {'amyloid_fibril', 'indel_gap_artifact_confirmed', 'DE_chain_end_artifact', 'near_copy', 'not_a_register_shift'}


def chains():
    C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
    F = pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz')
    C = C.merge(F, on=['pdb', 'chain'])
    C['light'] = C.ctype.isin(['K', 'L'])
    C['EM'] = C.method.eq('EM')
    return C


def case_runs():
    """All classified runs with a mode, labelled by case set.
    paper_set: strict / loose / copy / near_copy / artifact / None ; rest: deeper first occurrence, ledger-strict removed,
    not a paper case unit."""
    R = pd.read_csv(f'{SP}/deeper/runs_all.csv')
    L = pd.read_csv(f'{D}/synthesis/artifact_ledger_runs.csv')[['pdb', 'chain', 'imgt0', 'imgt1', 'flag']]
    R = R.merge(L, on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
    E = pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv')
    E = E[E.confirmed]
    st = {'confirmed first occurrence': 'strict', 'confirmed, loose-ledger doubt': 'loose', 'copy of an earlier model': 'copy',
          'near copy (ledger)': 'near_copy', 'artifact (synthesis ledger, strict)': 'artifact'}
    E['pset'] = E.status.map(st)
    k = E.drop_duplicates(['pdb', 'chain', 'imgt0', 'imgt1'])[['pdb', 'chain', 'imgt0', 'imgt1', 'pset']]
    R = R.merge(k, on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
    R['m'] = R['mode'].map(MODEMAP)
    return R


def mh(df, x, y='case', strata='stratum'):
    """Mantel-Haenszel OR over strata; returns dict(OR, lo, hi, p, n_case, n_ctrl, pct_case, pct_ctrl_std)."""
    tabs = []
    for s, g in df.groupby(strata):
        a = int(((g[y] == 1) & (g[x] == 1)).sum()); b = int(((g[y] == 1) & (g[x] == 0)).sum())
        c = int(((g[y] == 0) & (g[x] == 1)).sum()); d = int(((g[y] == 0) & (g[x] == 0)).sum())
        if a + b == 0 or c + d == 0: continue
        tabs.append(np.array([[a, b], [c, d]], float))
    out = dict(n_case=int(df[y].sum()), n_ctrl=int((df[y] == 0).sum()))
    if not tabs: return out
    nc = sum(t[0].sum() for t in tabs)
    out['pct_case'] = 100 * sum(t[0, 0] for t in tabs) / nc
    # control prevalence standardised to the case distribution over strata
    out['pct_ctrl_std'] = 100 * sum(t[0].sum() * t[1, 0] / t[1].sum() for t in tabs) / nc
    try:
        st = StratifiedTable([t + 0.0 for t in tabs])
        if len(tabs) == 1:
            t2 = Table2x2(tabs[0] + (0.5 if (tabs[0] == 0).any() else 0))
            out['OR'] = t2.oddsratio; out['lo'], out['hi'] = t2.oddsratio_confint(); out['p'] = fisher_exact(tabs[0])[1]
        else:
            out['OR'] = st.oddsratio_pooled; out['lo'], out['hi'] = st.oddsratio_pooled_confint(); out['p'] = st.test_null_odds(correction=True).pvalue
            out['se_log'] = (np.log(out['hi']) - np.log(out['lo'])) / (2 * 1.96)
        if 'se_log' not in out: out['se_log'] = (np.log(out['hi']) - np.log(out['lo'])) / (2 * 1.96)
    except Exception as e:
        out['err'] = str(e)
    return out


def gee(df, x, y='case', strata='stratum', groups='paper'):
    """Logistic GEE (independence), feature + stratum fixed effects, robust SE clustered by paper."""
    d = df[[x, y, strata, groups]].dropna().copy(); d[x] = d[x].astype(float); d[y] = d[y].astype(float)
    # drop strata without cases
    keep = d.groupby(strata)[y].transform('sum') > 0; d = d[keep]
    try:
        f = f'{y} ~ {x}' + (f' + C({strata})' if d[strata].nunique() > 1 else '')
        m = smf.gee(f, groups=pd.factorize(d[groups])[0], data=d, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
        b, se = m.params[x], m.bse[x]
        return dict(OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=m.pvalues[x], n_clusters=d[groups].nunique())
    except Exception as e:
        return dict(err=str(e))


def min_detectable_or(n1, p0, n0=None, alpha=0.05, power=0.8):
    """Smallest OR>1 detectable with given power (two-sided) for n1 cases vs n0 controls with control prevalence p0."""
    if n1 is None or n1 < 2 or p0 is None or not (0 < p0 < 1): return np.nan
    n0 = n0 or 10 ** 6
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    for OR in np.exp(np.linspace(0.001, np.log(200), 2000)):
        p1 = OR * p0 / (1 - p0 + OR * p0)
        pbar = (n1 * p1 + n0 * p0) / (n1 + n0)
        se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n0)); se1 = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        if (abs(p1 - p0) - za * se0) / se1 >= zb: return OR
    return np.inf


def z_het(a, b):
    """z-test of two log ORs from dicts with OR and se_log."""
    try:
        z = (np.log(a['OR']) - np.log(b['OR'])) / np.sqrt(a['se_log'] ** 2 + b['se_log'] ** 2)
        return 2 * norm.sf(abs(z))
    except Exception:
        return np.nan


def fmt(r, pct=True):
    if r is None or 'OR' not in r or r.get('OR') is None or not np.isfinite(r.get('OR', np.nan)): return 'n/a'
    return f"OR {r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.2g}"
