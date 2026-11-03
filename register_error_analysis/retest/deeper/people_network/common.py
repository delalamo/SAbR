"""Shared helpers for the people_network lens."""
import numpy as np, pandas as pd
from scipy.stats import hypergeom

OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/people_network'
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
CACHE = f'{SP}/cache/people_network'


def load_studies(method='EM'):
    S = pd.read_csv(f'{CACHE}/studies.csv.gz', parse_dates=['dep0'])
    S = S[(S.method == method) & S.grp.isin(['err', 'clean'])].reset_index(drop=True)
    S['y'] = (S.grp == 'err').astype(int)
    S['akeys'] = S.akeys.fillna('')
    S['akeys_nonlast'] = S.akeys_nonlast.fillna('')
    return S


def load_entries(method='EM'):
    E = pd.read_csv(f'{CACHE}/entries_people.csv.gz', parse_dates=['dep', 'study_dep0'])
    E = E[(E.method == method) & E.grp.isin(['err', 'clean'])].reset_index(drop=True)
    E['y'] = (E.grp == 'err').astype(int)
    E['akeys'] = E.akeys.fillna('')
    E['akeys_nonlast'] = E.akeys_nonlast.fillna('')
    return E


def strat_exact(member, y, strata):
    """Exact stratified (conditional) test of association between a binary membership and a binary outcome.
    Under H0 the error count among members in stratum s is hypergeometric(N_s, m_s, n_s); strata are convolved.
    Returns observed, expected, p_upper (enrichment), p_two (sum of probs <= P(obs)), MH odds ratio."""
    member = np.asarray(member, bool); y = np.asarray(y, int); strata = np.asarray(strata)
    pmf = np.array([1.0]); obs = 0; exp = 0.0; num = 0.0; den = 0.0
    for s in np.unique(strata[member]):
        ix = strata == s
        N = ix.sum(); m = y[ix].sum(); n = member[ix].sum()
        a = int((y[ix] & member[ix]).sum())
        obs += a
        lo, hi = max(0, n - (N - m)), min(n, m)
        k = np.arange(lo, hi + 1)
        p = hypergeom.pmf(k, N, m, n)
        full = np.zeros(hi + 1); full[lo:] = p
        pmf = np.convolve(pmf, full)
        exp += n * m / N
        b = n - a; c = m - a; d = N - n - c
        num += a * d / N; den += b * c / N
    pmf = pmf / pmf.sum()
    p_up = pmf[obs:].sum()
    p_two = min(1.0, pmf[pmf <= pmf[obs] * (1 + 1e-7)].sum())
    orr = num / den if den > 0 else np.inf
    return obs, exp, p_up, p_two, orr


def bh(p):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q
