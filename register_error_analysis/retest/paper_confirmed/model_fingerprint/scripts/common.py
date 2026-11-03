"""Shared paths, group labels and statistics for the model-fingerprint re-test on the paper-confirmed case set."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DEEP = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint'
TAB = f'{OUT}/tables'
WINDOW_END = '2022-04-05'


def ybin(y):
    y = np.asarray(y)
    return np.where(y <= 2018, '<=2018', np.where(y >= 2024, '>=2024', y.astype(str)))


def ybin_win(y):
    """Year bins for the paper window (cases 2001-2021)."""
    y = np.asarray(y)
    return np.where(y <= 2017, '<=2017', np.where(y <= 2019, '2018-19', y.astype(str)))


def load_frame():
    return pd.read_csv(f'{TAB}/chain_frame.csv.gz', dtype={'chain': str})


# ---------------------------------------------------------------- statistics
def mde_or(se):
    """Minimum OR detectable with 80% power at two-sided alpha 0.05 given the SE of the log OR (protective side)."""
    return float(np.exp(-2.80 * se)) if np.isfinite(se) else np.nan


def mde_or_binary(n1, n0, p0, direction=-1, alpha=0.05, power=0.8):
    """Minimum detectable OR (Wald test on log OR, expected cells) for n1 cases vs n0 controls with control
    prevalence p0. direction -1: OR<1 (protective), +1: OR>1."""
    from scipy.optimize import brentq
    if n1 < 1 or n0 < 1 or not (0 < p0 < 1): return np.nan
    za = stats.norm.ppf(1 - alpha / 2); zb = stats.norm.ppf(power)
    def gap(lor):
        o = np.exp(lor); p1 = o * p0 / (1 - p0 + o * p0)
        se = np.sqrt(1 / (n1 * p1) + 1 / (n1 * (1 - p1)) + 1 / (n0 * p0) + 1 / (n0 * (1 - p0)))
        return abs(lor) / se - (za + zb)
    grid = np.linspace(1e-3, 8, 4000) * (1 if direction > 0 else -1)
    g = np.array([gap(x) for x in grid])
    ok = np.where(g >= 0)[0]
    if not len(ok): return np.nan   # not detectable at any OR (too few cases)
    i = ok[0]
    lor = brentq(gap, grid[i - 1], grid[i]) if i > 0 else grid[0]
    return float(np.exp(lor))


def logit(d, y, x, covars=(), cluster='cluster', weights=None):
    """Binomial GLM y ~ x + covars, cluster-robust SE. Returns (OR, lo, hi, p, se)."""
    f = f'{y} ~ {x}' + (' + ' + ' + '.join(covars) if covars else '')
    kw = {} if weights is None else {'var_weights': d[weights]}
    m = smf.glm(f, data=d, family=sm.families.Binomial(), **kw).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cluster])[0]})
    b, se = m.params[x], m.bse[x]
    return dict(OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=m.pvalues[x], se=se)


def ols(d, y, x, covars=(), cluster='cluster', weights=None):
    f = f'{y} ~ {x}' + (' + ' + ' + '.join(covars) if covars else '')
    if weights is None:
        m = smf.ols(f, data=d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cluster])[0]})
    else:
        m = smf.wls(f, data=d, weights=d[weights]).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cluster])[0]})
    b, se = m.params[x], m.bse[x]
    return dict(est=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=m.pvalues[x], se=se)


def fisher(a, b, c, d_):
    """2x2 [[a,b],[c,d]] with a = exposed cases. Returns Haldane OR, Woolf CI, Fisher p."""
    p = stats.fisher_exact([[a, b], [c, d_]])[1]
    A, B, Cc, D = [v + 0.5 for v in (a, b, c, d_)]
    orr = A * D / (B * Cc); se = np.sqrt(1 / A + 1 / B + 1 / Cc + 1 / D)
    return dict(OR=orr, lo=orr * np.exp(-1.96 * se), hi=orr * np.exp(1.96 * se), p=p, se=se)


def het_z(b1, se1, b2, se2):
    """z-test for the difference of two log ORs (or two differences) from independent samples."""
    z = (b1 - b2) / np.sqrt(se1 ** 2 + se2 ** 2)
    return float(2 * stats.norm.sf(abs(z)))


def wald_equal(model, a, b):
    """Wald test of coefficient a == coefficient b in a fitted statsmodels result."""
    names = list(model.params.index)
    r = np.zeros(len(names)); r[names.index(a)] = 1; r[names.index(b)] = -1
    return float(model.t_test(r).pvalue)


def paper_weights(d, cls='y', cluster='cluster'):
    """one vote per paper per class: each paper's chains of a class sum to 1"""
    return 1.0 / d.groupby([cluster, cls])[cls].transform('size')


def fmt_or(r):
    if r is None or not np.isfinite(r.get('OR', np.nan)):
        return 'n/a'
    return f"OR {r['OR']:.3g} ({r['lo']:.3g}-{r['hi']:.3g}), p={r['p']:.2g}"


def fmt_diff(r, unit=''):
    if r is None or not np.isfinite(r.get('est', np.nan)):
        return 'n/a'
    return f"{r['est']:+.3g}{unit} ({r['lo']:+.3g} to {r['hi']:+.3g}), p={r['p']:.2g}"
