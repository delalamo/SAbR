"""Shared loaders, group labels and statistics for the paper-confirmed re-test of the "antibodies are ordinary" family
(deeper README section 7; lenses seq_novelty, struct_unusual, literature_dean)."""
import warnings, numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings('ignore')

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DEEP = '/home/user/SAbR/register_error_analysis/retest/deeper'
PC = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed'
OUT = f'{PC}/antibody_ordinary'
TAB = f'{OUT}/tables'
CACHE = f'{SP}/paperord'
WINDOW_END = '2022-04-05'
STRICT_LEDGER = {'amyloid_fibril', 'indel_gap_artifact_confirmed', 'DE_chain_end_artifact', 'near_copy', 'not_a_register_shift'}


def chain_groups():
    """Per-chain group labels. Reuses model_fingerprint's chain_frame (grp, amyloid) and adds the synthesis-ledger filter
    for the 'rest' comparator: a rest chain must carry >=1 first-occurrence run that is not a strict ledger artifact."""
    C = pd.read_csv(f'{PC}/model_fingerprint/tables/chain_frame.csv.gz', dtype={'chain': str}, low_memory=False,
                    usecols=['pdb', 'chain', 'ctype', 'method', 'res', 'year', 'deposit', 'release', 'paper', 'unit', 'vhh',
                             'err_first', 'err_any', 'paper_case_strict', 'paper_case_first', 'paper_case_incl_copies',
                             'in_paper_window', 'paper_flagged_entry', 'paper_control', 'amyloid', 'grp', 'sars2'])
    R = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
    L = pd.read_csv(f'{DEEP}/synthesis/artifact_ledger_runs.csv', dtype={'chain': str})[['pdb', 'chain', 'imgt0', 'imgt1', 'flag']]
    R = R.merge(L, on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
    R['genuine'] = ~R.flag.isin(STRICT_LEDGER)
    gen = R.groupby(['pdb', 'chain']).genuine.any()
    C = C.join(gen.rename('has_genuine_run'), on=['pdb', 'chain'])
    C['has_genuine_run'] = C.has_genuine_run.fillna(False).astype(bool)
    C['rest'] = (C.grp == 'rest_case') & C.has_genuine_run & ~C.amyloid
    C['rest_win'] = C.rest & C.in_paper_window
    C['clean_all'] = ~C.err_any.astype(bool) & ~C.amyloid
    C['EM'] = C.method.eq('EM')
    return C


def unit_labels(C):
    """Unit (= paper x V sequence) labels from chain labels."""
    g = C.groupby('unit')
    U = pd.DataFrame({
        'paper_strict': g.paper_case_strict.any(),
        'paper_incl': g.paper_case_incl_copies.any() | g.paper_case_first.any(),
        'rest': g.rest.any(),
        'rest_win': g.rest_win.any(),
        'control': g.paper_control.all(),
        'clean_all': g.clean_all.all(),
        'any_amyloid': g.amyloid.any(),
    })
    U['rest'] &= ~U.paper_incl; U['rest_win'] &= ~U.paper_incl
    return U


def ybin_win(y):
    y = np.asarray(y, float)
    return np.where(y <= 2017, '<=2017', np.where(y <= 2019, '2018-19', np.where(y >= 2021, '>=2021', '2020')))


def ybin_full(y):
    y = np.asarray(y, float)
    return np.where(y <= 2018, '<=2018', np.where(y >= 2024, '>=2024', y.astype(int).astype(str)))


# ------------------------------------------------------------------ statistics
def logit_or(d, x, covs, y='y', groups=None, weights=None):
    """Logistic OR for x (binary or standardised continuous) with covariates; if groups given, GEE (independence) with
    cluster-robust SE; weights -> freq-weighted GLM with cluster-robust SE (one vote per paper)."""
    d = d.dropna(subset=[x] + [c.replace('C(', '').replace(')', '') for c in covs]).copy()
    covs = [c for c in covs if d[c.replace('C(', '').replace(')', '')].nunique() > 1]
    f = f'{y} ~ {x}' + ''.join(' + ' + c for c in covs)
    try:
        if weights is not None:
            m = smf.glm(f, data=d, family=sm.families.Binomial(), var_weights=d[weights]).fit(
                cov_type='cluster', cov_kwds={'groups': pd.factorize(d[groups])[0]})
        elif groups is not None:
            m = smf.gee(f, groups=pd.factorize(d[groups])[0], data=d, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Independence()).fit()
        else:
            m = smf.logit(f, data=d).fit(disp=0, maxiter=300)
        b, se = m.params[x], m.bse[x]
        if not np.isfinite(se) or se > 8:
            raise ValueError('separation')
        return dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)),
                    p=float(2 * norm.sf(abs(b / se))), b=float(b), se=float(se), n1=int(d[y].sum()), n0=int((d[y] == 0).sum()))
    except Exception as e:
        return dict(err=str(e)[:80], n1=int(d[y].sum()), n0=int((d[y] == 0).sum()))


def lin_diff(d, x, covs, y='y', groups=None, weights=None):
    """Adjusted mean difference of a continuous x between y=1 and y=0 (OLS of x on y + covariates), cluster-robust."""
    d = d.dropna(subset=[x] + [c.replace('C(', '').replace(')', '') for c in covs]).copy()
    covs = [c for c in covs if d[c.replace('C(', '').replace(')', '')].nunique() > 1]
    f = f'{x} ~ {y}' + ''.join(' + ' + c for c in covs)
    try:
        kw = dict(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[groups])[0]}) if groups else {}
        m = (smf.wls(f, data=d, weights=d[weights]) if weights else smf.ols(f, data=d)).fit(**kw)
        b, se = m.params[y], m.bse[y]
        return dict(diff=float(b), lo=float(b - 1.96 * se), hi=float(b + 1.96 * se), p=float(m.pvalues[y]), b=float(b),
                    se=float(se), n1=int(d[y].sum()), n0=int((d[y] == 0).sum()))
    except Exception as e:
        return dict(err=str(e)[:80])


def fisher(a1, n1, a0, n0):
    t = np.array([[a1, n1 - a1], [a0, n0 - a0]])
    OR, p = stats.fisher_exact(t)
    h = t + 0.5
    orh = h[0, 0] * h[1, 1] / (h[0, 1] * h[1, 0]); se = np.sqrt((1 / h).sum())
    return dict(OR=float(orh), lo=float(np.exp(np.log(orh) - 1.96 * se)), hi=float(np.exp(np.log(orh) + 1.96 * se)),
                p=float(p), b=float(np.log(orh)), se=float(se), fisher_OR=float(OR))


def z_het(a, b):
    """Two-sided p for difference of two estimates given dicts with b (log OR or diff) and se."""
    try:
        z = (a['b'] - b['b']) / np.sqrt(a['se'] ** 2 + b['se'] ** 2)
        return float(2 * norm.sf(abs(z)))
    except Exception:
        return np.nan


def mde_or(n1, p0, n0=None, alpha=0.05, power=0.8, lower=False):
    """Smallest OR (>1, or <1 if lower) detectable with given power for n1 cases vs n0 controls at control prevalence p0."""
    if not n1 or n1 < 2 or p0 is None or not (0 < p0 < 1):
        return np.nan
    n0 = n0 or 10 ** 6
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    grid = np.exp(np.linspace(0.001, np.log(200), 3000))
    for OR in (1 / grid if lower else grid):
        p1 = OR * p0 / (1 - p0 + OR * p0)
        pbar = (n1 * p1 + n0 * p0) / (n1 + n0)
        se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n0)); se1 = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        if (abs(p1 - p0) - za * se0) / se1 >= zb:
            return float(OR)
    return np.inf if not lower else 0.0


def f_or(r, k='OR'):
    if not r or k not in r or not np.isfinite(r.get(k, np.nan)):
        return 'n/a' + (f" ({r.get('err')})" if r and r.get('err') else '')
    return f"{r[k]:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.2g}"


def f_diff(r, nd=2):
    if not r or 'diff' not in r:
        return 'n/a'
    return f"{r['diff']:+.{nd}f} ({r['lo']:+.{nd}f} to {r['hi']:+.{nd}f}) p={r['p']:.2g}"


def paper_weights(d, y='y', paper='paper'):
    """weight = 1 / (# rows of the same paper and the same outcome): one vote per paper per group."""
    return 1.0 / d.groupby([paper, y])[y].transform('size')
