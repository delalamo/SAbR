#!/usr/bin/env python3
"""Does the same person link error studies across different labs?  Does an error "travel" with people?

1. Static linking (cryo-EM studies).  Statistics over people (ORCID-merged strict name keys from PDB audit
   authors):  (a) people on error studies of >= 2 different last-author labs, (b) the largest number of distinct
   labs' error studies that one person links, (c) people who are a non-last author on >= 2 error studies of
   different labs (model-builder / core-staff pattern).  Null: error labels permuted among studies within
   deposit-year x resolution-bin x size-bin strata (2,000 permutations), which keeps each study's baseline risk.
   Repeated without collision-prone name-only keys.
2. Temporal carrier test.  For each cryo-EM study, look at people on it (other than its last author) who were on an
   EARLIER antibody study (X-ray or EM, deposited before this study's first deposition) of a DIFFERENT lab.
   Exposure = at least one such earlier study had a first-occurrence error.  Among studies with any such earlier
   cross-lab study, logit: error ~ exposure + log2(1 + n earlier cross-lab studies) + year category + resolution +
   log2(V chains) (+ last-author antibody cryo-EM experience).  Same for the lab's own earlier error (persistence).
"""
import sys, collections, warnings
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from common import OUT, CACHE, load_studies, bh
warnings.filterwarnings('ignore')
rng = np.random.default_rng(5)
A = pd.read_csv(f'{CACHE}/audit_authors_keyed.csv.gz')
amb = set(A[A.common & ~A.pkey.str.startswith('o:')].pkey)

S = load_studies('EM')
S['stratum'] = S.ycat + '|' + S.resbin + '|' + S.sizebin
sets = S.akeys.map(lambda s: set(s.split('|')) - {''}).tolist()
sets_nl = S.akeys_nonlast.map(lambda s: set(s.split('|')) - {''}).tolist()
labs = S.lab.fillna('?').values


def link_stats(y, sets, sets_nl, drop=set()):
    labs_of = collections.defaultdict(set); labs_nl = collections.defaultdict(set)
    for i in np.where(y == 1)[0]:
        for k in sets[i] - drop:
            labs_of[k].add(labs[i])
        for k in sets_nl[i] - drop:
            labs_nl[k].add(labs[i])
    a = sum(1 for v in labs_of.values() if len(v) >= 2)
    b = max([len(v) for v in labs_of.values()] or [0])
    c = sum(1 for v in labs_nl.values() if len(v) >= 2)
    return a, b, c


strata = S.stratum.values
groups = [np.where(strata == s)[0] for s in np.unique(strata)]
rows = []
for drop, lab in [(set(), 'all person keys'), (amb, 'collision-prone name-only keys dropped')]:
    obs = link_stats(S.y.values, sets, sets_nl, drop)
    null = []
    for _ in range(2000):
        yp = S.y.values.copy()
        for g in groups:
            yp[g] = rng.permutation(yp[g])
        null.append(link_stats(yp, sets, sets_nl, drop))
    null = np.array(null)
    for j, nm in enumerate(['people on error studies of >=2 different labs', 'max distinct labs linked by one person',
                            'non-last authors on error studies of >=2 different labs']):
        rows.append(dict(keys=lab, stat=nm, observed=obs[j], null_mean=null[:, j].mean(), null_sd=null[:, j].std(),
                         ratio=obs[j] / null[:, j].mean() if null[:, j].mean() else np.nan,
                         p_upper=(1 + (null[:, j] >= obs[j]).sum()) / 2001, p_lower=(1 + (null[:, j] <= obs[j]).sum()) / 2001))
L = pd.DataFrame(rows)
L.to_csv(f'{OUT}/tables/crosslab_linking_perm.csv', index=False)
pd.set_option('display.width', 250)
print(L.round(3).to_string())

# ------------------------------------------------------------------ temporal carrier test
ALL = pd.read_csv(f'{CACHE}/studies.csv.gz', parse_dates=['dep0'])
ALL['akeys'] = ALL.akeys.fillna('')
ALL = ALL.groupby('study').agg(dep0=('dep0', 'min'), lab=('lab', 'first'), akeys=('akeys', lambda s: '|'.join(s)),
                               anyerr=('grp', lambda s: (s == 'err').any()), allclean=('grp', lambda s: (s == 'clean').all())).reset_index()
ALL['sets'] = ALL.akeys.map(lambda s: set(s.split('|')) - {''})
inv = collections.defaultdict(list)
for i, (d, lb, ss, e) in enumerate(zip(ALL.dep0, ALL.lab, ALL.sets, ALL.anyerr)):
    for k in ss:
        inv[k].append((d, lb, e, i))
lab_hist = collections.defaultdict(list)
for d, lb, e in zip(ALL.dep0, ALL.lab, ALL.anyerr):
    lab_hist[lb].append((d, e))

Sexp = pd.read_csv(f'{CACHE}/study_experience.csv.gz')[['study', 'l_lab_prior_abem', 'first_abem_team_na']]
S = S.merge(Sexp, on='study', how='left')
Akey_last = dict(zip(S.study, S.lab_pkey))
out = []
for st, d, lb, ss, lp in zip(S.study, S.dep0, S.lab, sets, S.lab_pkey):
    prior_x = {}; prior_x_na = {}
    for k in ss:
        if k == lp:
            continue
        for (d2, lb2, e2, i2) in inv.get(k, []):
            if d2 < d and lb2 != lb:
                prior_x[i2] = e2
                if k not in amb:
                    prior_x_na[i2] = e2
    own = [e for d2, e in lab_hist.get(lb, []) if d2 < d]
    out.append(dict(n_prior_x=len(prior_x), exp_x=int(any(prior_x.values())), n_prior_x_err=sum(prior_x.values()),
                    n_prior_x_na=len(prior_x_na), exp_x_na=int(any(prior_x_na.values())),
                    n_prior_own=len(own), exp_own=int(any(own))))
S = pd.concat([S, pd.DataFrame(out)], axis=1)
S['l_n_prior_x'] = np.log2(1 + S.n_prior_x); S['l_n_prior_x_na'] = np.log2(1 + S.n_prior_x_na); S['l_n_prior_own'] = np.log2(1 + S.n_prior_own)
S['frac_prior_x_err'] = S.n_prior_x_err / S.n_prior_x.replace(0, np.nan)
T = []


def fit(d, f, t):
    m = smf.glm(f, d, family=sm.families.Binomial()).fit()
    ci = np.exp(m.conf_int().loc[t]).values
    return np.exp(m.params[t]), ci[0], ci[1], m.pvalues[t]


for nm, d, f, t in [
    ('carrier: a non-last co-author was on an earlier error study of another lab', S[S.n_prior_x > 0],
     'y ~ exp_x + l_n_prior_x + C(ycat) + res_med + l2_nv', 'exp_x'),
    ('carrier, + lab antibody cryo-EM experience', S[S.n_prior_x > 0],
     'y ~ exp_x + l_n_prior_x + l_lab_prior_abem + C(ycat) + res_med + l2_nv', 'exp_x'),
    ('carrier, collision-prone names dropped', S[S.n_prior_x_na > 0],
     'y ~ exp_x_na + l_n_prior_x_na + C(ycat) + res_med + l2_nv', 'exp_x_na'),
    ('carrier: share of earlier cross-lab studies that had errors', S[S.n_prior_x > 0],
     'y ~ frac_prior_x_err + l_n_prior_x + C(ycat) + res_med + l2_nv', 'frac_prior_x_err'),
    ('any earlier cross-lab study via a co-author (connectedness)', S,
     'y ~ I(n_prior_x > 0) + C(ycat) + res_med + l2_nv', 'I(n_prior_x > 0)[T.True]'),
    ('persistence: the same lab had an earlier error study', S[S.n_prior_own > 0],
     'y ~ exp_own + l_n_prior_own + C(ycat) + res_med + l2_nv', 'exp_own'),
    ('persistence, + lab antibody cryo-EM experience', S[S.n_prior_own > 0],
     'y ~ exp_own + l_n_prior_own + l_lab_prior_abem + C(ycat) + res_med + l2_nv', 'exp_own')]:
    o, lo, hi, p = fit(d, f, t)
    expo = d[t.split('[')[0].replace('I(n_prior_x > 0)', 'n_prior_x')] if t != 'I(n_prior_x > 0)[T.True]' else (d.n_prior_x > 0).astype(int)
    if t == 'frac_prior_x_err':
        ev, cv = d.loc[d.y == 1, t].mean(), d.loc[d.y == 0, t].mean()
    else:
        ev, cv = d.loc[expo == 1, 'y'].mean(), d.loc[expo == 0, 'y'].mean()
    T.append(dict(test=nm, n=len(d), n_err=int(d.y.sum()), n_exposed=int((expo > 0).sum()), err_rate_exposed_or_mean_err=ev,
                  err_rate_unexposed_or_mean_clean=cv, OR=o, lo=lo, hi=hi, p=p))
T = pd.DataFrame(T)
T['q'] = bh(T.p)
T.to_csv(f'{OUT}/tables/carrier_persistence_tests.csv', index=False)
print(T.round(4).to_string())
S.drop(columns=['akeys', 'akeys_nonlast']).to_csv(f'{CACHE}/study_carrier_persistence.csv.gz', index=False)
