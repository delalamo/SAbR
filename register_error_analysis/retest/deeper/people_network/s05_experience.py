#!/usr/bin/env python3
"""Lab and team experience before the study's first deposition (PDB-wide, entries.idx author lists), cryo-EM.

Experience measures (counts of PDB entries deposited before the study's first deposition; same-study entries
excluded by construction):
  lab_*      last author (loose name key, as the other lenses)
  team_max_* most experienced audit author on the entry (strict name key); *_na versions ignore authors whose
             surname is collision-prone (common East-Asian / Western surnames) to limit name-collision inflation
  nonlast_max_*  most experienced non-last author (proxy for the hands-on model builder)
  first_em_lab / first_em_team / first_abem_lab   no prior cryo-EM (antibody cryo-EM) entry
  has_hub    a PDB-wide multi-lab cryo-EM person (>= 5 labs as non-last author) is on the entry (core-staff proxy)
Models: logit, entry level with SEs clustered by study, and study level; adjusted for deposit-year category,
resolution and log2(number of V chains).  Sensitivities: drop collision-prone last-author names, drop SARS-CoV-2,
by period, joint model.
"""
import sys, re, collections, warnings
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from common import OUT, CACHE, load_studies, load_entries, bh
warnings.filterwarnings('ignore')

E = load_entries('EM')
A = pd.read_csv(f'{CACHE}/audit_authors_keyed.csv.gz')
COMMON = set(A[A.common].sur.unique())
IDX = pd.read_pickle(f'{CACHE}/idx_parsed.pkl')
abset = set(IDX[IDX.ab].pdb)

# per strict key history
hist = collections.defaultdict(lambda: collections.defaultdict(list))
for d, em, ab, sk in zip(IDX.dep.values.astype('datetime64[D]'), IDX.em.values, IDX.ab.values, IDX.skeys.values):
    for k in set(sk.split('|')):
        h = hist[k]; h['all'].append(d)
        if em: h['em'].append(d)
        if ab: h['ab'].append(d)
        if em and ab: h['abem'].append(d)
for k in hist:
    for t in list(hist[k]):
        hist[k][t] = np.sort(np.array(hist[k][t]))


def prior(k, t, d):
    h = hist.get(k)
    if h is None or t not in h:
        return 0
    return int(np.searchsorted(h[t], np.datetime64(d, 'D')))


akeys = A.groupby('pdb').apply(lambda g: list(zip(g.skey, g.common, g.is_last))).to_dict()
hubs = set(pd.read_csv(f'{CACHE}/hubs_private.csv').iloc[:, 0])
rows = []
for p, d in zip(E.pdb, E.study_dep0):
    r = {}
    lst = akeys.get(p, [])
    for t in ['em', 'ab', 'abem', 'all']:
        r[f'team_max_{t}_na'] = max([prior(k, t, d) for k, c, l in lst if not c] or [0])
        r[f'nonlast_max_{t}_na'] = max([prior(k, t, d) for k, c, l in lst if not c and not l] or [0])
    r['n_em_people_na'] = sum(1 for k, c, l in lst if not c and prior(k, 'em', d) > 0)
    r['has_hub'] = int(any(k in hubs and not l for k, c, l in lst))
    rows.append(r)
E = pd.concat([E, pd.DataFrame(rows)], axis=1)
for c in [c for c in E.columns if c.startswith(('lab_prior', 'team_max', 'nonlast_max'))] + ['n_auth', 'n_em_people_na', 'team_n_em_people']:
    E['l_' + c] = np.log2(1 + E[c])
E['first_em_team_na'] = (E.team_max_em_na == 0).astype(int)
E['first_abem_team_na'] = (E.team_max_abem_na == 0).astype(int)

# study-level frame
agg = {c: 'first' for c in E.columns if c.startswith(('l_', 'first_', 'lab_prior', 'team_max', 'nonlast', 'has_hub', 'n_em'))}
agg.update(dict(y='max', res='median', n_v='sum', ycat='first', period='first', sars2='max', lab_common='first', lab='first', year='first'))
S = E.groupby('study').agg(agg).reset_index()
S = S[S.study.isin(set(load_studies('EM').study))].reset_index(drop=True)   # same study set as the other scripts
S['l2_nv'] = np.log2(S.n_v)
S['has_hub'] = E.groupby('study').has_hub.max().reindex(S.study).values

EXPO = [('l_lab_prior_all', 'last author: prior PDB entries (log2 1+n)'),
        ('l_lab_prior_em', 'last author: prior cryo-EM entries'),
        ('l_lab_prior_ab', 'last author: prior antibody entries'),
        ('l_lab_prior_abem', 'last author: prior antibody cryo-EM entries'),
        ('first_em_lab', 'last author: first cryo-EM entry'),
        ('first_ab_lab', 'last author: first antibody entry'),
        ('first_abem_lab', 'last author: first antibody cryo-EM entry'),
        ('l_team_max_all_na', 'most experienced author: prior PDB entries'),
        ('l_team_max_em_na', 'most experienced author: prior cryo-EM entries'),
        ('l_team_max_ab_na', 'most experienced author: prior antibody entries'),
        ('l_team_max_abem_na', 'most experienced author: prior antibody cryo-EM entries'),
        ('first_em_team_na', 'no author with a prior cryo-EM entry'),
        ('first_abem_team_na', 'no author with a prior antibody cryo-EM entry'),
        ('l_nonlast_max_em_na', 'most experienced non-last author: prior cryo-EM entries'),
        ('l_nonlast_max_abem_na', 'most experienced non-last author: prior antibody cryo-EM entries'),
        ('l_n_em_people_na', 'authors with any prior cryo-EM entry (log2 1+n)'),
        ('has_hub', 'multi-lab cryo-EM hub person (core-staff proxy) on the entry'),
        ('l_n_auth', 'audit authors (log2 1+n)')]


def fit(d, f, term, cluster=None):
    try:
        if cluster is not None:
            m = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cluster])[0]})
        else:
            m = smf.glm(f, d, family=sm.families.Binomial()).fit()
        ci = np.exp(m.conf_int().loc[term]).values
        return np.exp(m.params[term]), ci[0], ci[1], m.pvalues[term]
    except Exception as ex:
        return np.nan, np.nan, np.nan, np.nan


T = []
for t, lab in EXPO:
    desc = lambda d: (d.loc[d.y == 1, t].mean(), d.loc[d.y == 0, t].mean()) if not t.startswith('l_') else \
        (np.median(2 ** d.loc[d.y == 1, t] - 1), np.median(2 ** d.loc[d.y == 0, t] - 1))
    for unit, d, f, cl in [('study', S, f'y ~ {t} + C(ycat) + res + l2_nv', None),
                           ('entry', E, f'y ~ {t} + C(ycat) + res + l2_nv', 'study'),
                           ('study, last-author name not collision-prone', S[~S.lab_common.astype(bool)], f'y ~ {t} + C(ycat) + res + l2_nv', None),
                           ('study, non-SARS-CoV-2', S[~S.sars2.astype(bool)], f'y ~ {t} + C(ycat) + res + l2_nv', None),
                           ('study, 2020-21', S[S.period == '2020-21'], f'y ~ {t} + C(ycat) + res + l2_nv', None),
                           ('study, 2022+', S[S.period == '2022+'], f'y ~ {t} + C(ycat) + res + l2_nv', None)]:
        o, lo, hi, p = fit(d, f, t, cl)
        e_, c_ = desc(d)
        T.append(dict(exposure=lab, term=t, unit=unit, err_value=e_, control_value=c_, OR=o, lo=lo, hi=hi, p=p,
                      n=len(d), n_err=int(d.y.sum())))
T = pd.DataFrame(T)
prim = T.unit.isin(['study', 'entry'])
T.loc[prim, 'q_primary'] = bh(T.loc[prim, 'p'])
T.to_csv(f'{OUT}/tables/experience_tests.csv', index=False)
pd.set_option('display.width', 250)
print(T[prim].round(4).to_string())
print('\nSensitivities (study level):')
print(T[~prim].pivot_table(index='exposure', columns='unit', values='OR').round(2).to_string())
print(T[~prim].pivot_table(index='exposure', columns='unit', values='p').round(3).to_string())

# joint model: is the hub/core-staff effect explained by experience and team size?
J = []
for f in ['y ~ has_hub + C(ycat) + res + l2_nv',
          'y ~ has_hub + l_lab_prior_abem + C(ycat) + res + l2_nv',
          'y ~ has_hub + l_lab_prior_abem + l_team_max_em_na + l_n_auth + C(ycat) + res + l2_nv',
          'y ~ l_lab_prior_abem + l_team_max_em_na + l_nonlast_max_abem_na + l_n_auth + C(ycat) + res + l2_nv']:
    m = smf.glm(f, S, family=sm.families.Binomial()).fit()
    for t in [x for x in m.params.index if not x.startswith(('C(', 'Intercept', 'res', 'l2_nv'))]:
        ci = np.exp(m.conf_int().loc[t]).values
        J.append(dict(model=f, term=t, OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t]))
J = pd.DataFrame(J)
J.to_csv(f'{OUT}/tables/experience_joint_models.csv', index=False)
print('\nJoint models (study level):')
print(J.round(3).to_string())

# descriptive: error rate by bins
S['labbin'] = pd.cut(2 ** S.l_lab_prior_abem - 1, [-1, 0, 4, 19, 1e6], labels=['0', '1-4', '5-19', '>=20'])
S['teambin'] = pd.cut(2 ** S.l_team_max_em_na - 1, [-1, 0, 9, 49, 1e6], labels=['0', '1-9', '10-49', '>=50'])
d1 = S.groupby(['period', 'labbin']).y.agg(['size', 'sum', 'mean']).round(3)
d2 = S.groupby(['period', 'teambin']).y.agg(['size', 'sum', 'mean']).round(3)
d3 = S.groupby(['period', 'has_hub']).y.agg(['size', 'sum', 'mean']).round(3)
with open(f'{OUT}/tables/experience_bins.txt', 'w') as fo:
    fo.write('Error rate per study by last-author prior antibody cryo-EM entries\n' + d1.to_string() +
             '\n\nby most-experienced author prior cryo-EM entries (collision-prone names ignored)\n' + d2.to_string() +
             '\n\nby multi-lab hub on the study\n' + d3.to_string() + '\n')
print(open(f'{OUT}/tables/experience_bins.txt').read())
S.to_csv(f'{CACHE}/study_experience.csv.gz', index=False)
