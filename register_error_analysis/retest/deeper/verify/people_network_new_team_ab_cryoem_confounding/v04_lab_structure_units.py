#!/usr/bin/env python3
"""(a) Lab structure: GEE exchangeable by lab, inverse-lab-size weights, error rate of experienced studies by lab productivity.
(b) Any-run outcome with the investigator's main (_na) flag, including copied/artifact-only studies.
(c) Antibody-in-paper unit level, SEs clustered by study."""
import warnings, collections, numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
PN = f'{SP}/cache/people_network'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/people_network_new_team_ab_cryoem_confounding/tables'
S = pd.read_csv(f'{SP}/cache/verify_people_newteam_conf/study_frame.csv.gz')
BASE = 'C(ycat) + res_med + l2_nv'
rows = []


def rec(label, m, term, d, extra=None):
    ci = np.exp(m.conf_int().loc[term]).values
    r = dict(test=label, n=len(d), n_err=int(d.y.sum()), OR=np.exp(m.params[term]), lo=ci[0], hi=ci[1], p=m.pvalues[term])
    r.update(extra or {})
    rows.append(r)


# (a) lab structure
S['lab_n'] = S.groupby('lab_key').study.transform('size')
S['w_lab'] = 1 / S.lab_n
S['w_lab'] = S.w_lab / S.w_lab.mean()
S = S.sort_values('lab_key').reset_index(drop=True)
m = smf.gee(f'y ~ new + {BASE}', 'lab_key', S, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
rec('GEE exchangeable by last-author lab', m, 'new', S, dict(rho=m.cov_struct.dep_params))
m = smf.glm(f'y ~ new + {BASE}', S, family=sm.families.Binomial(), var_weights=S.w_lab).fit(
    cov_type='cluster', cov_kwds={'groups': pd.factorize(S.lab_key)[0]})
rec('weights 1/(studies of the lab), clustered SE', m, 'new', S)
S['l2_labn'] = np.log2(S.lab_n)
m = smf.glm(f'y ~ new + {BASE} + l2_labn', S, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(S.lab_key)[0]})
rec('+ log2 lab productivity (studies of the lab in the set)', m, 'new', S, dict(OR_l2_labn=np.exp(m.params['l2_labn']), p_l2_labn=m.pvalues['l2_labn']))
S['labbin'] = pd.cut(S.lab_n, [0, 1, 3, 9, 1000], labels=['1', '2-3', '4-9', '>=10']).astype(str)
print('error rate by lab productivity x new team:')
print(S.groupby(['labbin', 'new']).y.agg(['size', 'sum', 'mean']).unstack('new').round(3).to_string())
# restrict to labs with <=3 studies in the set (removes prolific labs)
d = S[S.lab_n <= 3]
m = smf.glm(f'y ~ new + {BASE}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.lab_key)[0]})
rec('labs with <=3 studies in the set', m, 'new', d)
d = S[S.lab_n >= 4]
m = smf.glm(f'y ~ new + {BASE}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.lab_key)[0]})
rec('labs with >=4 studies in the set', m, 'new', d)

# (b) any-run outcome with the _na flag: compute the flag for copied/artifact-only (excl) studies
A = pd.read_csv(f'{PN}/audit_authors_keyed.csv.gz')
IDX = pd.read_pickle(f'{PN}/idx_parsed.pkl')
Ep = pd.read_csv(f'{PN}/entries_people.csv.gz', parse_dates=['study_dep0'])
St = pd.read_csv(f'{PN}/studies.csv.gz', parse_dates=['dep0'])
Ex = St[(St.method == 'EM') & (St.grp == 'excl')]
ExE = Ep[(Ep.method == 'EM') & Ep.study.isin(Ex.study)]
first_e = ExE.groupby('study').first()          # same convention as s05 ('first' entry of the study)
need = set(A[A.pdb.isin(first_e.pdb)].skey)
hist = collections.defaultdict(list)
for dd, em, ab, sk in zip(IDX.dep.values.astype('datetime64[D]'), IDX.em.values, IDX.ab.values, IDX.skeys.values):
    if em and ab:
        for k in set(sk.split('|')) & need:
            hist[k].append(dd)
hist = {k: np.sort(np.array(v)) for k, v in hist.items()}
ak = A.groupby('pdb').apply(lambda g: list(zip(g.skey, g.common))).to_dict()
flag = {}
for st, r in first_e.iterrows():
    dd = np.datetime64(r.study_dep0, 'D')
    vals = [int(np.searchsorted(hist[k], dd)) if k in hist else 0 for k, c in ak.get(r.pdb, []) if not c]
    flag[st] = int(max(vals or [0]) == 0)
X = Ex[['study', 'dep0', 'res_med', 'n_v', 'lab']].copy()
X['new'] = X.study.map(flag)
X['y'] = 1
X['ycat'] = pd.cut(X.dep0.dt.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100], labels=['<=2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)
X['l2_nv'] = np.log2(X.n_v)
X['lab_key'] = X.lab.fillna('?' + X.study)
print('\ncopied/artifact-only studies:', len(X), 'new-team (na flag):', X.new.sum())
SA = pd.concat([S[['study', 'y', 'new', 'ycat', 'res_med', 'l2_nv', 'lab_key']], X[['study', 'y', 'new', 'ycat', 'res_med', 'l2_nv', 'lab_key']]])
m = smf.glm(f'y ~ new + {BASE}', SA, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(SA.lab_key)[0]})
rec('outcome = any run (first-occurrence or copied/artifact), na flag', m, 'new', SA)

# (c) unit level (antibody-in-paper units in the cryo-EM study set)
CH = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'unit', 'ctype', 'vhh', 'species', 'res', 'err_first', 'err_any', 'method'])
st_of = dict(zip(Ep.pdb, Ep.study))
CH = CH[CH.method == 'EM'].copy()
CH['study'] = CH.pdb.map(st_of)
CH = CH[CH.study.isin(S.study)]
U = CH.groupby(['study', 'unit']).agg(y=('err_first', 'max'), anyrun=('err_any', 'max'), ctype=('ctype', 'first'), vhh=('vhh', 'max'),
                                      mouse=('species', lambda s: (s == 'mouse').any()), res=('res', 'median'), n=('pdb', 'size')).reset_index()
U = U[(U.y == 1) | (U.anyrun == 0)]            # error units vs clean units
U['y'] = U.y.astype(int)
U = U.merge(S[['study', 'new', 'ycat', 'lab_key', 'sars2']], on='study')
U['l2n'] = np.log2(U.n)
m = smf.glm('y ~ new + C(ycat) + res + l2n + C(ctype) + vhh + mouse + sars2', U, family=sm.families.Binomial()).fit(
    cov_type='cluster', cov_kwds={'groups': pd.factorize(U.study)[0]})
rec('unit level (antibody-in-paper), clustered by study, + chain type, nanobody, mouse, SARS-CoV-2', m, 'new', U)
m = smf.glm('y ~ new + C(ycat) + res + l2n + C(ctype) + vhh + mouse + sars2', U, family=sm.families.Binomial()).fit(
    cov_type='cluster', cov_kwds={'groups': pd.factorize(U.lab_key)[0]})
rec('unit level, clustered by lab', m, 'new', U)
for ct in ['H', 'K', 'L']:
    d = U[U.ctype == ct]
    m = smf.glm('y ~ new + C(ycat) + res + l2n + vhh + mouse + sars2' if ct == 'H' else 'y ~ new + C(ycat) + res + l2n + mouse + sars2', d,
                family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.study)[0]})
    rec(f'unit level, chain type {ct}', m, 'new', d)
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/v04_lab_structure_units.csv', index=False)
pd.set_option('display.width', 250)
print(T.round(4).to_string())
