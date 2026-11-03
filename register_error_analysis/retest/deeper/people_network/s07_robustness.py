#!/usr/bin/env python3
"""Robustness checks for the two people-level positives (cryo-EM, study level):
  (1) team new to antibody cryo-EM (no audit author with a prior antibody cryo-EM PDB entry);
  (2) lab persistence (the same last author had an earlier first-occurrence error study).
Adjustments: antigen class (viral capsid; enzyme/other viral), any nanobody, any mouse V, SARS-CoV-2, novel V
region (no earlier PDB entry with >=95% V identity, from the seq_novelty lens if available), first antibody
structure of the antigen family.  Persistence variants: published only; earlier error >= 180 d before; earlier error
on a different antigen; earlier error in <=2021 and later study in 2022+.  Mode concordance within labs.
"""
import sys, collections, warnings
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from common import OUT, SP, CACHE, load_studies, bh
warnings.filterwarnings('ignore')
rng = np.random.default_rng(3)

S = pd.read_csv(f'{CACHE}/study_carrier_persistence.csv.gz', parse_dates=['dep0'])
X = pd.read_csv(f'{CACHE}/study_experience.csv.gz')[['study', 'first_abem_team_na', 'l_team_max_abem_na', 'l_lab_prior_ab']]
S = S.drop(columns=[c for c in ['first_abem_team_na'] if c in S]).merge(X, on='study', how='left')
E = pd.read_csv(f'{CACHE}/entries_people.csv.gz')
E = E[E.method == 'EM']
F = pd.read_csv(f'{OUT}/../antigen_competition/tables/focal_entries_antigen_features.csv.gz', usecols=['pdb', 'tgt_cls', 'tgt_keys', 'first_fam'])
CH = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'vhh', 'species'])
ch = CH.groupby('pdb').agg(vhh=('vhh', 'max'), mouse=('species', lambda s: (s == 'mouse').any()))
E = E.merge(F, on='pdb', how='left').merge(ch, left_on='pdb', right_index=True, how='left')
st = E.groupby('study').agg(capsid=('tgt_cls', lambda s: (s == 'viral capsid').any()),
                            enzyme=('tgt_cls', lambda s: s.isin(['enzyme', 'viral other']).any()),
                            vhh=('vhh', 'max'), mouse=('mouse', 'max'), first_fam=('first_fam', 'max'),
                            tgt=('tgt_keys', lambda s: set('|'.join(s.dropna().astype(str)).replace(';', '|').split('|')) - {'', 'nan'}))
S = S.merge(st, left_on='study', right_index=True, how='left')
for c in ['capsid', 'enzyme', 'vhh', 'mouse', 'first_fam']:
    S[c] = S[c].fillna(0).astype(int)
S['sars2'] = S.sars2.astype(int)
# novel V (seq_novelty lens), optional
try:
    N = pd.read_csv(f'{OUT}/../seq_novelty/tables/chain_novelty.csv.gz')
    col = [c for c in N.columns if 'novel' in c.lower()]
    print('seq_novelty columns:', col[:6])
except Exception as ex:
    N = None

T = []


def fit(d, f, t, label):
    m = smf.glm(f, d, family=sm.families.Binomial()).fit()
    ci = np.exp(m.conf_int().loc[t]).values
    T.append(dict(test=label, n=len(d), n_err=int(d.y.sum()), OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t]))


base = 'C(ycat) + res_med + l2_nv'
adj = base + ' + capsid + enzyme + vhh + mouse + sars2 + first_fam'
fit(S, f'y ~ first_abem_team_na + {base}', 'first_abem_team_na', 'new-to-antibody-cryo-EM team: base model')
fit(S, f'y ~ first_abem_team_na + {adj}', 'first_abem_team_na', 'new-to-antibody-cryo-EM team: + antigen class, nanobody, mouse, SARS-CoV-2, first-of-family')
fit(S, f'y ~ first_abem_team_na + l_lab_prior_ab + {adj}', 'first_abem_team_na', '... + last-author prior antibody entries')
fit(S[S.published], f'y ~ first_abem_team_na + {adj}', 'first_abem_team_na', '... published studies only')
# name-collision checks: the *_na flag ignores collision-prone surnames, so an all-common-surname team is "new" by
# construction.  (a) flag computed from all strict keys (collisions inflate experience instead);
# (b) teams with >= 2 non-collision-prone authors; (c) + number of audit authors and last-author country.
Eh = E.groupby('study').agg(team_max_abem_all=('team_max_abem', 'first'), n_auth_e=('n_auth', 'max'))
Aa = pd.read_csv(f'{CACHE}/audit_authors_keyed.csv.gz')
nc = Aa[~Aa.common].groupby('pdb').size()
Eh['n_noncommon'] = E.assign(nc=E.pdb.map(nc).fillna(0)).groupby('study').nc.max()
S = S.merge(Eh, left_on='study', right_index=True, how='left')
S['first_abem_team_all'] = (S.team_max_abem_all == 0).astype(int)
S['l_n_auth'] = np.log2(1 + S.n_auth_e)
PMc = pd.read_csv(f'{OUT}/../paper_meta/paper_meta.csv', usecols=['doi', 'last_country']).set_index('doi').last_country
S['cn'] = S.doi.map(PMc).fillna('unknown')
S['china'] = (S.cn == 'China').astype(int); S['usa'] = (S.cn == 'USA').astype(int)
print('share of studies flagged new team: na-flag', S.first_abem_team_na.mean().round(3), 'all-keys flag', S.first_abem_team_all.mean().round(3))
fit(S, f'y ~ first_abem_team_all + {adj}', 'first_abem_team_all', 'new-team flag from all name keys (collisions inflate experience)')
fit(S[S.n_noncommon >= 2], f'y ~ first_abem_team_na + {adj}', 'first_abem_team_na', '... teams with >=2 non-collision-prone authors')
fit(S, f'y ~ first_abem_team_na + l_n_auth + china + usa + {adj}', 'first_abem_team_na', '... + number of audit authors + last-author country (China, USA)')
fit(S[(S.capsid == 0) & (S.enzyme == 0)], f'y ~ first_abem_team_na + {base} + vhh + mouse + sars2', 'first_abem_team_na', '... excluding capsid and enzyme targets')

P = S[S.n_prior_own > 0].copy()
fit(P, f'y ~ exp_own + l_n_prior_own + {base}', 'exp_own', 'lab persistence: base model')
fit(P, f'y ~ exp_own + l_n_prior_own + {adj}', 'exp_own', 'lab persistence: + antigen class, nanobody, mouse, SARS-CoV-2, first-of-family')
fit(P, f'y ~ exp_own + l_n_prior_own + first_abem_team_na + {adj}', 'exp_own', 'lab persistence: + new-team flag')
fit(P[P.published], f'y ~ exp_own + l_n_prior_own + {adj}', 'exp_own', 'lab persistence: published only')

# variants that need the lab's history
ALL = pd.read_csv(f'{CACHE}/studies.csv.gz', parse_dates=['dep0'])
Eall = pd.read_csv(f'{CACHE}/entries_people.csv.gz').merge(F[['pdb', 'tgt_keys']], on='pdb', how='left')
tg = Eall.groupby('study').tgt_keys.apply(lambda s: set('|'.join(s.dropna().astype(str)).replace(';', '|').split('|')) - {'', 'nan'})
H = ALL.groupby('study').agg(dep0=('dep0', 'min'), lab=('lab', 'first'), err=('grp', lambda s: (s == 'err').any())).reset_index()
H['tgt'] = H.study.map(tg)
byl = {lb: g.sort_values('dep0') for lb, g in H.groupby('lab')}
v180, vdiff, vcross = [], [], []
for lb, d, tg_ in zip(P.lab, P.dep0, P.tgt):
    g = byl.get(lb)
    g = g[g.dep0 < d]
    tg_ = tg_ if isinstance(tg_, set) else set()
    v180.append(int((g.err & (g.dep0 <= d - pd.Timedelta(days=180))).any()))
    vdiff.append(int(any(e and not ((t if isinstance(t, set) else set()) & tg_) for e, t in zip(g.err, g.tgt))))
    vcross.append(int((g.err & (g.dep0.dt.year <= 2021)).any()))
P['exp_own_180'] = v180; P['exp_own_diffag'] = vdiff; P['exp_own_le2021'] = vcross
fit(P, f'y ~ exp_own_180 + l_n_prior_own + {adj}', 'exp_own_180', 'lab persistence: earlier error >= 180 d before')
fit(P, f'y ~ exp_own_diffag + l_n_prior_own + {adj}', 'exp_own_diffag', 'lab persistence: earlier error on a different antigen')
Q = P[P.year >= 2022]
fit(Q, f'y ~ exp_own_le2021 + l_n_prior_own + {adj}', 'exp_own_le2021', 'lab persistence: error <=2021 predicts 2022+ study')
T = pd.DataFrame(T)
T['q'] = bh(T.p)
T.to_csv(f'{OUT}/tables/robustness_tests.csv', index=False)
pd.set_option('display.width', 250)
print(T.round(4).to_string())
print('\npersistence descriptive: error rate if lab had an earlier error study vs not (labs with earlier studies):')
for c in ['exp_own', 'exp_own_180', 'exp_own_diffag']:
    print(c, P.groupby(c).y.agg(['size', 'sum', 'mean']).round(3).to_dict('index'))
print('2022+ studies, by lab error <=2021:', Q.groupby('exp_own_le2021').y.agg(['size', 'sum', 'mean']).round(3).to_dict('index'))

# ---------------------------------------------------------------- mode concordance within labs
R = pd.read_csv(f'{SP}/runs_first_occurrence.csv', usecols=['pdb', 'mode', 'method'])
R = R[R.method == 'EM']
st_of = dict(zip(E.pdb, E.study))
R['study'] = R.pdb.map(st_of)
modes = R.dropna(subset=['study']).groupby('study')['mode'].apply(set)
SE = S[(S.y == 1) & S.study.isin(modes.index)][['study', 'lab', 'period']].copy()
SE['modes'] = SE.study.map(modes)
SE = SE[SE.modes.map(lambda m: bool(m - {'other'}))]


def concord(df):
    c = 0; n = 0
    for lb, g in df.groupby('lab'):
        ms = g.modes.tolist()
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                n += 1; c += bool(ms[i] & ms[j])
    return c, n


obs, npairs = concord(SE)
null = []
for _ in range(5000):
    sh = SE.copy()
    for p_, g in sh.groupby('period'):
        sh.loc[g.index, 'modes'] = rng.permutation(g.modes.values)
    null.append(concord(sh)[0])
null = np.array(null)
res = dict(test='same-lab pairs of error studies sharing an error mode', pairs=npairs, observed=obs, null_mean=null.mean(),
           p_upper=(1 + (null >= obs).sum()) / 5001, labs_with_2plus=int((SE.lab.value_counts() >= 2).sum()))
print('\nmode concordance:', res)
pd.DataFrame([res]).to_csv(f'{OUT}/tables/mode_concordance_within_lab.csv', index=False)
print('mode mix of error studies in labs with >=2 error studies vs others:')
SE['multi'] = SE.lab.map(SE.lab.value_counts()) >= 2
mm = SE.explode('modes').groupby(['multi', 'modes']).size().unstack(0).fillna(0)
print((mm / mm.sum()).round(2).to_string())

# ---------------------------------------------------------------- definitional overlap: copied/artifact-only studies
# Experienced teams can only have *copied* errors (their own earlier structure); those studies are excluded from
# both groups in the main analysis.  Count them as errors here (outcome = any run of any kind).
SA = pd.read_csv(f'{CACHE}/studies.csv.gz')
SA = SA[(SA.method == 'EM') & SA.grp.isin(['err', 'clean', 'excl'])].copy()
SA['y_any'] = SA.grp.isin(['err', 'excl']).astype(int)
EA = pd.read_csv(f'{CACHE}/entries_people.csv.gz')
SA['first_abem_team_all'] = SA.study.map(EA.groupby('study').team_max_abem.first().eq(0).astype(int))
m = smf.glm('y_any ~ first_abem_team_all + C(ycat) + res_med + l2_nv', SA, family=sm.families.Binomial()).fit()
ci = np.exp(m.conf_int().loc['first_abem_team_all']).values
print('\noutcome = any run (first-occurrence or copied/artifact), all-keys new-team flag: OR %.2f (%.2f-%.2f) p=%.2g n=%d n_any=%d'
      % (np.exp(m.params['first_abem_team_all']), ci[0], ci[1], m.pvalues['first_abem_team_all'], len(SA), SA.y_any.sum()))
print('excluded (copied/artifact-only) studies by new-team flag:', SA[SA.grp == 'excl'].first_abem_team_all.value_counts().to_dict())
pd.DataFrame([dict(test='new-team flag (all keys), outcome = any run incl. copied/artifact', n=len(SA), n_err=int(SA.y_any.sum()),
                   OR=np.exp(m.params['first_abem_team_all']), lo=ci[0], hi=ci[1], p=m.pvalues['first_abem_team_all'])]).to_csv(
    f'{OUT}/tables/robustness_any_run_outcome.csv', index=False)

# ---------------------------------------------------------------- is network isolation just inexperience?
S['connected_x'] = (S.n_prior_x > 0).astype(int)
rows = []
for f in ['y ~ connected_x + C(ycat) + res_med + l2_nv',
          'y ~ connected_x + first_abem_team_na + C(ycat) + res_med + l2_nv',
          'y ~ connected_x + first_abem_team_na + l_n_auth + C(ycat) + res_med + l2_nv']:
    m = smf.glm(f, S, family=sm.families.Binomial()).fit()
    for t in ['connected_x', 'first_abem_team_na']:
        if t in m.params:
            ci = np.exp(m.conf_int().loc[t]).values
            rows.append(dict(model=f, term=t, OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t]))
print('\nconnectedness vs inexperience:'); print(pd.DataFrame(rows).round(4).to_string())
print('cross-tab connected_x x new team:', pd.crosstab(S.connected_x, S.first_abem_team_na).values.tolist())
pd.DataFrame(rows).to_csv(f'{OUT}/tables/connectedness_vs_experience.csv', index=False)

# ---------------------------------------------------------------- antibody-specific or general cryo-EM inexperience?
X5 = pd.read_csv(f'{CACHE}/study_experience.csv.gz')[['study', 'first_em_team_na']]
S = S.merge(X5, on='study', how='left')
S['exp3'] = np.where(S.first_em_team_na == 1, '1 no cryo-EM', np.where(S.first_abem_team_na == 1, '2 cryo-EM, no antibody cryo-EM', '3 antibody cryo-EM'))
print('\nerror rate by team experience category:'); print(S.groupby('exp3').y.agg(['size', 'sum', 'mean']).round(3).to_string())
m = smf.glm("y ~ C(exp3, Treatment('3 antibody cryo-EM')) + C(ycat) + res_med + l2_nv", S, family=sm.families.Binomial()).fit()
r3 = []
for t in [x for x in m.params.index if 'exp3' in x]:
    ci = np.exp(m.conf_int().loc[t]).values
    r3.append(dict(term=t, OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t]))
print(pd.DataFrame(r3).round(4).to_string())
pd.DataFrame(r3).to_csv(f'{OUT}/tables/experience_3cat.csv', index=False)

# ---------------------------------------------------------------- X-ray replication (antibody experience, all name keys)
SX = pd.read_csv(f'{CACHE}/studies.csv.gz')
SX = SX[(SX.method == 'X-ray') & SX.grp.isin(['err', 'clean'])].copy()
SX['y'] = (SX.grp == 'err').astype(int)
EX = pd.read_csv(f'{CACHE}/entries_people.csv.gz')
SX['first_ab_team_all'] = SX.study.map(EX[EX.method == 'X-ray'].groupby('study').team_max_ab.first().eq(0).astype(int))
SX['per'] = pd.cut(SX.year, [0, 2009, 2017, 2100], labels=['<=2009', '2010-17', '2018+']).astype(str)
m = smf.glm('y ~ first_ab_team_all + C(per) + res_med + l2_nv', SX, family=sm.families.Binomial()).fit()
ci = np.exp(m.conf_int().loc['first_ab_team_all']).values
print('\nX-ray: team with no prior antibody PDB entry (all keys): OR %.2f (%.2f-%.2f) p=%.3g; n=%d, err=%d; rates %s'
      % (np.exp(m.params['first_ab_team_all']), ci[0], ci[1], m.pvalues['first_ab_team_all'], len(SX), SX.y.sum(),
         SX.groupby('first_ab_team_all').y.agg(['size', 'sum']).values.tolist()))
pd.DataFrame([dict(test='X-ray: team with no prior antibody PDB entry (all name keys)', n=len(SX), n_err=int(SX.y.sum()),
                   OR=np.exp(m.params['first_ab_team_all']), lo=ci[0], hi=ci[1], p=m.pvalues['first_ab_team_all'])]).to_csv(
    f'{OUT}/tables/xray_new_team.csv', index=False)

# ---------------------------------------------------------------- is the new-team effect a map-quality effect?
CQ = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'lq_z', 'irm', 'fit'])
CQ['study'] = CQ.pdb.map(dict(zip(E.pdb, E.study)))
q = CQ.groupby('study').agg(lq_z=('lq_z', 'median'),
                            af_start=('irm', lambda s: s.astype(str).str.contains('AlphaFold|predicted|computational', case=False).any()),
                            flex=('fit', lambda s: s.astype(str).str.contains('flex', case=False).any()))
S2 = S.merge(q, left_on='study', right_index=True, how='left')
S2['af_start'] = S2.af_start.fillna(False).astype(int); S2['flex'] = S2.flex.fillna(False).astype(int)
rows = []
for f in ['y ~ first_abem_team_na + lq_z + C(ycat) + res_med + l2_nv',
          'y ~ first_abem_team_na + lq_z + af_start + flex + C(ycat) + res_med + l2_nv']:
    d = S2.dropna(subset=['lq_z'])
    m = smf.glm(f, d, family=sm.families.Binomial()).fit()
    for t in ['first_abem_team_na', 'lq_z', 'af_start', 'flex']:
        if t in m.params:
            ci = np.exp(m.conf_int().loc[t]).values
            rows.append(dict(model=f, term=t, n=len(d), OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t]))
print('\nnew-team effect adjusted for local map fit and modelling route:'); print(pd.DataFrame(rows).round(4).to_string())
print('new teams vs others: median lq_z', S2.groupby('first_abem_team_na').lq_z.median().round(3).to_dict(),
      'AlphaFold-type start', S2.groupby('first_abem_team_na').af_start.mean().round(3).to_dict(),
      'flexible fit', S2.groupby('first_abem_team_na').flex.mean().round(3).to_dict())
pd.DataFrame(rows).to_csv(f'{OUT}/tables/new_team_mapq_adjusted.csv', index=False)
