#!/usr/bin/env python3
"""Adversarial re-test of 'new team (no audit author with a prior antibody cryo-EM entry) -> ~2x first-occurrence
register errors' (people_network, cryo-EM, study level).  Uses the independent rebuild from v01_rebuild.py."""
import sys, json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
CV = f'{SP}/cache/verify_newteam'
PNC = f'{SP}/cache/people_network'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/people_network_new_team_ab_cryoem_integrity'
EE = pd.read_pickle(f'{CV}/entries_exposures.pkl')
EXPCOLS = [c for c in EE.columns if c.startswith(('max_', 'last_abem', 'n_auth', 'n_noncommon'))]


def study_frame(skey='study', method='EM', outcome='main'):
    """main: err = any first-occurrence entry; clean = all entries clean; studies with only clean+excl dropped.
    excl_as_err: excl entries count as errors.  excl_as_clean: excl entries count as clean."""
    E = EE[(EE.skey == skey) & (EE.method == method) & (EE.grp != 'amyloid')].copy()
    if outcome == 'excl_as_err':
        E['grp'] = E.grp.replace({'excl': 'err'})
    elif outcome == 'excl_as_clean':
        E['grp'] = E.grp.replace({'excl': 'clean'})
    g = E.groupby(skey)
    S = pd.DataFrame({'y': g.grp.apply(lambda s: int((s == 'err').any())),
                      'keep': g.grp.apply(lambda s: (s == 'err').any() or (s == 'clean').all()),
                      'dep0': g.dep.min(), 'n_entries': g.size()})
    Ek = E[E.grp.isin(['err', 'clean'])]
    gk = Ek.groupby(skey)
    S['res'] = gk.res.median(); S['n_v'] = gk.n_v.sum()
    S = S[S.keep].copy()
    # exposure: union of authors across the study's entries = max over entries (all evaluated at study dep0)
    for c in EXPCOLS:
        S[c] = g[c].max()
    # investigator-style: exposure from the earliest entry only
    first = E.sort_values(['dep', 'pdb']).groupby(skey).first()
    for c in ['max_abem_strict_na', 'max_abem_strict']:
        S[c + '_first'] = first[c]
    S['last_common'] = first.last_common; S['all_common'] = g.all_common.max()
    S['lastk'] = first.lastk
    S['year'] = S.dep0.dt.year
    S['ycat'] = pd.cut(S.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100],
                       labels=['<=2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)
    S['yfac'] = S.year.clip(lower=2016).astype(str)
    S['half'] = S.year.astype(str) + 'H' + np.where(S.dep0.dt.month <= 6, '1', '2')
    S['l2_nv'] = np.log2(S.n_v)
    S['published'] = ~S.index.str.startswith(('unpub', 'single'))
    S['res_bin'] = pd.cut(S.res, [0, 3.3, 3.7, 4.2, 10]).astype(str)
    return S.reset_index().rename(columns={skey: 'study'})


def logit(d, f, t, cluster=None):
    try:
        m = smf.glm(f, d, family=sm.families.Binomial()).fit()
        ci = np.exp(m.conf_int().loc[t]).values
        return dict(OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t], n=len(d), n_err=int(d.y.sum()))
    except Exception as ex:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(d), n_err=int(d.y.sum()), err=str(ex)[:60])


def desc(d, x):
    e = d[d.y == 1][x].mean(); c = d[d.y == 0][x].mean()
    r1 = d[d[x] == 1].y.mean(); r0 = d[d[x] == 0].y.mean()
    return dict(share_err=e, share_clean=c, rate_new=r1, rate_exp=r0, n_new=int((d[x] == 1).sum()),
                n_err_new=int(d[(d[x] == 1)].y.sum()))


R = []


def test(label, d, x, f=None, family='primary'):
    f = f or f'y ~ {x} + C(ycat) + res + l2_nv'
    r = dict(test=label, family=family, exposure=x, formula=f); r.update(desc(d, x)); r.update(logit(d, f, x))
    R.append(r); return r


S = study_frame()
print('study counts (EM):', S.y.value_counts().to_dict(), ' [investigator: 127 err / 1175 clean]')

# ------------------------------------------------------------------ exposure variants
for c, nm in [('max_abem_strict_na', 'nt_strict_na'), ('max_abem_strict', 'nt_strict'), ('max_abem_loose', 'nt_loose'),
              ('max_abem_sur', 'nt_sur'), ('max_abem_strict_na_first', 'nt_strict_na_first'),
              ('max_abem_strict_first', 'nt_strict_first'), ('max_abem_alt_strict', 'nt_alt_strict'),
              ('max_abem_union_strict', 'nt_union_strict'), ('last_abem_strict', 'nt_last_strict'),
              ('last_abem_loose', 'nt_last_loose'), ('max_em_strict', 'nt_em_strict')]:
    S[nm] = (S[c] == 0).astype(int)

# ------------------------------------------------------------------ agreement with the investigator's flag
I = pd.read_csv(f'{PNC}/study_experience.csv.gz')[['study', 'first_abem_team_na', 'y']]
Is = pd.read_csv(f'{PNC}/studies.csv.gz')
Is = Is[(Is.method == 'EM') & Is.grp.isin(['err', 'clean'])]
# map our study ids (doi:...) to theirs (paper key) via entries
Ep = pd.read_csv(f'{PNC}/entries_people.csv.gz', usecols=['pdb', 'study', 'method'])
Ep = Ep[Ep.method == 'EM']
ours = EE[(EE.skey == 'study') & (EE.method == 'EM')][['pdb', 'study']]
mp = ours.merge(Ep, on='pdb', suffixes=('_v', '_i')).drop_duplicates(['study_v', 'study_i'])
print('study id map: our studies', mp.study_v.nunique(), 'their studies', mp.study_i.nunique(),
      'one-to-one?', mp.groupby('study_v').study_i.nunique().max(), mp.groupby('study_i').study_v.nunique().max())
J = S.merge(mp.drop_duplicates('study_v')[['study_v', 'study_i']], left_on='study', right_on='study_v', how='left') \
     .merge(I.rename(columns={'y': 'y_inv'}), left_on='study_i', right_on='study', how='left', suffixes=('', '_inv'))
print('matched to investigator study frame:', J.first_abem_team_na.notna().sum(), 'of', len(J))
jj = J.dropna(subset=['first_abem_team_na'])
print('outcome agreement:', (jj.y == jj.y_inv).mean().round(4))
print('new-team flag agreement (ours nt_strict_na_first vs theirs):', (jj.nt_strict_na_first == jj.first_abem_team_na).mean().round(4))
print(pd.crosstab(jj.nt_strict_na_first, jj.first_abem_team_na, margins=True))
print('flag agreement (ours union nt_strict_na vs theirs):', (jj.nt_strict_na == jj.first_abem_team_na).mean().round(4))
S = S.merge(J[['study', 'first_abem_team_na']].rename(columns={'first_abem_team_na': 'nt_inv'}), on='study', how='left')

# ------------------------------------------------------------------ 1. reproduce headline
test('REPRO investigator flag (their study frame values merged)', S.dropna(subset=['nt_inv']).assign(nt_inv=lambda d: d.nt_inv.astype(int)), 'nt_inv')
test('rebuild: strict keys, common surnames ignored, first entry (= investigator definition)', S, 'nt_strict_na_first')
test('rebuild: strict keys, common surnames ignored, union of study entries', S, 'nt_strict_na')

# ------------------------------------------------------------------ 2. name-key handling
test('strict keys, ALL names counted (no common-surname exclusion)', S, 'nt_strict')
test('loose keys (surname+first initial), all names', S, 'nt_loose')
test('surname only, all names (maximal collision; most conservative)', S, 'nt_sur')
test('last author only, strict', S, 'nt_last_strict')
test('last author only, loose', S, 'nt_last_loose')
test('antibody set = regex+molecule name (no ANARCI), strict all names', S, 'nt_alt_strict')
test('antibody set = union, strict all names', S, 'nt_union_strict')
test('no author with ANY prior cryo-EM entry (strict, all names)', S, 'nt_em_strict')
# all-common-surname teams are 'new' by construction under the _na definition
print('\nall-common-surname teams:', S.all_common.sum(), 'flagged new (na):', S[S.all_common].nt_strict_na.mean().round(3),
      'flagged new (all names):', S[S.all_common].nt_strict.mean().round(3),
      'error rate', S[S.all_common].y.mean().round(3), 'vs others', S[~S.all_common].y.mean().round(3))
print('share new team (na flag) by last author common surname:', S.groupby('last_common').nt_strict_na.mean().round(3).to_dict())
print('share new team (all-name flag) by last author common surname:', S.groupby('last_common').nt_strict.mean().round(3).to_dict())
test('na flag, excluding all-common-surname teams', S[~S.all_common], 'nt_strict_na')
test('na flag, last author surname not common', S[~S.last_common], 'nt_strict_na')
test('na flag, >=2 non-common names', S[S.n_noncommon >= 2], 'nt_strict_na')
test('all-name flag, last author surname not common', S[~S.last_common], 'nt_strict')

# ------------------------------------------------------------------ 3. period adjustment
for f, lab in [('y ~ nt_strict + C(yfac) + res + l2_nv', 'calendar-year factor'),
               ('y ~ nt_strict + C(half) + res + l2_nv', 'half-year factor'),
               ('y ~ nt_strict + C(ycat) + C(res_bin) + l2_nv', 'resolution bins'),
               ('y ~ nt_strict + C(ycat) + res + l2_nv + np.log2(n_entries) + np.log2(1+n_auth)', '+ entries, author count'),
               ('y ~ nt_strict', 'unadjusted')]:
    test(f'all-name flag, {lab}', S, 'nt_strict', f, family='adjustment')
# exact year-stratified (CMH-type) test
from statsmodels.stats.contingency_tables import StratifiedTable
tabs = []
for yv, g in S.groupby('ycat'):
    t = pd.crosstab(g.nt_strict, g.y).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
    tabs.append(t)
st = StratifiedTable(tabs)
print('\nCMH (all-name flag, strata = year category): OR_MH %.2f, p=%.4g, homogeneity p=%.3g' %
      (st.oddsratio_pooled, st.test_null_odds().pvalue, st.test_equal_odds().pvalue))
tabs = [pd.crosstab(g.nt_strict_na, g.y).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values for _, g in S.groupby('ycat')]
st2 = StratifiedTable(tabs)
print('CMH (na flag): OR_MH %.2f, p=%.4g, homogeneity p=%.3g' % (st2.oddsratio_pooled, st2.test_null_odds().pvalue, st2.test_equal_odds().pvalue))

# ------------------------------------------------------------------ 4. per year / era heterogeneity
print('\nper deposit-year (all-name flag): ')
yy = []
for yv, g in S.groupby('ycat'):
    t = pd.crosstab(g.nt_strict, g.y).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
    o, p = fisher_exact(t.values)
    yy.append(dict(ycat=yv, n=len(g), n_err=int(g.y.sum()), rate_new=g[g.nt_strict == 1].y.mean(), rate_exp=g[g.nt_strict == 0].y.mean(),
                   share_new=g.nt_strict.mean(), OR=o, p_fisher=p))
yy = pd.DataFrame(yy); print(yy.round(3).to_string()); yy.to_csv(f'{OUT}/per_year_all_name_flag.csv', index=False)
yy2 = []
for yv, g in S.groupby('ycat'):
    t = pd.crosstab(g.nt_strict_na, g.y).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
    o, p = fisher_exact(t.values)
    yy2.append(dict(ycat=yv, n=len(g), n_err=int(g.y.sum()), rate_new=g[g.nt_strict_na == 1].y.mean(),
                    rate_exp=g[g.nt_strict_na == 0].y.mean(), OR=o, p_fisher=p))
print('per year (na flag):'); print(pd.DataFrame(yy2).round(3).to_string())
S['late'] = (S.year >= 2024).astype(int)
for x in ['nt_strict', 'nt_strict_na']:
    test(f'{x}, deposited <=2023', S[S.late == 0], x, family='era')
    test(f'{x}, deposited >=2024', S[S.late == 1], x, family='era')
    test(f'{x}, deposited <=2019', S[S.year <= 2019], x, family='era')
    test(f'{x}, deposited 2020-21', S[S.year.between(2020, 2021)], x, family='era')
    test(f'{x}, deposited 2022-23', S[S.year.between(2022, 2023)], x, family='era')

# ------------------------------------------------------------------ 5. outcome and unit variants
for oc in ['excl_as_err', 'excl_as_clean']:
    So = study_frame(outcome=oc)
    So['nt_strict'] = (So.max_abem_strict == 0).astype(int); So['nt_strict_na'] = (So.max_abem_strict_na == 0).astype(int)
    test(f'outcome {oc}, all-name flag', So, 'nt_strict', family='outcome')
    test(f'outcome {oc}, na flag', So, 'nt_strict_na', family='outcome')
Ss = study_frame(skey='study_single')
Ss['nt_strict'] = (Ss.max_abem_strict == 0).astype(int); Ss['nt_strict_na'] = (Ss.max_abem_strict_na == 0).astype(int)
test('unpublished entries as single-entry studies, all-name flag', Ss, 'nt_strict', family='outcome')
test('published studies only, all-name flag', S[S.published], 'nt_strict', family='outcome')
# entry level, clustered by study
Ee = EE[(EE.skey == 'study') & (EE.method == 'EM') & EE.grp.isin(['err', 'clean'])].copy()
Ee['y'] = (Ee.grp == 'err').astype(int)
Ee['ycat'] = pd.cut(Ee.dep0.dt.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100]).astype(str)
Ee['l2_nv'] = np.log2(Ee.n_v)
ent = Ee.groupby('study').max_abem_strict.transform('max')
Ee['nt_strict'] = (ent == 0).astype(int)
m = smf.glm('y ~ nt_strict + C(ycat) + res + l2_nv', Ee, family=sm.families.Binomial()).fit(
    cov_type='cluster', cov_kwds={'groups': pd.factorize(Ee.study)[0]})
ci = np.exp(m.conf_int().loc['nt_strict']).values
R.append(dict(test='entry level, all-name flag, SE clustered by study', family='outcome', exposure='nt_strict',
              OR=np.exp(m.params['nt_strict']), lo=ci[0], hi=ci[1], p=m.pvalues['nt_strict'], n=len(Ee), n_err=int(Ee.y.sum())))

# ------------------------------------------------------------------ 6. within-lab comparison (learning vs lab type)
# conditional logit by last author (loose key) among labs with >=2 studies and discordant outcomes
from statsmodels.discrete.conditional_models import ConditionalLogit
for x in ['nt_last_loose', 'nt_strict']:
    L = S.dropna(subset=['lastk']).copy()
    L = L[L.groupby('lastk').study.transform('size') >= 2]
    disc = L.groupby('lastk').y.transform(lambda s: s.nunique() > 1) & L.groupby('lastk')[x].transform(lambda s: s.nunique() > 1)
    Ld = L[disc]
    try:
        cm = ConditionalLogit(Ld.y, Ld[[x, 'res']].assign(yr=Ld.year - 2020), groups=pd.factorize(Ld.lastk)[0]).fit(disp=0)
        ci = np.exp(cm.conf_int().loc[x]).values
        R.append(dict(test=f'within last-author lab (conditional logit, +res, +year), {x}', family='within_lab', exposure=x,
                      OR=np.exp(cm.params[x]), lo=ci[0], hi=ci[1], p=cm.pvalues[x], n=len(Ld), n_err=int(Ld.y.sum()),
                      n_labs=Ld.lastk.nunique()))
    except Exception as ex:
        print('clogit failed', ex)
# between-lab: labs' first-ever study (new) vs one-off labs
S['lab_n_studies'] = S.groupby('lastk').study.transform('size')
S['oneoff'] = (S.lab_n_studies == 1).astype(int)
test('one-off last-author labs (only one study in the set), all-name flag', S[S.oneoff == 1], 'nt_strict', family='within_lab')
test('labs with >=2 studies, all-name flag', S[S.oneoff == 0], 'nt_strict', family='within_lab')
test('all-name flag adjusted for one-off lab', S, 'nt_strict', 'y ~ nt_strict + oneoff + C(ycat) + res + l2_nv', family='within_lab')

# ------------------------------------------------------------------ 7. leave-one-out fragility
lo_rows = []
for col in ['lastk', 'ycat']:
    for v in S[col].dropna().unique():
        d = S[S[col] != v]
        r = logit(d, 'y ~ nt_strict + C(ycat) + res + l2_nv', 'nt_strict')
        r2 = logit(d, 'y ~ nt_strict_na + C(ycat) + res + l2_nv', 'nt_strict_na')
        lo_rows.append(dict(drop_by=col, value=v, n_dropped=int((S[col] == v).sum()), OR_all=r['OR'], p_all=r['p'], OR_na=r2['OR'], p_na=r2['p']))
LO = pd.DataFrame(lo_rows)
LO.to_csv(f'{OUT}/leave_one_out.csv', index=False)
print('\nleave-one-lab-out (all-name flag): OR range %.2f-%.2f, max p %.4f; (na flag) OR %.2f-%.2f, max p %.4f' %
      (LO[LO.drop_by == 'lastk'].OR_all.min(), LO[LO.drop_by == 'lastk'].OR_all.max(), LO[LO.drop_by == 'lastk'].p_all.max(),
       LO[LO.drop_by == 'lastk'].OR_na.min(), LO[LO.drop_by == 'lastk'].OR_na.max(), LO[LO.drop_by == 'lastk'].p_na.max()))
print('leave-one-year-category-out:'); print(LO[LO.drop_by == 'ycat'].round(4).to_string())
# drop the k labs with the most new-team error studies
top = S[(S.y == 1) & (S.nt_strict == 1)].lastk.value_counts()
print('labs with >1 new-team error study (all-name flag):', (top > 1).sum(), top.head(5).values.tolist())

# permutation test within year category (all-name flag), statistic = adjusted log OR
rng = np.random.default_rng(11)
obs = np.log(logit(S, 'y ~ nt_strict + C(ycat) + res + l2_nv', 'nt_strict')['OR'])
null = []
for _ in range(1000):
    Sp = S.copy()
    Sp['nt_strict'] = Sp.groupby('ycat').nt_strict.transform(lambda s: rng.permutation(s.values))
    null.append(np.log(logit(Sp, 'y ~ nt_strict + C(ycat) + res + l2_nv', 'nt_strict')['OR']))
null = np.array(null)
print('\nwithin-year permutation (all-name flag): obs logOR %.3f, perm p two-sided %.4f' %
      (obs, (1 + (np.abs(null) >= abs(obs)).sum()) / 1001))
R.append(dict(test='within-year permutation, all-name flag (1000 perms)', family='primary', exposure='nt_strict',
              OR=np.exp(obs), p=(1 + (np.abs(null) >= abs(obs)).sum()) / 1001, n=len(S), n_err=int(S.y.sum())))

R = pd.DataFrame(R)
R['q_all'] = multipletests(R.p.fillna(1), method='fdr_bh')[1]
R.to_csv(f'{OUT}/verify_tests.csv', index=False)
print(R[['test', 'n', 'n_err', 'share_err', 'share_clean', 'rate_new', 'rate_exp', 'OR', 'lo', 'hi', 'p', 'q_all']].round(4).to_string())
S.to_csv(f'{CV}/study_frame_verify.csv.gz', index=False)
