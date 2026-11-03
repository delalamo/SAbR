#!/usr/bin/env python3
"""Confounding checks for the 'new team' (no prior antibody cryo-EM entry) association with first-occurrence errors.
Study level, cryo-EM, error studies vs clean studies (copied/artifact-only studies excluded, as the investigator did)."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.contingency_tables import StratifiedTable
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/people_network_new_team_ab_cryoem_confounding/tables'
S = pd.read_csv(f'{SP}/cache/verify_people_newteam_conf/study_frame.csv.gz')
S['lq_z_med'] = S.lq_z_med.fillna(S.lq_z_med.median())
S['q_med'] = S.q_med.fillna(S.q_med.median())
S['tier'] = S.tier.fillna('Unpublished')
rows = []


def fit(d, f, label, term='new', cluster='lab_key'):
    try:
        if cluster:
            m = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cluster])[0]})
        else:
            m = smf.glm(f, d, family=sm.families.Binomial()).fit()
        ci = np.exp(m.conf_int().loc[term]).values
        r = dict(test=label, formula=f, cluster=cluster or 'none', n=len(d), n_err=int(d.y.sum()),
                 n_new=int(d[term].sum()) if term in d else np.nan,
                 rate_new=d.loc[d[term] == 1, 'y'].mean() if term in d else np.nan,
                 rate_old=d.loc[d[term] == 0, 'y'].mean() if term in d else np.nan,
                 OR=np.exp(m.params[term]), lo=ci[0], hi=ci[1], p=m.pvalues[term])
    except Exception as ex:
        r = dict(test=label, formula=f, OR=np.nan, p=np.nan, err=str(ex)[:80])
    rows.append(r)
    return r


BASE = 'C(ycat) + res_med + l2_nv'
fit(S, f'y ~ new + {BASE}', 'A0 reproduce investigator base model (no clustering)', cluster=None)
fit(S, f'y ~ new + {BASE}', 'A1 base model, SEs clustered by last-author lab')
# ---- time
fit(S, f'y ~ new + C(dep_year) + res_med + l2_nv', 'B1 deposit year (each year) instead of year category')
fit(S, f'y ~ new + bs(dep_dec, df=6) + res_med + l2_nv', 'B2 spline of deposit date (df 6)')
fit(S, f'y ~ new + C(dep_half) + res_med + l2_nv', 'B3 deposit half-year fixed effects')
# ---- resolution
fit(S, f'y ~ new + C(ycat) + bs(res_med, df=4) + l2_nv', 'C1 resolution spline (df 4)')
fit(S, f'y ~ new + C(ycat) * res_med + l2_nv', 'C2 resolution x year-category interaction')
# ---- size / chain type / SARS-CoV-2
fit(S, f'y ~ new + {BASE} + l2_ne', 'D1 + log2 number of entries')
fit(S, f'y ~ new + {BASE} + any_K + any_L + vhh_any + vhh_only', 'D2 + chain-type composition (kappa, lambda, any/only nanobody)')
fit(S, f'y ~ new + {BASE} + sars2', 'D3 + SARS-CoV-2')
fit(S, f'y ~ new + {BASE} + sars2 + C(ycat):sars2', 'D4 + SARS-CoV-2 x year category')
# ---- antibody atypicality / novelty (location-specific atypicality already known)
fit(S, f'y ~ new + {BASE} + mouse_any + vhh_any + atyp_any + min_vid', 'E1 + mouse, nanobody, atypical CDR length, min germline identity')
fit(S, f'y ~ new + {BASE} + novel_any', 'E2 + any antibody never solved before (seq_novelty)')
fit(S, f'y ~ new + {BASE} + aux_any + subject_any', 'E3 + auxiliary/subject role')
fit(S, f'y ~ new + {BASE} + capsid + enzyme + first_fam + novel_target', 'E4 + antigen class, first-of-family, novel target')
# ---- local map / unbuilt residues / modelling route (simpler known factors)
fit(S, f'y ~ new + {BASE} + lq_z_med + q_med', 'F1 + local map-model fit (median lq_z, chain Q)')
fit(S, f'y ~ new + {BASE} + unobs_max', 'F2 + unbuilt-residue fraction (max over V chains)')
fit(S, f'y ~ new + {BASE} + af_any + exp_start_any + irm_none + rigid_any + flex_any', 'F3 + starting-model type and fitting protocol')
# ---- people/paper
fit(S, f'y ~ new + {BASE} + l_n_auth', 'G1 + number of audit authors')
fit(S, f'y ~ new + {BASE} + C(ctry3)', 'G2 + last-author country (China/USA/other/unknown)')
fit(S, f'y ~ new + {BASE} + C(tier)', 'G3 + journal tier')
fit(S, f'y ~ new + {BASE} + l_lab_prior_ab', 'G4 + last-author prior antibody entries')
FULL = ('bs(dep_dec, df=6) + bs(res_med, df=4) + l2_nv + l2_ne + any_K + any_L + vhh_any + sars2 + mouse_any + atyp_any + '
        'min_vid + novel_any + aux_any + capsid + enzyme + first_fam + lq_z_med + q_med + unobs_max + af_any + exp_start_any + '
        'irm_none + rigid_any + flex_any + l_n_auth + C(ctry3) + C(tier)')
fit(S, f'y ~ new + {FULL}', 'H1 kitchen-sink model (all of the above)')
fit(S, f'y ~ new_all + {FULL}', 'H2 kitchen-sink, all-name-key flag', term='new_all')
fit(S, f'y ~ new_all + {BASE}', 'H3 base, all-name-key flag', term='new_all')
# ---- restrictions
for lab, d in [('R1 non-SARS-CoV-2 studies', S[S.sars2 == 0]), ('R2 SARS-CoV-2 studies', S[S.sars2 == 1]),
               ('R3 all antibodies novel (no copying possible)', S[S.novel_all == 1]),
               ('R4 some antibody solved before', S[S.novel_all == 0]),
               ('R5 single-entry studies', S[S.n_entries == 1]), ('R6 multi-entry studies', S[S.n_entries > 1]),
               ('R7 res < 3.5', S[S.res_med < 3.5]), ('R8 res >= 3.5', S[S.res_med >= 3.5]),
               ('R9 no nanobody', S[S.vhh_any == 0]), ('R10 any nanobody', S[S.vhh_any == 1]),
               ('R11 published', S[S.published == 1]), ('R12 China last author', S[S.china == 1]),
               ('R13 non-China', S[S.china == 0]), ('R14 deposited <=2023', S[S.dep_year <= 2023]),
               ('R15 deposited >=2024', S[S.dep_year >= 2024]), ('R16 deposited 2020-2023', S[S.dep_year.between(2020, 2023)]),
               ('R17 last-author name not collision-prone', S[S.lab_common == False]),
               ('R18 no auxiliary antibody', S[S.aux_any == 0])]:
    yc = 'C(ycat)' if d.ycat.nunique() > 1 else '1'
    fit(d, f'y ~ new + {yc} + res_med + l2_nv', lab)
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/v02_adjusted_models.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(T[['test', 'n', 'n_err', 'n_new', 'rate_new', 'rate_old', 'OR', 'lo', 'hi', 'p']].round(4).to_string())

# ---- Mantel-Haenszel stratified by deposit year x resolution bin
mh = []
for nm, strat in [('year', S.dep_year.astype(str)), ('year x resbin', S.dep_year.astype(str) + '|' + S.resbin),
                  ('half-year x resbin', S.dep_half + '|' + S.resbin),
                  ('year x sars2 x resbin', S.dep_year.astype(str) + '|' + S.sars2.astype(str) + '|' + S.resbin)]:
    tabs = []
    for s, g in S.groupby(strat):
        t = np.array([[((g.new == 1) & (g.y == 1)).sum(), ((g.new == 1) & (g.y == 0)).sum()],
                      [((g.new == 0) & (g.y == 1)).sum(), ((g.new == 0) & (g.y == 0)).sum()]])
        if t.sum(0).min() > 0 and t.sum(1).min() > 0:
            tabs.append(t)
    st = StratifiedTable(tabs)
    tst = st.test_null_odds(correction=True)
    bd = st.test_equal_odds()
    ci = st.oddsratio_pooled_confint()
    mh.append(dict(strata=nm, n_strata_informative=len(tabs), n=int(sum(t.sum() for t in tabs)), MH_OR=st.oddsratio_pooled,
                   lo=ci[0], hi=ci[1], p=tst.pvalue, breslow_day_p=bd.pvalue))
MH = pd.DataFrame(mh)
MH.to_csv(f'{OUT}/v02_mantel_haenszel.csv', index=False)
print('\nMantel-Haenszel:'); print(MH.round(4).to_string())

# ---- covariate balance: new vs experienced teams
bal = []
for c in ['dep_dec', 'res_med', 'l2_nv', 'n_entries', 'sars2', 'any_K', 'any_L', 'vhh_any', 'mouse_any', 'atyp_any', 'min_vid',
          'novel_any', 'novel_all', 'aux_any', 'capsid', 'enzyme', 'first_fam', 'lq_z_med', 'q_med', 'unobs_max', 'af_any',
          'exp_start_any', 'irm_none', 'rigid_any', 'flex_any', 'l_n_auth', 'china', 'usa', 'published', 'lab_common']:
    a = S.loc[S.new == 1, c].astype(float); b = S.loc[S.new == 0, c].astype(float)
    sd = np.sqrt((a.var() + b.var()) / 2)
    bal.append(dict(cov=c, new_mean=a.mean(), exp_mean=b.mean(), smd=(a.mean() - b.mean()) / sd if sd > 0 else np.nan))
B = pd.DataFrame(bal)
B.to_csv(f'{OUT}/v02_balance_new_vs_experienced.csv', index=False)
print('\nBalance (new team vs experienced):'); print(B.round(3).to_string())
