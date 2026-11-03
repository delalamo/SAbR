#!/usr/bin/env python3
"""Heterogeneity, within-lab, lab-resampling, IPW and by-year checks for the new-team association."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/people_network_new_team_ab_cryoem_confounding/tables'
S = pd.read_csv(f'{SP}/cache/verify_people_newteam_conf/study_frame.csv.gz')
S['lq_z_med'] = S.lq_z_med.fillna(S.lq_z_med.median()); S['q_med'] = S.q_med.fillna(S.q_med.median())
S['tier'] = S.tier.fillna('Unpublished')
S['late'] = (S.dep_year >= 2024).astype(int)
S['china'] = S.china.astype(int)
S['lab_common'] = S.lab_common.astype(bool).astype(int)
BASE = 'C(ycat) + res_med + l2_nv'
rng = np.random.default_rng(11)
out = []


def glm(d, f, cluster='lab_key'):
    return smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cluster])[0]})


# ---- 1. interactions
for mod in ['late', 'vhh_any', 'china', 'sars2', 'novel_all', 'mouse_any', 'lab_common']:
    m = glm(S, f'y ~ new * {mod} + {BASE}')
    t = f'new:{mod}'
    out.append(dict(test=f'interaction new x {mod}', OR=np.exp(m.params[t]), lo=np.exp(m.conf_int().loc[t][0]),
                    hi=np.exp(m.conf_int().loc[t][1]), p=m.pvalues[t], n=len(S)))
    for v in [0, 1]:
        d = S[S[mod] == v]
        mm = glm(d, f'y ~ new + {BASE}')
        out.append(dict(test=f'  stratum {mod}={v}', OR=np.exp(mm.params['new']), lo=np.exp(mm.conf_int().loc['new'][0]),
                        hi=np.exp(mm.conf_int().loc['new'][1]), p=mm.pvalues['new'], n=len(d), n_err=int(d.y.sum()),
                        rate_new=d[d.new == 1].y.mean(), rate_old=d[d.new == 0].y.mean()))
# three-way strata era x nanobody
for (late, vhh), d in S.groupby(['late', 'vhh_any']):
    yc = 'C(ycat)' if d.ycat.nunique() > 1 else '1'
    mm = glm(d, f'y ~ new + {yc} + res_med + l2_nv')
    out.append(dict(test=f'  stratum late={late}, vhh_any={vhh}', OR=np.exp(mm.params['new']), p=mm.pvalues['new'], n=len(d),
                    n_err=int(d.y.sum()), rate_new=d[d.new == 1].y.mean(), rate_old=d[d.new == 0].y.mean()))
for (late, ch), d in S.groupby(['late', 'china']):
    yc = 'C(ycat)' if d.ycat.nunique() > 1 else '1'
    mm = glm(d, f'y ~ new + {yc} + res_med + l2_nv')
    out.append(dict(test=f'  stratum late={late}, china={ch}', OR=np.exp(mm.params['new']), p=mm.pvalues['new'], n=len(d),
                    n_err=int(d.y.sum()), rate_new=d[d.new == 1].y.mean(), rate_old=d[d.new == 0].y.mean()))
# non-China, non-collision-prone last author, all-name-key flag (cleanest name measurement)
for lab, d, term in [('non-China & last-author name not collision-prone, na flag', S[(S.china == 0) & (S.lab_common == 0)], 'new'),
                     ('non-China & last-author name not collision-prone, all-key flag', S[(S.china == 0) & (S.lab_common == 0)], 'new_all'),
                     ('China, all-key flag', S[S.china == 1], 'new_all'),
                     ('non-China, all-key flag', S[S.china == 0], 'new_all')]:
    mm = glm(d, f'y ~ {term} + {BASE}')
    out.append(dict(test=lab, OR=np.exp(mm.params[term]), lo=np.exp(mm.conf_int().loc[term][0]), hi=np.exp(mm.conf_int().loc[term][1]),
                    p=mm.pvalues[term], n=len(d), n_err=int(d.y.sum()), rate_new=d[d[term] == 1].y.mean(), rate_old=d[d[term] == 0].y.mean()))
H = pd.DataFrame(out)
H.to_csv(f'{OUT}/v03_heterogeneity.csv', index=False)
pd.set_option('display.width', 250)
print(H.round(4).to_string())

# ---- 2. by deposit year (crude), leave-one-year-out
by = S.groupby(['ycat', 'new']).y.agg(['size', 'sum', 'mean']).unstack('new')
print('\nby year category:'); print(by.round(3).to_string())
loyo = []
for yc in sorted(S.ycat.unique()):
    d = S[S.ycat != yc]
    mm = glm(d, f'y ~ new + {BASE}')
    loyo.append(dict(left_out=yc, n=len(d), OR=np.exp(mm.params['new']), p=mm.pvalues['new']))
L = pd.DataFrame(loyo); L.to_csv(f'{OUT}/v03_leave_one_year_out.csv', index=False)
print('\nleave-one-year-category-out:'); print(L.round(4).to_string())

# ---- 3. one study per lab (random), 500 draws; labs with collision-prone keys kept as is
ors, ps = [], []
for _ in range(500):
    d = S.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('lab_key')
    try:
        m = smf.glm(f'y ~ new + {BASE}', d, family=sm.families.Binomial()).fit()
        ors.append(np.exp(m.params['new'])); ps.append(m.pvalues['new'])
    except Exception:
        pass
ors, ps = np.array(ors), np.array(ps)
res_lab = dict(test='one study per last-author lab (500 random draws)', n_per_draw=int(S.lab_key.nunique()),
               OR_median=np.median(ors), OR_2_5=np.percentile(ors, 2.5), OR_97_5=np.percentile(ors, 97.5),
               p_median=np.median(ps), frac_p_lt_05=(ps < 0.05).mean())
print('\n', res_lab)

# ---- 4. within-lab conditional logit (non-collision-prone lab key; labs with outcome variation)
W = S[S.lab_common == 0].copy()
W['ylate'] = (W.dep_year >= 2022).astype(int)
W['dep_c'] = W.dep_dec - 2022
g = W.groupby('lab_key')
Wd = W[g.y.transform('nunique').eq(2)].copy()
Wd['n_lab'] = Wd.groupby('lab_key').study.transform('size')
cl = []
for f_terms in [['new'], ['new', 'dep_c'], ['new', 'ylate'], ['new', 'dep_c', 'res_med', 'l2_nv']]:
    try:
        m = ConditionalLogit(Wd.y.values, Wd[f_terms].astype(float).values, groups=Wd.lab_key.values).fit(method='bfgs', maxiter=2000, disp=0)
        cl.append(dict(test='within-lab conditional logit: ' + '+'.join(f_terms), n_studies=len(Wd), n_labs=Wd.lab_key.nunique(),
                       n_labs_discordant_exposure=int(Wd.groupby('lab_key').new.nunique().eq(2).sum()),
                       n_err=int(Wd.y.sum()), OR=np.exp(m.params[0]), lo=np.exp(m.conf_int()[0][0]), hi=np.exp(m.conf_int()[0][1]), p=m.pvalues[0]))
    except Exception as ex:
        cl.append(dict(test='within-lab ' + '+'.join(f_terms), err=str(ex)[:80]))
print('\nwithin-lab informative labs:', Wd.lab_key.nunique(), 'studies', len(Wd))
print(Wd.groupby('new').y.agg(['size', 'sum', 'mean']))
D2 = Wd[Wd.groupby('lab_key').new.transform('nunique').eq(2)]
print('labs with both new and experienced studies:', D2.lab_key.nunique(), D2.groupby('new').y.agg(['size', 'sum', 'mean']).to_dict('index'))
print('their year distribution by new:', D2.groupby('new').dep_year.describe()[['mean', '50%']].to_dict('index'))
CLd = pd.DataFrame(cl); print(CLd.round(4).to_string())

# ---- 5. IPW (propensity of being a new team from full covariates)
FULLps = ('bs(dep_dec, df=6) + bs(res_med, df=4) + l2_nv + l2_ne + any_K + any_L + vhh_any + sars2 + mouse_any + atyp_any + min_vid + '
          'novel_any + aux_any + capsid + enzyme + first_fam + lq_z_med + q_med + unobs_max + af_any + exp_start_any + irm_none + '
          'rigid_any + flex_any + l_n_auth + C(ctry3) + C(tier)')
ps_m = smf.glm(f'new ~ {FULLps}', S, family=sm.families.Binomial()).fit()
S['ps'] = ps_m.predict(S).clip(0.02, 0.98)
S['w'] = np.where(S.new == 1, 1 / S.ps, 1 / (1 - S.ps))
S['w'] = S.w / S.w.mean()
mi = smf.glm('y ~ new', S, family=sm.families.Binomial(), freq_weights=None, var_weights=S.w).fit(
    cov_type='cluster', cov_kwds={'groups': pd.factorize(S.lab_key)[0]})
rw = S.groupby('new').apply(lambda d: np.average(d.y, weights=d.w))
ipw = dict(test='IPW (propensity on full covariates incl. country, tier, map fit, antigen class)', OR=np.exp(mi.params['new']),
           lo=np.exp(mi.conf_int().loc['new'][0]), hi=np.exp(mi.conf_int().loc['new'][1]), p=mi.pvalues['new'],
           wrate_new=rw[1], wrate_old=rw[0], ps_auc=None)
from scipy.stats import mannwhitneyu
roc_auc_score = lambda yb, sc: mannwhitneyu(sc[yb == 1], sc[yb == 0]).statistic / ((yb == 1).sum() * (yb == 0).sum())
ipw['ps_auc'] = roc_auc_score(S.new.values, S.ps.values)
print('\n', ipw)
pd.DataFrame([res_lab, ipw] + cl).to_csv(f'{OUT}/v03_lab_resampling_withinlab_ipw.csv', index=False)
