#!/usr/bin/env python3
"""Era dependence (leave-era-out, antigen-class adjustment within 2022-23) and X-ray replication."""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import fisher_exact
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
CV = f'{SP}/cache/verify_newteam'
DEEP = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DEEP}/verify/people_network_new_team_ab_cryoem_integrity'
S = pd.read_csv(f'{CV}/study_frame_verify2.csv.gz', parse_dates=['dep0'])
EE = pd.read_pickle(f'{CV}/entries_exposures.pkl')
R = []


def fit(d, f, t, label):
    try:
        m = smf.glm(f, d, family=sm.families.Binomial()).fit()
        ci = np.exp(m.conf_int().loc[t]).values
        R.append(dict(test=label, OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t], n=len(d), n_err=int(d.y.sum())))
    except Exception as ex:
        R.append(dict(test=label, n=len(d), n_err=int(d.y.sum()), note=str(ex)[:50]))


S['era'] = pd.cut(S.year, [0, 2021, 2023, 2100], labels=['<=2021', '2022-23', '>=2024']).astype(str)
for x in ['nt_inv', 'nt_strict_na', 'nt_strict', 'nt_conf']:
    d = S.dropna(subset=[x])
    fit(d[d.era != '2022-23'], f'y ~ {x} + C(ycat) + res + l2_nv', x, f'{x}: all years EXCEPT 2022-23')
    fit(d[d.year != 2023], f'y ~ {x} + C(ycat) + res + l2_nv', x, f'{x}: all years except 2023')
    for e, g in d.groupby('era'):
        t = pd.crosstab(g[x], g.y).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
        R.append(dict(test=f'{x}: era {e}, raw counts new err/n vs exp err/n', OR=fisher_exact(t)[0], p=fisher_exact(t)[1],
                      n=len(g), n_err=int(g.y.sum()), counts=f'{t[0,0]}/{t[0].sum()} vs {t[1,0]}/{t[1].sum()}'))

# antigen-class / reagent adjustment within 2022-23 (investigator's covariates)
F = pd.read_csv(f'{DEEP}/antigen_competition/tables/focal_entries_antigen_features.csv.gz', usecols=['pdb', 'tgt_cls', 'first_fam'])
CH = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'vhh', 'species', 'sars2'])
ch = CH.groupby('pdb').agg(vhh=('vhh', 'max'), mouse=('species', lambda s: (s == 'mouse').any()))
E = EE[(EE.skey == 'study') & (EE.method == 'EM')].merge(F, on='pdb', how='left').merge(ch, left_on='pdb', right_index=True, how='left')
st = E.groupby('study').agg(capsid=('tgt_cls', lambda s: (s == 'viral capsid').any()),
                            enzyme=('tgt_cls', lambda s: s.isin(['enzyme', 'viral other']).any()),
                            vhh=('vhh', 'max'), mouse=('mouse', 'max'), first_fam=('first_fam', 'max'), sars2=('sars2', 'max'))
S = S.merge(st.astype(float).fillna(0), left_on='study', right_index=True, how='left')
adj = 'C(ycat) + res + l2_nv + capsid + enzyme + vhh + mouse + sars2 + first_fam'
for x in ['nt_strict', 'nt_conf', 'nt_strict_na']:
    fit(S, f'y ~ {x} + {adj}', x, f'{x}: all years, + antigen class, nanobody, mouse, SARS-CoV-2, first-of-family')
    fit(S[S.era == '2022-23'], f'y ~ {x} + {adj}', x, f'{x}: 2022-23, + antigen class etc.')
    fit(S[S.era == '<=2021'], f'y ~ {x} + {adj}', x, f'{x}: <=2021, + antigen class etc.')
    fit(S[S.era == '2022-23'], f'y ~ {x} + {adj} + np.log2(1+n_auth)', x, f'{x}: 2022-23, + antigen class + author count')
print('2022-23 new-team (all names) studies, capsid', S[(S.era == '2022-23') & (S.nt_strict == 1)].capsid.sum(),
      'vhh', S[(S.era == '2022-23') & (S.nt_strict == 1)].vhh.sum())

# ---------------------------------------------------------------- X-ray replication with the rebuild
X = EE[(EE.skey == 'study') & (EE.method == 'X-ray') & EE.grp.isin(['err', 'clean', 'excl'])].copy()
g = X.groupby('study')
SX = pd.DataFrame({'y': g.grp.apply(lambda s: int((s == 'err').any())), 'keep': g.grp.apply(lambda s: (s == 'err').any() or (s == 'clean').all()),
                   'res': g.res.median(), 'n_v': g.n_v.sum(), 'year': g.dep0.min().dt.year,
                   'max_ab_strict': g.max_ab_strict.max(), 'max_ab_strict_na': g.max_ab_strict_na.max()})
SX = SX[SX.keep]
SX['per'] = pd.cut(SX.year, [0, 2009, 2017, 2100]).astype(str); SX['l2_nv'] = np.log2(SX.n_v)
for c in ['max_ab_strict', 'max_ab_strict_na']:
    SX['nt'] = (SX[c] == 0).astype(int)
    fit(SX, 'y ~ nt + C(per) + res + l2_nv', 'nt', f'X-ray: no author with prior antibody PDB entry ({c})')
    print(c, SX.groupby('nt').y.agg(['size', 'sum', 'mean']).round(3).values.tolist())
R = pd.DataFrame(R)
R.to_csv(f'{OUT}/verify_era_xray.csv', index=False)
print(R.round(4).to_string())
