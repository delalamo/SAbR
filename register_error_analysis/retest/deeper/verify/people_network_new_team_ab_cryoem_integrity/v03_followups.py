#!/usr/bin/env python3
"""Follow-ups: year heterogeneity (LR interaction), co-author-confirmed experience for collision-prone surnames,
novel-V adjustment (copying/first-occurrence overlap), country adjustment of the 2020-21 na-flag effect,
exact stratified tests and larger permutation runs."""
import sys, json, warnings, collections, pickle
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
from common import strat_exact
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
CV = f'{SP}/cache/verify_newteam'
DEEP = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DEEP}/verify/people_network_new_team_ab_cryoem_integrity'
S = pd.read_csv(f'{CV}/study_frame_verify.csv.gz', parse_dates=['dep0'])
EE = pd.read_pickle(f'{CV}/entries_exposures.pkl')
EE = EE[(EE.skey == 'study') & (EE.method == 'EM') & (EE.grp != 'amyloid')]
R = []


def fit(d, f, t, label, fam):
    m = smf.glm(f, d, family=sm.families.Binomial()).fit()
    ci = np.exp(m.conf_int().loc[t]).values
    r = dict(test=label, family=fam, OR=np.exp(m.params[t]), lo=ci[0], hi=ci[1], p=m.pvalues[t], n=len(d), n_err=int(d.y.sum()))
    R.append(r); return m


# ------------------------------------------------------------ 1. formal year heterogeneity
for x in ['nt_strict', 'nt_strict_na', 'nt_inv']:
    d = S.dropna(subset=[x])
    m0 = smf.glm(f'y ~ {x} + C(ycat) + res + l2_nv', d, family=sm.families.Binomial()).fit()
    m1 = smf.glm(f'y ~ {x} * C(ycat) + res + l2_nv', d, family=sm.families.Binomial()).fit()
    lr = 2 * (m1.llf - m0.llf); df = m1.df_model - m0.df_model
    print(f'{x}: new-team x year-category interaction LR={lr:.2f}, df={df}, p={chi2.sf(lr, df):.4g}')
    R.append(dict(test=f'heterogeneity across 7 deposit-year categories ({x}), LR test', family='heterogeneity',
                  p=chi2.sf(lr, df), n=len(d), n_err=int(d.y.sum())))
    # 3 eras: <=2021, 2022-23, >=2024
    d = d.assign(era=pd.cut(d.year, [0, 2021, 2023, 2100], labels=['<=2021', '2022-23', '>=2024']).astype(str))
    for e, g in d.groupby('era'):
        fit(g, f'y ~ {x} + C(ycat) + res + l2_nv', x, f'{x}, era {e}', 'era')

# ------------------------------------------------------------ 2. co-author-confirmed experience for collision-prone names
# re-parse entries.idx through v01 helpers
src = open(f'{OUT}/v01_rebuild.py').read().split('# ------------------------------------------------------------- antibody-entry sets')[0]
exec(src)  # defines idx (pdb -> dep, em, au), key(), parse_authors()
ab = pickle.load(open(f'{CV}/absets.pkl', 'rb'))['ab_anarci']
abem_by_key = collections.defaultdict(list)   # strict key -> list of (dep, set(other strict keys))
for p, r in idx.items():
    if r['em'] and p in ab:
        ks = {a['strict'] for a in r['au']}
        for k in ks:
            abem_by_key[k].append((r['dep'], ks))


def confirmed_exp(pdbs, d):
    """max over authors of prior abem entries; for collision-prone surnames an earlier entry counts only if it shares
    >=1 other author (strict key) with the current study."""
    au = {}
    for p in pdbs:
        for a in idx[p]['au']:
            au[a['strict']] = a['common']
    best = 0
    for k, common in au.items():
        others = set(au) - {k}
        n = sum(1 for dd, ks in abem_by_key.get(k, []) if dd < d and (not common or (ks & others)))
        best = max(best, n)
    return best


dep0 = S.set_index('study').dep0
pd_by_study = EE.groupby('study').pdb.apply(list)
S['max_abem_conf'] = [confirmed_exp(pd_by_study[s], np.datetime64(dep0[s].date(), 'D')) for s in S.study]
S['nt_conf'] = (S.max_abem_conf == 0).astype(int)
print('\nnew-team share: na', S.nt_strict_na.mean().round(3), 'all-names', S.nt_strict.mean().round(3), 'co-author-confirmed', S.nt_conf.mean().round(3))
print(pd.crosstab(S.nt_conf, S.y, margins=True))
fit(S, 'y ~ nt_conf + C(ycat) + res + l2_nv', 'nt_conf', 'co-author-confirmed experience for collision-prone names', 'definition')
d = S.assign(era=pd.cut(S.year, [0, 2021, 2023, 2100], labels=['<=2021', '2022-23', '>=2024']).astype(str))
for e, g in d.groupby('era'):
    fit(g, 'y ~ nt_conf + C(ycat) + res + l2_nv', 'nt_conf', f'co-author-confirmed, era {e}', 'era')

# ------------------------------------------------------------ 3. novel-V adjustment / restriction
nv = pd.read_csv(f'{DEEP}/pdb_meta/tables/chain_novel_v.csv.gz')
print('chain_novel_v columns', nv.columns.tolist()[:10])
e_nov = nv.groupby('pdb').novel_v.mean()
EE['nov_frac'] = EE.pdb.map(e_nov)
st_nov = EE.groupby('study').nov_frac.mean()
S['all_novel'] = (S.study.map(st_nov) == 1).astype(int)
S['any_known'] = 1 - S.all_novel
print('all-novel-V share: new teams', S[S.nt_strict == 1].all_novel.mean().round(3), 'experienced', S[S.nt_strict == 0].all_novel.mean().round(3))
for x in ['nt_strict', 'nt_strict_na']:
    fit(S, f'y ~ {x} + all_novel + C(ycat) + res + l2_nv', x, f'{x} + all V regions novel', 'novelty')
    fit(S[S.all_novel == 1], f'y ~ {x} + C(ycat) + res + l2_nv', x, f'{x}, studies with only novel V regions', 'novelty')

# ------------------------------------------------------------ 4. 2020-21 na-flag effect vs country / common names
PM = pd.read_csv(f'{DEEP}/paper_meta/paper_meta.csv', usecols=['doi', 'last_country'])
PM['doi'] = 'doi:' + PM.doi.astype(str).str.lower()
S['cn'] = S.study.map(PM.drop_duplicates('doi').set_index('doi').last_country).fillna('unknown')
S['china'] = (S.cn == 'China').astype(int); S['usa'] = (S.cn == 'USA').astype(int)
P = S[S.year.between(2020, 2021)]
print('\n2020-21: na flag by last-author common surname:', P.groupby('last_common').nt_strict_na.mean().round(3).to_dict(),
      ' error rate by last_common:', P.groupby('last_common').y.mean().round(3).to_dict(),
      ' error rate China:', P.groupby('china').y.agg(['size', 'mean']).round(3).values.tolist())
fit(P, 'y ~ nt_strict_na + C(ycat) + res + l2_nv', 'nt_strict_na', '2020-21, na flag, base', 'na_2020_21')
fit(P, 'y ~ nt_strict_na + china + usa + C(ycat) + res + l2_nv', 'nt_strict_na', '2020-21, na flag + China/USA', 'na_2020_21')
fit(P[~P.last_common.astype(bool)], 'y ~ nt_strict_na + C(ycat) + res + l2_nv', 'nt_strict_na', '2020-21, na flag, last author not collision-prone', 'na_2020_21')
fit(P, 'y ~ nt_conf + C(ycat) + res + l2_nv', 'nt_conf', '2020-21, co-author-confirmed flag', 'na_2020_21')
fit(S, 'y ~ nt_strict_na + china + usa + C(ycat) + res + l2_nv', 'nt_strict_na', 'all years, na flag + China/USA', 'definition')
fit(S, 'y ~ nt_strict + china + usa + C(ycat) + res + l2_nv', 'nt_strict', 'all years, all-name flag + China/USA', 'definition')

# ------------------------------------------------------------ 5. exact stratified tests + permutation
rng = np.random.default_rng(5)
for x in ['nt_strict', 'nt_strict_na', 'nt_conf']:
    for strata in ['ycat', 'half']:
        o, e, pu, p2, orr = strat_exact(S[x].values, S.y.values, S[strata].values)
        R.append(dict(test=f'exact stratified test ({strata}), {x}', family='exact', OR=orr, p=p2, n=len(S), n_err=int(S.y.sum()),
                      observed=o, expected=e))
        print(f'exact stratified {x} by {strata}: obs {o} exp {e:.1f} MH OR {orr:.2f} p_two {p2:.4g}')
    # permutation of exposure within ycat x resolution-bin strata, statistic = MH OR (log)
    obs = np.log(strat_exact(S[x].values, S.y.values, S.ycat.values)[4])
    grp = (S.ycat + S.res_bin).values
    null = []
    xv = S[x].values.copy()
    for _ in range(4000):
        xp = xv.copy()
        for gname in np.unique(grp):
            ix = np.where(grp == gname)[0]
            xp[ix] = rng.permutation(xp[ix])
        y = S.y.values; num = 0; den = 0
        for s in np.unique(S.ycat.values):
            ix = S.ycat.values == s
            a = ((xp == 1) & (y == 1) & ix).sum(); b = ((xp == 1) & (y == 0) & ix).sum()
            c = ((xp == 0) & (y == 1) & ix).sum(); dd = ((xp == 0) & (y == 0) & ix).sum(); N = ix.sum()
            num += a * dd / N; den += b * c / N
        null.append(np.log(num / den) if den > 0 and num > 0 else 0)
    null = np.array(null)
    pp = (1 + (np.abs(null) >= abs(obs)).sum()) / (len(null) + 1)
    print(f'permutation within year x resolution bins, {x}: MH OR {np.exp(obs):.2f}, p={pp:.4f}')
    R.append(dict(test=f'permutation within year x resolution bins (4000), {x}', family='exact', OR=np.exp(obs), p=pp,
                  n=len(S), n_err=int(S.y.sum())))

# ------------------------------------------------------------ 6. dose response / audit-author count
S['l_abem'] = np.log2(1 + S.max_abem_strict)
fit(S, 'y ~ l_abem + C(ycat) + res + l2_nv', 'l_abem', 'per doubling of most-experienced author prior antibody cryo-EM entries (all names)', 'dose')
fit(S[S.nt_strict == 0], 'y ~ l_abem + C(ycat) + res + l2_nv', 'l_abem', 'per doubling, experienced teams only (all names)', 'dose')
S['authbin'] = pd.cut(S.n_auth, [0, 2, 4, 8, 1000]).astype(str)
print('\nnew-team share and error rate by audit-author count:')
print(S.groupby('authbin').agg(n=('y', 'size'), new_all=('nt_strict', 'mean'), new_na=('nt_strict_na', 'mean'), err=('y', 'mean')).round(3))
for ab_, g in S.groupby('authbin'):
    if g.y.sum() >= 10:
        fit(g, 'y ~ nt_strict + C(ycat) + res + l2_nv', 'nt_strict', f'all-name flag within audit-author bin {ab_}', 'authors')

R = pd.DataFrame(R)
R['q_within_file'] = multipletests(R.p.fillna(1), method='fdr_bh')[1]
R.to_csv(f'{OUT}/verify_followups.csv', index=False)
print(R.round(4).to_string())
S.to_csv(f'{CV}/study_frame_verify2.csv.gz', index=False)
