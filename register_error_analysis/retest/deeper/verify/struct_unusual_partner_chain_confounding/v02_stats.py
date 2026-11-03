"""Confounding checks for the 'partner_chain' finding (error-free partner chain has more clash residues and a weak
trend to more Ramachandran 'allowed').  Output tables/confounding_tests.csv, tables/strata.csv, tables/within_entry.csv
"""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_partner_chain_confounding/tables'
rng = np.random.default_rng(7)

U = pd.read_csv(f'{INV}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
D = pd.read_csv(f'{OUT}/partner_decomp.csv', dtype={'chain': str, 'partner': str})
s = pd.read_csv(f'{INV}/sample.csv', dtype={'chain': str})
E = pd.read_csv(f'{INV}/entry_vrpt.csv')
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str},
                 usecols=['pdb', 'chain', 'deposit', 'last_author', 'irm', 'fit', 'sars2', 'lq_z'])
M = U.merge(D.drop(columns=['set_id']), on=['unit', 'pdb', 'chain', 'partner'], how='left')
M = M.merge(s[['unit', 'pdb', 'sars2', 'last_author', 'deposit', 'irm', 'fit']].drop_duplicates(['unit', 'pdb']),
            on=['unit', 'pdb'], how='left')
M = M.merge(E[['pdb', 'rest_clash', 'rest_rama_allowed', 'rest_q', 'e_clashscore', 'e_Q_score', 'rest_n']], on='pdb', how='left')
M['case'] = (M.role_cc == 'case').astype(int)
M['sars2'] = M.sars2.astype(float)
M['dep'] = pd.to_datetime(M.deposit)
M['yearf'] = M.dep.dt.year + M.dep.dt.dayofyear / 366
M['period'] = np.where(M.year <= 2021, 'le2021', 'ge2022')
M['pclass'] = np.where(M.cclass == 'H', 'VL_partner', 'VH_partner')
M['resband'] = np.where(M.res < 3.5, 'lt3.5', 'ge3.5')
M['irm_rec'] = M.irm.fillna('none recorded').ne('none recorded').astype(int)
PU = M[M.partner_err_any == False].copy()
PU['qbin'] = pd.qcut(PU.p_q_V, 5, labels=False, duplicates='drop')
print('PU', PU.role_cc.value_counts().to_dict())

rows = []


def fit(y, df, rhs='C(cclass) + C(ybin) + res', group='pdb', label='', fam=''):
    cols = [y, 'case', group] + [c for c in df.columns if c in rhs.replace('C(', ' ').replace(')', ' ').replace('+', ' ').replace('*', ' ').split()]
    x = df.dropna(subset=list(dict.fromkeys(cols))).copy()
    x[y] = x[y].astype(float)
    f = f'{y} ~ case + {rhs}' if rhs else f'{y} ~ case'
    try:
        m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[group])[0]})
        rows.append(dict(family=fam, metric=y, model=label, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                         case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(),
                         effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1],
                         p=m.pvalues['case'], cluster=group, rhs=rhs))
    except Exception as e:
        rows.append(dict(family=fam, metric=y, model=label, note=str(e)[:100]))


BASE = 'C(cclass) + C(ybin) + res'
for y in ['p_clash_V', 'p_rama_allowed_V']:
    fam = 'main'
    fit(y, PU, BASE, 'pdb', 'reproduce: cclass+ybin+res, cluster entry', fam)
    fit(y, PU, BASE, 'paper', 'cluster by paper', fam)
    fit(y, PU, BASE, 'last_author', 'cluster by last author (lab)', fam)
    fit(y, PU, BASE + ' + yearf', 'paper', '+ continuous deposition date', fam)
    fit(y, PU, BASE + ' + sars2', 'paper', '+ SARS-CoV-2', fam)
    fit(y, PU, BASE + ' + p_q_V', 'paper', '+ partner Q (linear)', fam)
    fit(y, PU, BASE + ' + C(qbin)', 'paper', '+ partner Q (quintiles)', fam)
    fit(y, PU, BASE + ' + p_q_V + q_entry', 'paper', '+ partner Q + entry Q', fam)
    fit(y, PU, BASE + ' + p_unbuilt_int + p_sc_trunc_V', 'paper', '+ partner unbuilt + side-chain truncation', fam)
    rest = 'rest_clash' if y == 'p_clash_V' else 'rest_rama_allowed'
    fit(y, PU, BASE + f' + {rest}', 'paper', f'+ same metric in non-V rest of entry ({rest})', fam)
    fit(y, PU, BASE + ' + irm_rec', 'paper', '+ starting model recorded', fam)
    fit(y, PU, BASE + ' + yearf + sars2 + p_q_V + q_entry + p_unbuilt_int + p_sc_trunc_V + ' + rest, 'paper',
        'full: + date, SARS2, partner Q, entry Q, unbuilt, truncation, rest-of-entry', fam)
    fit(y, PU[PU.p_dis_res == 0], BASE, 'paper', 'partners with zero sub-threshold (1-2 res) disagreements', fam)
    fit(y, PU[PU.sars2 == 0], BASE, 'paper', 'non-SARS-CoV-2 only', fam)
    fit(y, PU[PU.year.between(2020, 2022)], BASE, 'paper', '2020-2022 only', fam)
    # one unit per paper: average cases within paper and controls within paper (papers are disjoint by design)
    agg = PU.groupby(['paper', 'case']).agg(**{y: (y, 'mean'), 'res': ('res', 'mean'), 'year': ('year', 'mean'),
                                               'p_q_V': ('p_q_V', 'mean'), 'vlp': ('cclass', lambda c: (c == 'H').mean())}).reset_index()
    agg['ybin'] = pd.cut(agg.year, [0, 2019.5, 2020.5, 2021.5, 2022.5, 2023.5, 2024.5, 2100]).astype(str)
    fit(y, agg, 'vlp + C(ybin) + res', 'paper', 'one row per paper x group (paper means)', fam)
    fit(y, agg, 'vlp + C(ybin) + res + p_q_V', 'paper', 'one row per paper x group + partner Q', fam)
    # matched-set Wilcoxon, restricted to sets where the case partner and >=1 control partner exist
    c = PU[PU.case == 1].set_index('set_id')[y]
    k = PU[PU.case == 0].groupby('set_id')[y].mean()
    dd = (c - k).dropna()
    rows.append(dict(family=fam, metric=y, model='matched-set Wilcoxon (case minus mean of own controls)', n_case=len(dd),
                     effect=dd.mean(), p=stats.wilcoxon(dd).pvalue))

# ---- clash decomposition: which counterpart drives the excess
for y in ['pc_intra', 'pc_intra_only', 'pc_unit', 'pc_unit_site', 'pc_other', 'pc_iface_any', 'pc_noniface_any',
          'pc_noniface_intra_only', 'p_clash_mag_mean']:
    fit(y, PU, BASE, 'paper', 'decomposition (cclass+ybin+res, cluster paper)', 'decomposition')
    fit(y, PU, BASE + ' + p_q_V', 'paper', 'decomposition + partner Q', 'decomposition')
# sub-threshold disagreements (does the partner carry hidden slips?)
PU['any_dis'] = (PU.p_dis_res > 0).astype(float)
fit('any_dis', PU, BASE, 'paper', 'partner has any 1-2 residue disagreement (LPM)', 'definition')
# partner Q itself (is the partner sitting in poorer density?)
fit('p_q_V', PU, BASE, 'paper', 'partner mean Q-score', 'mapfit')
fit('p_q_V', PU, BASE + ' + q_entry', 'paper', 'partner mean Q-score + entry Q', 'mapfit')

T = pd.DataFrame(rows)
T['q_bh_within_table'] = np.nan
ok = T.p.notna()
T.loc[ok, 'q_bh_within_table'] = multipletests(T.loc[ok, 'p'], method='fdr_bh')[1]
T.to_csv(f'{OUT}/confounding_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(T[['metric', 'model', 'n_case', 'n_ctrl', 'case_mean', 'ctrl_mean', 'effect', 'ci_lo', 'ci_hi', 'p', 'q_bh_within_table']].round(4).to_string())

# ---- strata
srows = []
for y in ['p_clash_V', 'p_rama_allowed_V']:
    for var in ['period', 'pclass', 'resband']:
        for lev, x in PU.groupby(var):
            rhs = 'C(ybin) + res' + ('' if var == 'pclass' else ' + C(cclass)')
            x = x.dropna(subset=[y])
            m = smf.ols(f'{y} ~ case + {rhs}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.paper)[0]})
            srows.append(dict(metric=y, stratum=f'{var}={lev}', n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                              case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(),
                              effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0],
                              ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case']))
    # interaction with period
    m = smf.ols(f'{y} ~ case*C(period) + C(cclass) + C(ybin) + res', PU).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(PU.paper)[0]})
    t = [k for k in m.params.index if k.startswith('case:')][0]
    srows.append(dict(metric=y, stratum='case x period interaction', effect=m.params[t], p=m.pvalues[t]))
S = pd.DataFrame(srows); S.to_csv(f'{OUT}/strata.csv', index=False)
print(S.round(4).to_string())

# ---- correlation of partner clash with error chain's own beyond-site clash among cases
cc = PU[PU.case == 1]
print('case: spearman(partner clash, unit clash_out)', stats.spearmanr(cc.p_clash_V, cc.clash_out, nan_policy='omit'))
kk = PU[PU.case == 0]
print('ctrl: spearman(partner clash, unit clash_out)', stats.spearmanr(kk.p_clash_V, kk.clash_out, nan_policy='omit'))
print('case vs ctrl partner Q', PU.groupby('case').p_q_V.mean().round(3).to_dict(),
      'entry Q', PU.groupby('case').q_entry.mean().round(3).to_dict(),
      'rest clash', PU.groupby('case').rest_clash.mean().round(2).to_dict())
