"""Population-wide re-test of the V-minus-rest geometry gap: every cryo-EM entry in entries.csv with a validation
report (193 first-occurrence error entries vs ~2,570 clean entries, plus 64 entries with copied/artifact-only runs).
Unit = entry; SEs clustered by paper (citation key). Covariates: deposition-year bin (or continuous date),
resolution, SARS-CoV-2, composition, local map fit, unbuilt V residues. V side recomputed with exclusion of
SAbR/ANARCI-disagreeing residues; rest split into constant-domain residues on V chains (C) and other chains (O)."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CC = f'{SP}/deeper/cache/verify_su_gap'
SU = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_confounding/tables'
T = pd.read_pickle(f'{CC}/tallies.pkl'); I = pd.read_pickle(f'{CC}/vchain_info.pkl'); H = pd.read_pickle(f'{CC}/entry_hdr.pkl')
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
d = d[d.method == 'EM']
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
ent = ent[ent.method == 'EM'].set_index('pdb')
s = pd.read_csv(f'{SU}/sample.csv', dtype={'chain': str})
role = s.groupby('pdb').role_cc.first()

# ---------- per-entry metrics
def agg(mask):
    g = T[mask].groupby('pdb')[['n', 'allowed', 'rama_out', 'clash', 'rota_out', 'q_sum', 'n_q']].sum()
    return pd.DataFrame({'n': g.n, 'A': 100 * g.allowed / g.n, 'Ro': 100 * g.rama_out / g.n,
                         'Cl': 100 * g.clash / g.n, 'Rt': 100 * g.rota_out / g.n, 'Q': g.q_sum / g.n_q.replace(0, np.nan)})


parts = {'V': T.cat == 'V', 'Vex': (T.cat == 'V') & T.dbin.isin(['d11p', 'clean']),
         'Vex3': (T.cat == 'V') & T.dbin.isin(['d4_10', 'd11p', 'clean']),
         'Vclean': (T.cat == 'V') & (T.dbin == 'clean'), 'C': T.cat == 'C', 'O': T.cat == 'O',
         'rest': T.cat.isin(['C', 'O'])}
E = pd.DataFrame(index=sorted(T.pdb.unique()))
for k, m in parts.items():
    a = agg(m)
    for c in a.columns:
        E[f'{k}_{c}'] = a[c]
E = E.join(H.set_index('pdb'))
for k in ['V', 'Vex', 'Vex3', 'Vclean', 'C', 'O']:
    for m in ['A', 'Cl', 'Q', 'Rt']:
        E[f'gap_{k}_{m}'] = E[f'{k}_{m}'] - E[f'rest_{m}'] if k != 'C' and k != 'O' else E[f'{k}_{m}'] - E[f'O_{m}']
# V minus O only (ignoring constant domains)
for m in ['A', 'Cl', 'Q']:
    E[f'gapVO_{m}'] = E[f'V_{m}'] - E[f'O_{m}']
info = I.groupby('pdb').agg(nVch=('chain', 'size'), unb=('unbuilt_int', 'mean'), ndis=('n_dis', 'sum'),
                            maxrun=('max_run', 'max'))
E = E.join(info)
dm = d.groupby('pdb').agg(paper=('paper', 'first'), last_author=('last_author', 'first'),
                          vhh_frac=('vhh', 'mean'), aux_frac=('role', lambda x: (x == 'Auxiliary').mean()), kl_frac=('ctype', lambda x: (x != 'H').mean()))
E = E.join(dm).join(ent[['res', 'year', 'deposit', 'err_first', 'err_any', 'sars2']])
E = E[E.err_first.notna()]
E['err_first'] = E.err_first.astype(int); E['err_any'] = E.err_any.astype(int); E['sars2'] = E.sars2.astype(int)
E['grp'] = np.where(E.err_first == 1, 'error', np.where(E.err_any == 1, 'copied_or_artifact_only', 'clean'))
E['ybin'] = pd.cut(E.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100],
                   labels=['le2019', 'y2020', 'y2021', 'y2022', 'y2023', 'y2024', 'ge2025']).astype(str)
E['tdep'] = (pd.to_datetime(E.deposit) - pd.Timestamp('2020-01-01')).dt.days / 365.25
E['vfrac'] = E.V_n / (E.V_n + E.rest_n)
E['log_rest_n'] = np.log10(E.rest_n.clip(lower=1))
E['has_C'] = (E.C_n.fillna(0) > 20).astype(int)
E['role_su'] = E.index.map(role).fillna('not in sample')
pap_err = E.groupby('paper').err_first.max()
E['paper_has_err'] = E.paper.map(pap_err)
E['sibling'] = ((E.err_first == 0) & (E.paper_has_err == 1)).astype(int)
E.index.name = 'pdb'
E.to_csv(f'{OUT}/population_entries.csv')
print(E.grp.value_counts().to_dict(), 'rest_n missing', E.rest_n.isna().sum())

rows = []


def fit(name, y, df, covs, case='case', group='paper', note=''):
    cols = [y, case, group] + [c.replace('C(', '').replace(')', '') for c in covs]
    x = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    covs = [c for c in covs if x[c.replace('C(', '').replace(')', '')].nunique() > 1]
    f = f'{y} ~ {case}' + ''.join(' + ' + c for c in covs)
    m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[group])[0]})
    rows.append(dict(analysis=name, y=y, covariates=' + '.join(covs), cluster=group, n_case=int(x[case].sum()),
                     n_ctrl=int((1 - x[case]).sum()), n_clusters=x[group].nunique(),
                     case_mean=x.loc[x[case] == 1, y].mean(), ctrl_mean=x.loc[x[case] == 0, y].mean(),
                     case_median=x.loc[x[case] == 1, y].median(), ctrl_median=x.loc[x[case] == 0, y].median(),
                     effect=m.params[case], ci_lo=m.conf_int().loc[case, 0], ci_hi=m.conf_int().loc[case, 1],
                     p=m.pvalues[case], note=note))


base = ['C(ybin)', 'res']
P = E[E.grp.isin(['error', 'clean'])].copy(); P['case'] = P.err_first
full = base + ['tdep', 'sars2', 'aux_frac', 'vhh_frac', 'kl_frac', 'vfrac', 'log_rest_n', 'has_C', 'gap_V_Q', 'V_Q',
               'e_atom_inclusion_backbone', 'unb']
for y in ['gap_V_A', 'gap_V_Cl']:
    fit('population: error vs all clean EM', y, P, base)
    fit('population: + deposition date, SARS-CoV-2', y, P, base + ['tdep', 'sars2'])
    fit('population: + composition', y, P, base + ['sars2', 'vhh_frac', 'kl_frac', 'vfrac', 'log_rest_n', 'has_C'])
    fit('population: + local map fit (V-rest Q gap, V Q, atom inclusion)', y, P, base + ['gap_V_Q', 'V_Q', 'e_atom_inclusion_backbone'])
    fit('population: + unbuilt internal V residues', y, P, base + ['unb'])
    fit('population: + all covariates', y, P, full)
    fit('population: 2020-2022', y, P[P.year.between(2020, 2022)], base)
    fit('population: 2023+', y, P[P.year >= 2023], base)
    fit('population: <=2019', y, P[P.year <= 2019], ['res'])
    fit('population: non-SARS-CoV-2', y, P[P.sars2 == 0], base)
    fit('population: SARS-CoV-2 only', y, P[P.sars2 == 1], base)
    fit('population: res <=3.5', y, P[P.res <= 3.5], base)
    fit('population: res >3.5', y, P[P.res > 3.5], base)
    fit('population: no VHH in entry', y, P[P.vhh_frac == 0], base)
    fit('population: entries not in struct_unusual sample', y, P[P.role_su == 'not in sample'], base)
    # paper-level unit: mean over entries per paper x group
    PP = P.groupby(['paper', 'case']).agg(**{y: (y, 'mean'), 'res': ('res', 'mean'), 'year': ('year', 'median')}).reset_index()
    PP['ybin'] = pd.cut(PP.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100]).astype(str)
    fit('population: unit = paper (mean of entries)', y, PP, base, group='paper')
# V side with the error site excluded; error-free V chains only; rest split
for y, lab in [('gap_Vex3_A', 'V excluding residues within 3 of any SAbR/ANARCI disagreement'),
               ('gap_Vex_A', 'V excluding residues within 10 of any disagreement'),
               ('gap_Vclean_A', 'V chains without any disagreeing residue only'),
               ('gap_Vex_Cl', 'clash, V excluding within 10'), ('gap_Vclean_Cl', 'clash, disagreement-free V chains only'),
               ('gapVO_A', 'V minus other chains only (constant domains dropped)'),
               ('gapVO_Cl', 'clash, V minus other chains only'),
               ('gap_C_A', 'constant-domain (non-V on antibody chains) minus other chains, allowed'),
               ('gap_C_Cl', 'constant-domain minus other chains, clash'),
               ('V_A', 'V % allowed (absolute)'), ('Vex_A', 'V % allowed excluding within 10 (absolute)'),
               ('O_A', 'other chains % allowed (absolute)'), ('C_A', 'constant domains % allowed (absolute)'),
               ('O_Cl', 'other chains % clash'), ('C_Cl', 'constant domains % clash'), ('O_Q', 'other chains Q'),
               ('O_Rt', 'other chains rotamer outliers'), ('O_Ro', 'other chains Rama outliers'),
               ('gap_V_Q', 'V minus rest Q'), ('V_Q', 'V Q (absolute)'),
               ('e_clashscore', 'entry clashscore')]:
    fit(f'population: {lab}', y, P, base)
    fit(f'population + all covariates: {lab}', y, P, [c for c in full if c not in (y,)] if y not in ('gap_V_Q', 'V_Q') else base + ['tdep', 'sars2', 'vhh_frac', 'kl_frac', 'vfrac', 'log_rest_n', 'has_C', 'unb'])
# other comparison groups
Q = E[E.grp.isin(['copied_or_artifact_only', 'clean'])].copy(); Q['case'] = (Q.grp != 'clean').astype(int)
for y in ['gap_V_A', 'gap_Vex_A', 'gap_V_Cl']:
    fit('copied/artifact-only entries vs clean', y, Q, base)
S = E[(E.grp == 'clean')].copy(); S['case'] = S.sibling
for y in ['gap_V_A', 'gap_V_Cl']:
    fit('clean sibling entries of error papers vs other clean', y, S, base)
# the investigator's controls vs other clean entries (selection check)
C2 = E[E.grp == 'clean'].copy(); C2['case'] = (C2.role_su == 'control').astype(int)
for y in ['gap_V_A', 'gap_V_Cl', 'V_A', 'O_A']:
    fit("struct_unusual controls vs other clean entries", y, C2, base)

R = pd.DataFrame(rows)
R['q_BH'] = multipletests(R.p, method='fdr_bh')[1]
R.to_csv(f'{OUT}/population_tests.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_rows', 300)
print(R[['analysis', 'y', 'n_case', 'n_ctrl', 'n_clusters', 'case_mean', 'ctrl_mean', 'effect', 'ci_lo', 'ci_hi', 'p', 'q_BH']].round(4).to_string())
