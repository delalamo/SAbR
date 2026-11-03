"""Re-analysis of the struct_unusual within-entry V-minus-rest geometry gap on the investigator's own sample
(159 case / 185 control EM entries), using their entry_vrpt.csv and residues.pkl.
Checks: reproduction; clustering by paper / last author; one entry per paper; extra covariates;
removing residues at/near SAbR-ANARCI disagreements; error-free V chains only."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SU = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_confounding/tables'
s = pd.read_csv(f'{SU}/sample.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
E = pd.read_csv(f'{SU}/entry_vrpt.csv')
emeta = s.groupby('pdb').agg(res=('res', 'first'), ybin=('ybin', 'first'), role_cc=('role_cc', 'first'),
                             sars2=('sars2', 'max'))
E = E.join(emeta, on='pdb')
E['case'] = (E.role_cc == 'case').astype(int)
dm = d.groupby('pdb').agg(paper=('paper', 'first'), last_author=('last_author', 'first'), year=('year', 'first'),
                          deposit=('deposit', 'first'), nV=('chain', 'size'), vhh_frac=('vhh', 'mean'),
                          kl_frac=('ctype', lambda x: (x != 'H').mean()), lq_z=('lq_z', 'mean'),
                          irm=('irm', 'first'), fit=('fit', 'first'))
E = E.join(dm, on='pdb')
E['dep'] = pd.to_datetime(E.deposit)
E['tdep'] = (E.dep - pd.Timestamp('2020-01-01')).dt.days / 365.25
E['vfrac'] = E.v_n / (E.v_n + E.rest_n)
E['log_rest_n'] = np.log10(E.rest_n.clip(lower=1))
E['sars2'] = E.sars2.astype(int)

# ---- residue-level recomputation of the V side with exclusions
R = pd.read_pickle(f'{SP}/deeper/cache/struct_unusual/residues.pkl')
R = R[R.rama.notna() | R.rota.notna()].copy()
R['dis'] = R.apos.notna() & ((R.apos != R.spos) | (R.ains.fillna('') != R.sins.fillna('')))
R = R.sort_values(['pdb', 'chain', 'idx'])


def mindist(g):
    i = g.idx.values; di = i[g.dis.values]
    if len(di) == 0:
        return pd.Series(np.full(len(i), 999), index=g.index)
    return pd.Series(np.abs(i[:, None] - di[None, :]).min(1), index=g.index)


R['dist'] = R.groupby(['pdb', 'chain'], group_keys=False).apply(mindist)
R['allowed'] = (R.rama == 'Allowed').astype(float)
R['clash'] = (R.nclash.fillna(0) > 0).astype(float)
chflag = d.set_index(['pdb', 'chain'])[['err_any', 'err_first']]
R = R.join(chflag, on=['pdb', 'chain'])
R['chain_has_dis'] = R.groupby(['pdb', 'chain']).dis.transform('any')

variants = {'all': R.index == R.index,
            'ex3': R.dist > 3,
            'ex10': R.dist > 10,
            'ex20': R.dist > 20,
            'nodischain': ~R.chain_has_dis,
            'erranyzero': R.err_any == 0}
VLAB = {'all': 'all V residues (reproduce)', 'ex3': 'exclude residues within 3 of any SAbR/ANARCI disagreement',
        'ex10': 'exclude within 10', 'ex20': 'exclude within 20',
        'nodischain': 'only V chains with no disagreeing residue', 'erranyzero': 'only V chains with err_any==0'}
for name, mask in variants.items():
    g = R[mask].groupby('pdb').agg(n=('allowed', 'size'), a=('allowed', 'mean'), c=('clash', 'mean'))
    E[f'vA_{name}'] = E.pdb.map(100 * g.a) - E.rest_rama_allowed
    E[f'vC_{name}'] = E.pdb.map(100 * g.c) - E.rest_clash
    E[f'vN_{name}'] = E.pdb.map(g.n)

# fraction of each case entry's V residues that are disagreeing
fr = R.groupby('pdb').agg(frac_dis=('dis', 'mean'), frac_near3=('dist', lambda x: (x <= 3).mean()))
E = E.join(fr, on='pdb')
# V unbuilt internal residues per V chain (from residues prev_si_gap)
R0 = pd.read_pickle(f'{SP}/deeper/cache/struct_unusual/residues.pkl')
ub = R0.groupby('pdb').prev_si_gap.apply(lambda x: x.fillna(0).sum()) / R0.groupby('pdb').chain.nunique()
E['unbuilt_per_vchain'] = E.pdb.map(ub)
qv = R0.groupby('pdb').q.mean()
E['vq_local'] = E.pdb.map(qv)

rows = []


def fit(name, y, df, covs, group, note=''):
    x = df[[y, 'case', group] + [c.replace('C(', '').replace(')', '') for c in covs]].dropna()
    f = f'{y} ~ case' + ''.join(' + ' + c for c in covs)
    m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[group])[0]})
    rows.append(dict(analysis=name, y=y, covariates=' + '.join(covs), cluster=group, n_case=int(x.case.sum()),
                     n_ctrl=int((1 - x.case).sum()), n_clusters=x[group].nunique(),
                     case_mean=x.loc[x.case == 1, y].mean(), ctrl_mean=x.loc[x.case == 0, y].mean(),
                     effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1],
                     p=m.pvalues['case'], note=note))


base = ['C(ybin)', 'res']
for y in ['gap_rama_allowed', 'gap_clash']:
    fit('reproduce (entry clusters)', y, E, base, 'pdb')
    fit('cluster by paper', y, E, base, 'paper')
    fit('cluster by last author', y, E, base, 'last_author')
    fit('+ continuous deposition date', y, E, base + ['tdep'], 'paper')
    fit('+ SARS-CoV-2', y, E, base + ['sars2'], 'paper')
    fit('+ composition (nV, VHH frac, light frac, V fraction, log rest size)', y, E,
        base + ['sars2', 'nV', 'vhh_frac', 'kl_frac', 'vfrac', 'log_rest_n'], 'paper')
    fit('+ rest-of-model baseline (rest_rama_allowed / rest_clash)', y, E,
        base + ['rest_rama_allowed' if 'rama' in y else 'rest_clash'], 'paper')
    fit('+ local map: V Q gap, V Q, entry atom inclusion', y, E, base + ['gap_q', 'vq_local', 'e_atom_inclusion_backbone'], 'paper')
    fit('+ unbuilt internal V residues per V chain', y, E, base + ['unbuilt_per_vchain'], 'paper')
    fit('+ all of the above', y, E, base + ['tdep', 'sars2', 'nV', 'vhh_frac', 'kl_frac', 'vfrac', 'log_rest_n',
                                            'gap_q', 'vq_local', 'e_atom_inclusion_backbone', 'unbuilt_per_vchain'], 'paper')
    fit('2020-2022 only', y, E[E.year.between(2020, 2022)], base, 'paper')
    fit('2023+ only', y, E[E.year >= 2023], base, 'paper')
    fit('<=2019 only', y, E[E.year <= 2019], ['res'], 'paper')
    fit('non-SARS-CoV-2 only', y, E[E.sars2 == 0], base, 'paper')
    fit('res <= 3.5 only', y, E[E.res <= 3.5], base, 'paper')
    fit('res > 3.5 only', y, E[E.res > 3.5], base, 'paper')
    # one entry per paper (first deposited)
    E1 = E.sort_values(['paper', 'deposit', 'pdb']).groupby(['paper', 'case']).head(1)
    fit('one entry per paper x role', y, E1, base, 'paper')
for name in variants:
    for k, lab in [('vA_', 'rama allowed'), ('vC_', 'clash')]:
        fit(f'V side recomputed: {VLAB[name]}', k + name, E, base, 'paper', lab)

T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/sample_reanalysis.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(T[['analysis', 'y', 'n_case', 'n_ctrl', 'n_clusters', 'case_mean', 'ctrl_mean', 'effect', 'ci_lo', 'ci_hi', 'p']].round(4).to_string())
print(E.groupby('case')[['frac_dis', 'frac_near3', 'unbuilt_per_vchain', 'vfrac', 'nV', 'vhh_frac', 'rest_n', 'v_n']].mean().round(3))
for name in variants:
    print(name, E.groupby('case')[f'vN_{name}'].describe()[['count', 'mean']].to_dict())
E.to_csv(f'{OUT}/sample_entries_augmented.csv', index=False)
