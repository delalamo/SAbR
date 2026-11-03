"""Entry-level V vs non-V metrics, several masks; compare with investigator's entry_vrpt.csv."""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUTC = f'{SP}/deeper/cache/verify_struct_unusual_rest'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_integrity'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
R = pd.read_pickle(f'{OUTC}/res.pkl')
R = R.drop_duplicates(['pdb', 'chain', 'resnum', 'ic', 'altcode'])
R['allowed'] = (R.rama == 'Allowed').astype(float)
R['nonfav'] = R.rama.isin(['Allowed', 'OUTLIER']).astype(float)
R['out'] = (R.rama == 'OUTLIER').astype(float)
R['rotout'] = (R.rota == 'OUTLIER').astype(float)
R['clash'] = R.clash.astype(float)


def agg(df, tag):
    g = df.groupby('pdb')
    return pd.DataFrame({f'{tag}_n': g.size(), f'{tag}_allowed': 100 * g.allowed.mean(), f'{tag}_nonfav': 100 * g.nonfav.mean(),
                         f'{tag}_clash': 100 * g.clash.mean(), f'{tag}_rotout': 100 * g.rotout.mean(),
                         f'{tag}_out': 100 * g.out.mean(), f'{tag}_q': g.q.mean()})


rest = R[~R.isV]
parts = [agg(R[R.isV], 'V'), agg(R[R.isV & ~R.m3], 'Vm3'), agg(R[R.isV & ~R.m10], 'Vm10'),
         agg(R[R.isV & ~R.chain_has_run], 'Vclean'), agg(R[R.isV & R.chain_has_run & ~R.m3], 'Vrunch_m3'),
         agg(rest, 'rest'), agg(rest[rest.is_vchain], 'restC'), agg(rest[~rest.is_vchain], 'restO')]
E = pd.concat(parts, axis=1)
S = pd.read_csv(f'{INV}/sample.csv', dtype={'chain': str})
meta = S.groupby('pdb').agg(role=('role_cc', 'first'), nroles=('role_cc', 'nunique'), res=('res', 'first'), ybin=('ybin', 'first'),
                            year=('year', 'first'), paper=('paper', 'first'), last_author=('last_author', 'first'),
                            sars2=('sars2', 'max'))
assert (meta.nroles == 1).all()
E = E.join(meta, how='right')
E['case'] = (E.role == 'case').astype(int)
# cross-check case/control definitions against entries.csv
ent = pd.read_csv(f'{SP}/deeper/entries.csv').set_index('pdb')
E = E.join(ent[['method', 'err_first', 'err_any', 'n_V_chains']].add_prefix('ent_'))
print('case entries: method', E[E.case == 1].ent_method.value_counts().to_dict(), 'err_first', E[E.case == 1].ent_err_first.value_counts().to_dict())
print('ctrl entries: method', E[E.case == 0].ent_method.value_counts().to_dict(), 'err_any', E[E.case == 0].ent_err_any.value_counts().to_dict())
# compare with investigator
I = pd.read_csv(f'{INV}/entry_vrpt.csv').set_index('pdb')
for a, b in [('V_allowed', 'v_rama_allowed'), ('rest_allowed', 'rest_rama_allowed'), ('V_clash', 'v_clash'), ('rest_clash', 'rest_clash'),
             ('V_n', 'v_n'), ('rest_n', 'rest_n')]:
    x = E[a] - I.loc[E.index, b]
    print(f'{a:14s} vs investigator {b:18s}: max|diff| {np.nanmax(np.abs(x)):.4g}  n differing>1e-6 {(np.abs(x) > 1e-6).sum()}')
for t in ['V', 'Vm3', 'Vm10', 'Vclean', 'Vrunch_m3']:
    for k in ['allowed', 'nonfav', 'clash', 'rotout', 'q']:
        E[f'gap_{t}_{k}'] = E[f'{t}_{k}'] - E[f'rest_{k}']
E['frac_V_inrun_m3'] = 1 - E.Vm3_n / E.V_n
E.to_csv(f'{OUT}/entry_metrics_verify.csv')
cols = ['V_n', 'rest_n', 'frac_V_inrun_m3', 'V_allowed', 'Vm3_allowed', 'Vm10_allowed', 'Vclean_allowed', 'Vrunch_m3_allowed',
        'rest_allowed', 'restC_allowed', 'restO_allowed', 'V_clash', 'Vm3_clash', 'Vclean_clash', 'rest_clash', 'restC_clash', 'restO_clash',
        'gap_V_allowed', 'gap_Vm3_allowed', 'gap_Vm10_allowed', 'gap_Vclean_allowed', 'gap_V_clash', 'gap_Vm3_clash', 'gap_Vclean_clash']
pd.set_option('display.width', 200)
print(E.groupby('case')[cols].mean().T.round(3))
print(E.groupby('case')[['Vclean_n', 'Vrunch_m3_n']].apply(lambda x: x.notna().sum()))
print('ctrl entries with any run residues in V (should be 0):', (E[E.case == 0].frac_V_inrun_m3 > 0).sum())
