"""Last confounding checks: refinement software flags / EM fitting protocol (clash rates depend strongly on whether the
refinement used hydrogens or clash-aware tools such as ISOLDE/Rosetta), and redundancy with the error chain's own
beyond-site clash rate.  Output tables/software_adjust.csv"""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_partner_chain_confounding/tables'
U = pd.read_csv(f'{INV}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
U['case'] = (U.role_cc == 'case').astype(int)
PU = U[U.partner_err_any == False].copy()
sw = ['sw_ModelAngelo', 'sw_ISOLDE', 'sw_Rosetta', 'sw_Coot', 'sw_Phenix', 'sw_Namdinator', 'sw_REFMAC', 'sw_BUSTER']
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'fit', 'irm'] + sw)
e = ch.groupby('pdb').agg({**{k: 'max' for k in sw}, 'fit': 'first', 'irm': 'first'}).reset_index()
PU = PU.merge(e, on='pdb', how='left')
PU['fitc'] = PU.fit.fillna('not recorded').where(PU.fit.fillna('not recorded').map(PU.fit.fillna('not recorded').value_counts()) >= 15, 'rare')
PU['irmc'] = PU.irm.fillna('none recorded').where(PU.irm.fillna('none recorded').map(PU.irm.fillna('none recorded').value_counts()) >= 15, 'rare')
BASE = 'C(cclass) + C(ybin) + res'
rows = []
for y in ['p_clash_V', 'p_rama_allowed_V']:
    for lab, rhs in [('base', BASE), ('+ software flags', BASE + ' + ' + ' + '.join(sw)),
                     ('+ software + fitting protocol + starting-model type', BASE + ' + ' + ' + '.join(sw) + ' + C(fitc) + C(irmc)'),
                     ('+ unit chain beyond-site clash (redundancy check)', BASE + ' + clash_out'),
                     ('+ unit chain beyond-site Rama allowed (redundancy check)', BASE + ' + rama_allowed_out')]:
        x = PU.dropna(subset=[y, 'clash_out', 'rama_allowed_out'])
        m = smf.ols(f'{y} ~ case + {rhs}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.paper)[0]})
        rows.append(dict(metric=y, model=lab, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()), effect=m.params['case'],
                         ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case']))
print(PU.groupby('case')[sw].mean().round(2).T)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/software_adjust.csv', index=False)
pd.set_option('display.width', 250); print(T.round(4).to_string())
