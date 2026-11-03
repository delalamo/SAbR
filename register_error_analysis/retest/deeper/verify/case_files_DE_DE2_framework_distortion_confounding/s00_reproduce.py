"""Reproduce DE2 (framework RMSD, confirmed EM D/E units vs matched / within-paper clean units) from the
investigator's cached feature table, using the same unit selection as case_files_DE/scripts/s07_stats.py."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = f'{SP}/deeper'; CACHE = f'{D}/cache/case_files_DE'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_confounding'
MODES = ['Dplus_CDR3short', 'Dminus_CDR3long', 'E_FR4bulge']


def load_sample():
    X = pd.read_pickle(f'{CACHE}/X_struct.pkl')
    ER = pd.read_pickle(f'{CACHE}/err_struct.pkl')[['pdb', 'chain', 'confirm']]
    X = X.merge(ER, on=['pdb', 'chain'], how='left')
    CH = pd.read_csv(f'{D}/chains.csv.gz')
    X = X.merge(CH[['pdb', 'chain', 'unit', 'paper', 'deposit']].rename(columns={'unit': 'u', 'paper': 'pp'}),
                on=['pdb', 'chain'], how='left')
    X = X[X.method == 'EM'].copy()
    X['err'] = (X.group == 'error').astype(int)
    X['yb'] = pd.cut(X.year, [0, 2019, 2020, 2021, 2023, 2030],
                     labels=['<=2019', '2020', '2021', '2022-23', '>=2024']).astype(str)
    X['ct'] = np.where(X.vhh == 1, 'VHH', X.ctype)
    X = X.sort_values(['group', 'pdb', 'chain']).drop_duplicates(['group', 'u'])
    return X


def sets(X, confirmed=True, modes=MODES):
    e = X[(X.group == 'error') & X['mode'].isin(modes)]
    if confirmed: e = e[e.confirm != 'not confirmed']
    mc = X[X.group == 'matched_clean']
    wp = X[(X.group == 'within_paper_clean') & X.pp.isin(e.pp)]
    return e, mc, wp


if __name__ == '__main__':
    X = load_sample()
    e, mc, wp = sets(X)
    print('errors', len(e), 'papers', e.pp.nunique(), 'matched', len(mc), 'wp', len(wp), 'wp papers', wp.pp.nunique())
    print('mean fw_rmsd err %.3f matched %.3f wp %.3f' % (e.fw_rmsd.mean(), mc.fw_rmsd.mean(), wp.fw_rmsd.mean()))
    print('median fw_rmsd err %.3f matched %.3f wp %.3f' % (e.fw_rmsd.median(), mc.fw_rmsd.median(), wp.fw_rmsd.median()))
    d = pd.concat([e, mc]).dropna(subset=['fw_rmsd']).copy()
    d['x'] = (d.fw_rmsd - d.fw_rmsd.mean()) / d.fw_rmsd.std()
    m = smf.logit('err ~ x + C(yb) + res + C(ct)', data=d).fit(disp=0, cov_type='cluster',
                                                              cov_kwds={'groups': pd.factorize(d.pp)[0]})
    print('matched OR/SD %.2f p %.2g n_err %d n_ctl %d' % (np.exp(m.params['x']), m.pvalues['x'], d.err.sum(), (d.err == 0).sum()))
    d = pd.concat([e, wp]).dropna(subset=['fw_rmsd'])
    z = d[d.pp.isin(d[d.err == 1].pp) & d.pp.isin(d[d.err == 0].pp)].copy()
    z['x'] = (z.fw_rmsd - z.fw_rmsd.mean()) / z.fw_rmsd.std()
    m = ConditionalLogit(z.err.values, z[['x']].values, groups=pd.factorize(z.pp)[0]).fit(disp=0)
    print('within-paper OR/SD %.2f p %.2g papers %d err %d ctl %d' % (np.exp(m.params[0]), m.pvalues[0], z.pp.nunique(),
                                                                       z.err.sum(), (z.err == 0).sum()))
    # error chains: confirmed vs not confirmed
    ea = X[(X.group == 'error') & X['mode'].isin(MODES)]
    print(ea.groupby('confirm').fw_rmsd.describe())
    print(ea.groupby('mode').fw_rmsd.describe())
