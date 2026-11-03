"""Residual association after removing every positive that the coordinate comparison (v05b) or prior checks show is not a
first-occurrence register error: 9bop, 9e89 (confirmed artifacts), 8c8t (HIV FR3-insertion class), 7m3l (coords = 2.4 A 7m3n),
9yfu (coords = 8tb7), 7y0o (coords = clean twin 7y0n), 8xlk (coordinate copy of 7te9).
Variants: (i) remaining 3 positives (7te9, 9dm7, 9ml5); (ii) rabbit VH removed from both arms (rabbit FR3 37 = germline norm);
(iii) only the reference-supported positive (7te9). BH over the investigator's 16 scan tests with T1 replaced.
Output tables/v06_residual.csv"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DE}/verify/case_files_ABCF_FR3_INDEL_C_integrity'
U = pd.read_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/units_em.pkl')
NOV = pd.read_csv(f'{DE}/seq_novelty/tables/chain_novelty.csv.gz', usecols=['pdb', 'chain', 'q_species_all'], low_memory=False)
C = pd.read_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/chains_feat.pkl').merge(NOV, on=['pdb', 'chain'], how='left')
rab = set(C[(C.q_species_all == 'rabbit')].unit)
U['rabbit'] = U.unit.isin(rab).astype(int)
U['yC'] = U.modes.map(lambda s: 'C_FR3' in s).astype(int)
d0 = U[(U.yC == 1) | (U.err_any == False)]
d0 = d0[d0.cclass.isin(set(d0[d0.yC == 1].cclass))]
print('rabbit units among clean FR3-atyp controls:', int(d0[(d0.yC == 0) & (d0.fr3_atyp == 1)].rabbit.sum()), '/', int(d0[d0.yC == 0].fr3_atyp.sum()),
      '; rabbit among C cases:', d0[(d0.yC == 1) & (d0.rabbit == 1)].pdb.tolist())
notgen = {'9bop', '9e89', '8c8t', '7m3l', '9yfu', '7y0o', '8xlk'}
drop_units = set(d0[(d0.yC == 1) & d0.pdb.isin(notgen)].unit)
rows = []
def t(name, d):
    a = int(d[d.yC == 1].fr3_atyp.sum()); ne = int(d.yC.sum()); c = int(d[d.yC == 0].fr3_atyp.sum()); nc = int((d.yC == 0).sum())
    o, p = fisher_exact([[a, ne - a], [c, nc - c]])
    oa = pa = np.nan
    try:
        m = smf.logit('yC ~ fr3_atyp + ybin + res + cclass', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
        oa, pa = np.exp(m.params['fr3_atyp']), m.pvalues['fr3_atyp']
    except Exception: pass
    rows.append(dict(test=name, err=f'{a}/{ne}', err_pct=round(100 * a / ne, 1), ctl=f'{c}/{nc}', ctl_pct=round(100 * c / nc, 2), OR_fisher=round(o, 1), p_fisher=p, OR_adj=oa, p_adj=pa))
t('investigator T1 (reproduced)', d0)
d1 = d0[~d0.unit.isin(drop_units)]
t('(i) all non-genuine positives removed: 7te9, 9dm7, 9ml5 remain', d1)
d2 = d1[d1.rabbit == 0]
t('(ii) + rabbit units removed from both arms', d2)
t('(iii) only the reference-supported positive (7te9) counted as exposed', d2.assign(fr3_atyp=np.where((d2.yC == 1) & (d2.pdb != '7te9'), 0, d2.fr3_atyp)))
R = pd.DataFrame(rows)
S = pd.read_csv(f'{DE}/case_files_ABCF/tables/scan_tests.csv')
for i, r in R.iterrows():
    ps = S.p.fillna(1).values.copy(); ps[0] = r.p_fisher
    R.loc[i, 'q_BH_in_investigator_family'] = multipletests(ps, method='fdr_bh')[1][0]
R.to_csv(f'{OUT}/tables/v06_residual.csv', index=False)
pd.set_option('display.width', 250); print(R.to_string())
