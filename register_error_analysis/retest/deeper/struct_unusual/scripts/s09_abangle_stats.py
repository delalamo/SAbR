"""VH-VL orientation of case vs control Fvs against the published ABangle reference distribution
(All_Angles.dat, 1,296 Fvs from SAbDab 2013, mostly X-ray). Mahalanobis distance of each Fv from the reference.
Output tables/fv_orientation.csv"""
import numpy as np, pandas as pd
from scipy import stats
from common import C, TAB, SP

A = ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']
ref = pd.read_csv(f'{C}/abangle_src/ABangle/data/All_Angles.dat', sep='\t')
ref = ref[ref.Method == 'X-RAY']
mu = ref[A].mean().values; cov = np.cov(ref[A].values.T); ci = np.linalg.inv(cov)
sd = ref[A].std().values
fv = pd.read_csv(f'{TAB}/fv_abangle.csv', dtype={'H': str, 'L': str})
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
meta = s.drop_duplicates('pdb').set_index('pdb')[['res', 'year', 'ybin']]
fv = fv.join(meta, on='pdb')
fv = fv[(fv.cdist <= 28) & fv[[a + '_sabr' for a in A]].notna().all(1)].copy()
# exclude controls whose chains have any disagreement run (they come from clean entries, so none should)
fv = fv[fv.fv_case | (~fv.H_any & ~fv.L_any)]
for w in ['sabr', 'anarci']:
    X = fv[[a + f'_{w}' for a in A]].values
    dlt = X - mu
    fv[f'maha_{w}'] = np.sqrt(np.einsum('ij,jk,ik->i', dlt, ci, dlt))
    fv[f'maxabsz_{w}'] = np.abs(dlt / sd).max(1)
thr = np.sqrt(stats.chi2.ppf(0.99, 6))
fv['outlier99'] = fv.maha_sabr > thr
fv.to_csv(f'{TAB}/fv_orientation.csv', index=False)
print('n', fv.fv_case.value_counts().to_dict(), 'chi2(6) 99% radius', round(thr, 2))
refX = ref[A].values - mu
rm = np.sqrt(np.einsum('ij,jk,ik->i', refX, ci, refX))
print('reference X-ray maha median', np.median(rm).round(2), 'frac>thr', (rm > thr).mean().round(3))
for g, x in fv.groupby('fv_case'):
    print('case' if g else 'ctrl', 'maha median', x.maha_sabr.median().round(2), 'frac>99%', x.outlier99.mean().round(3),
          'core rmsd H/L', x.rmsH_sabr.median().round(2), x.rmsL_sabr.median().round(2))
print(fv.groupby('fv_case')[[a + '_sabr' for a in A]].agg(['mean', 'std']).round(2).T)
print('ref', dict(zip(A, mu.round(2))), dict(zip(A, sd.round(2))))
