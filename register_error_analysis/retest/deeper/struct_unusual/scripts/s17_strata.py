"""Beyond-site geometry effects (Ramachandran 'allowed' and clash rates outside the site) stratified by chain class
(VH, VHH, VL) and by period (<=2021, >=2022). Output tables/beyond_site_strata.csv"""
import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
from common import TAB
warnings.filterwarnings('ignore')
U = pd.read_csv(f'{TAB}/unit_metrics.csv', dtype={'chain': str})
U['case'] = (U.role_cc == 'case').astype(int)
U['cls'] = U.cclass.replace({'K': 'VL', 'L': 'VL'})
rows = []
for y in ['rama_allowed_out', 'clash_out']:
    for c, x in U.groupby('cls'):
        x = x.dropna(subset=[y])
        m = smf.ols(f'{y} ~ case + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
        rows.append(dict(metric=y, stratum=c, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                         case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(), effect=m.params['case'], p=m.pvalues['case']))
    for per, x in U.assign(per=np.where(U.year <= 2021, '<=2021', '>=2022')).groupby('per'):
        x = x.dropna(subset=[y])
        m = smf.ols(f'{y} ~ case + C(cclass) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.pdb)[0]})
        rows.append(dict(metric=y, stratum=per, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                         case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(), effect=m.params['case'], p=m.pvalues['case']))
O = pd.DataFrame(rows); O.to_csv(f'{TAB}/beyond_site_strata.csv', index=False); print(O.round(4).to_string())
