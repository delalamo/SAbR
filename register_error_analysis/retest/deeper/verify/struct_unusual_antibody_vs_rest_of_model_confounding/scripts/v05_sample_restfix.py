"""struct_unusual entry_vrpt.csv sets rest_* = 0 when an entry has no non-V residue (denominator max(n,1)); 10 case
entries and 0 controls are affected, so their 'gap' equals the V-domain value and their 'rest' clash/Rama are 0.
Re-run the investigator's entry-level tests with those entries dropped, clustered by entry (as published) and by paper."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_confounding/tables'
E = pd.read_csv(f'{OUT}/sample_entries_augmented.csv')
rows = []
for sub, df in [('as published (rest_n==0 kept as 0)', E), ('rest_n>0 only', E[E.rest_n > 0])]:
    for y in ['gap_rama_allowed', 'gap_clash', 'gap_q', 'rest_rama_allowed', 'rest_clash', 'rest_rota_out',
              'rest_rama_out', 'rest_q', 'v_rama_allowed']:
        for grp in ['pdb', 'paper']:
            x = df[[y, 'case', 'ybin', 'res', grp]].dropna()
            m = smf.ols(f'{y} ~ case + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[grp])[0]})
            rows.append(dict(subset=sub, y=y, cluster=grp, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                             case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(),
                             effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0],
                             ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case']))
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/sample_restfix.csv', index=False)
pd.set_option('display.width', 250)
print(R.round(4).to_string())
