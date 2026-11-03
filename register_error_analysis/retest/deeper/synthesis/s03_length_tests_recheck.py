"""Re-check literature_dean's specific-length tests (unverified) without amyloid units and without the s01 strict artifact units.
Same model as literature_dean s05 (logit err ~ exposure + C(ybin) + res; cryo-EM units; copied-only units dropped)."""
import pandas as pd, statsmodels.formula.api as smf, numpy as np
D='/home/user/SAbR/register_error_analysis/retest/deeper'
u=pd.read_csv(f'{D}/literature_dean/tables/units_literature.csv.gz')
L=pd.read_csv(f'{D}/synthesis/artifact_ledger_runs.csv')
strict={'amyloid_fibril','indel_gap_artifact_confirmed','DE_chain_end_artifact','near_copy','not_a_register_shift'}
g=L.groupby('unit').flag.agg(lambda s:set(s)<=strict); rm=set(g[g].index)
amy=set(L[L.flag=='amyloid_fibril'].unit)
E=u[(u.all_em)&(~u.copied_only)]
tests={'heavy CDR2 IMGT7 [H2-9] vs 8':(E[(E.ctype=='H')&E.rL2.isin([7,8])],lambda d:(d.rL2==7).astype(int)),
       'lambda CDR1 IMGT8 [L1-13] vs 9':(E[(E.ctype=='L')&E.rL1.isin([8,9])],lambda d:(d.rL1==8).astype(int)),
       'kappa CDR1 IMGT12 [L1-17] vs 6':(E[(E.ctype=='K')&E.rL1.isin([6,12])],lambda d:(d.rL1==12).astype(int))}
rows=[]
for t,(d0,f) in tests.items():
    for lab,d in [('as published',d0),('minus amyloid',d0[~d0.unit.isin(amy)]),('minus strict artifacts',d0[~d0.unit.isin(rm)])]:
        d=d.assign(x=f(d),err=d.err.astype(int))
        m=smf.logit('err~x+C(ybin)+res',data=d).fit(disp=0)
        rows.append(dict(test=t,set=lab,err_exposed=f'{d[d.x==1].err.sum()}/{(d.x==1).sum()}',err_ref=f'{d[d.x==0].err.sum()}/{(d.x==0).sum()}',
                         OR=round(float(np.exp(m.params['x'])),2),p=float(m.pvalues['x'])))
R=pd.DataFrame(rows); print(R.to_string()); R.to_csv(f'{D}/synthesis/length_tests_recheck.csv',index=False)
