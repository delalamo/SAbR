"""Re-check the literature_dean 'canonical pull' (unverified by the adversarial lenses) after removing amyloid fibrils
and the consolidated detector-artifact / copy flags from s01. Uses literature_dean's own per-case table and null."""
import pandas as pd, numpy as np
D='/home/user/SAbR/register_error_analysis/retest/deeper'
A=pd.read_csv(f'{D}/literature_dean/tables/apparent_length_error_units.csv')
L=pd.read_csv(f'{D}/synthesis/artifact_ledger_runs.csv')
strict={'amyloid_fibril','indel_gap_artifact_confirmed','DE_chain_end_artifact','near_copy','not_a_register_shift'}
loose=strict|{'indel_gap_artifact_probable','DE_not_confirmed_other'}
REG={1:(27,38),2:(56,65),3:(105,117)}
def flagged(row,S):
    g=L[(L.pdb==row.pdb)&(L.chain==row.chain)]
    a,b=REG[int(row.cdr)]
    g=g[(g.imgt0<=b+2)&(g.imgt1>=a-2)]
    return len(g)>0 and set(g.flag)<=S
amy=set(L[L.flag=='amyloid_fibril'].pdb)
rng=np.random.default_rng(1)
rows=[]
for lab,sub in [('as published',A),('minus amyloid',A[~A.pdb.isin(amy)]),
                ('minus strict artifacts',A[[not flagged(r,strict) for r in A.itertuples()]]),
                ('minus loose artifacts',A[[not flagged(r,loose) for r in A.itertuples()]])]:
    for sl,s in [('all',sub),('EM',sub[sub.method=='EM'])]:
        o=int(s.more_common.sum()); e=s.p_exp.sum()
        sims=rng.binomial(1,np.tile(s.p_exp.values,(50000,1))).sum(1)
        p=max(min(1,2*min((sims>=o).mean(),(sims<=o).mean())),2e-5)
        rows.append(dict(set=lab,subset=sl,n=len(s),obs_more_common=o,exp=round(e,1),p=p,n_amyloid=int(s.pdb.isin(amy).sum())))
R=pd.DataFrame(rows); print(R.to_string()); R.to_csv(f'{D}/synthesis/canonical_pull_recheck.csv',index=False)
