# Is the niche-antigen / capsid / enzyme effect explained by the antibody itself being new (no earlier PDB entry of the same V region)?
# novel_v per chain is read (read-only) from the pdb_meta lens: ../pdb_meta/tables/chain_novel_v.csv.gz
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
nv=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/chain_novel_v.csv.gz')
e=nv.groupby('pdb').novel_v.mean().rename('novel_v_frac').reset_index()
m=pd.read_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz').merge(e,on='pdb',how='left').merge(pd.read_csv(f'{OUT}/tables/lab_experience.csv'),on='pdb',how='left')
m['all_novel']=(m.novel_v_frac==1).astype(int); m['lexp']=np.log2(1+m.lab_prior_ab)
m['cls2']=m.tgt_cls.replace({'viral other':'enzyme'}); m['cap']=(m.cls2=='viral capsid').astype(int); m['enz']=(m.cls2=='enzyme').astype(int)
EM=m[m.method=='EM'].copy()
print('EM share of entries whose antibodies are all new V regions, by class:')
print(EM.groupby('cls2').agg(n=('y','size'),all_novel=('all_novel','mean'),err=('y','mean')).round(3).sort_values('n',ascending=False).head(9).to_string())
print('EM error rate by all-novel:',EM.groupby('all_novel').y.agg(['mean','size','sum']).round(3).to_dict())
rows=[]
D=EM[EM.labs_acc_12m.notna()].copy()
for f in ['y ~ all_novel + C(ycat)+res','y ~ cap + enz + first_fam + all_novel + lexp + C(ycat)+res']:
    mod=smf.glm(f,D,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(D.paper)[0]})
    out={k:(round(np.exp(mod.params[k]),2),round(mod.pvalues[k],4)) for k in ['all_novel','cap','enz','first_fam','lexp'] if k in mod.params}
    print(f,out)
    for k,(o,p) in out.items(): rows.append(dict(test=f'novel-V model [{f}]: {k}',OR=o,p=p,n=len(D),n_err=int(D.y.sum())))
x=D[(D.period=='2020-21')&(~D.sars2)].copy(); x['zero']=(x.labs_acc_12m==0).astype(int)
P=x.groupby('paper').agg(y=('y','max'),zero=('zero','max'),nov=('all_novel','max'),cap=('cap','max'),enz=('enz','max'),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size'),lexp=('lexp','min')).reset_index(); P['ln']=np.log2(P.n); P['capenz']=P[['cap','enz']].max(axis=1)
print('2020-21 non-SARS2 papers: novel-V share by zero competitors:',P.groupby('zero').nov.mean().round(3).to_dict())
mod=smf.glm('y ~ zero + nov + capenz + lexp + C(ycat)+res+ln',P,family=sm.families.Binomial()).fit()
print('zero competing labs, + novel V + capsid/enzyme + experience: OR',round(np.exp(mod.params['zero']),2),np.exp(mod.conf_int().loc['zero']).round(2).tolist(),'p',round(mod.pvalues['zero'],4),'| novel V OR',round(np.exp(mod.params['nov']),2),'p',round(mod.pvalues['nov'],3))
rows.append(dict(test='2020-21 non-SARS2 papers, zero competing labs, + novel V + capsid/enzyme + lab experience',OR=np.exp(mod.params['zero']),p=mod.pvalues['zero'],n=len(P),n_err=int(P.y.sum())))
pd.DataFrame(rows).to_csv(f'{OUT}/tables/novel_v_checks.csv',index=False)
