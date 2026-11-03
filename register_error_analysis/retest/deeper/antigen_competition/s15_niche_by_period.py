# Paper-level "niche antigen" effect (no other lab deposited an antibody structure of the same antigen within +-12 months), by period, EM, non-SARS-CoV-2
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',45)
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
m=pd.read_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz').merge(pd.read_csv(f'{OUT}/tables/lab_experience.csv'),on='pdb',how='left')
m['cls2']=m.tgt_cls.replace({'viral other':'enzyme'}); m['capenz']=m.cls2.isin(['viral capsid','enzyme']).astype(int)
EM=m[(m.method=='EM')&m.labs_acc_12m.notna()&~m.sars2].copy(); EM['zero']=(EM.labs_acc_12m==0).astype(int)
rows=[]
for per in ['<=2019','2020-21','2022+']:
    x=EM[EM.period==per]
    P=x.groupby('paper').agg(y=('y','max'),zero=('zero','max'),capenz=('capenz','max'),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size'),lexp=('lab_prior_ab',lambda s:np.log2(1+s.min()))).reset_index(); P['ln']=np.log2(P.n)
    mod=smf.glm('y ~ zero + capenz + lexp + C(ycat)+res+ln' if per!='<=2019' else 'y ~ zero + capenz + lexp + res + ln',P,family=sm.families.Binomial()).fit()
    t=P.groupby('zero').y.agg(['mean','size','sum']).round(3); ci=np.exp(mod.conf_int().loc['zero'])
    print(per,'papers',len(P),'| niche papers',t.loc[1].to_dict(),'| contested',t.loc[0].to_dict(),'| adj OR',round(np.exp(mod.params['zero']),2),ci.round(2).tolist(),'p',round(mod.pvalues['zero'],4))
    rows.append(dict(test=f'niche antigen (0 other labs +-12m), paper level, non-SARS2, {per}, adj capsid/enzyme+lab experience',stratum=per,OR=np.exp(mod.params['zero']),lo=ci.iloc[0],hi=ci.iloc[1],p=mod.pvalues['zero'],n=len(P),n_err=int(P.y.sum()),
                     niche_rate=t.loc[1,'mean'],contested_rate=t.loc[0,'mean']))
pd.DataFrame(rows).to_csv(f'{OUT}/tables/niche_by_period.csv',index=False)
x=EM[(EM.period=='2020-21')&(EM.zero==1)&(EM.y==1)]
print(x.groupby('paper').agg(lab=('last_author','first'),n_err=('y','sum'),cls=('cls2','first'),antigen=('tgt_name','first'),desc=('tgt_desc','first'),dep=('deposit','min')).sort_values('dep').to_string())
