# Extra checks: (a) does target class explain the paper-level antigen-family dispersion? (b) 2020-21 niche-antigen effect adjusted for lab experience
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
m=pd.read_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz').merge(pd.read_csv(f'{OUT}/tables/lab_experience.csv'),on='pdb',how='left')
m['cls2']=m.tgt_cls.replace({'viral other':'enzyme'}); m['capenz']=m.cls2.isin(['viral capsid','enzyme']).astype(int)
EM=m[m.method=='EM'].copy(); EM['fam1']=EM.tgt_fam.fillna('none').str.split('|').str[0]
P=EM.groupby('paper').agg(y=('y','max'),fam1=('fam1',lambda s:s.mode().iat[0]),capenz=('capenz','max'),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size'),sars2=('sars2','max')).reset_index(); P['ln']=np.log2(P.n)
rows=[]
for lab,D,f in [('paper level, null = year+res+size',P,'y ~ C(ycat)+res+ln'),('paper level, null = year+res+size+capsid/enzyme',P,'y ~ capenz + C(ycat)+res+ln'),
                ('paper level non-SARS2, null = year+res+size+capsid/enzyme',P[~P.sars2],'y ~ capenz + C(ycat)+res+ln')]:
    D=D.copy(); mod=smf.glm(f,D,family=sm.families.Binomial()).fit(); D['p_hat']=mod.predict(D); g=D.groupby('fam1')
    def stat(yv):
        o=pd.Series(yv,index=D.index).groupby(D.fam1).sum(); ex=g.p_hat.sum(); v=g.p_hat.apply(lambda q:(q*(1-q)).sum()); return (((o-ex)**2/v.clip(lower=1e-9))[ex>0]).sum()
    obs=stat(D.y.values); rng=np.random.default_rng(3); sims=np.array([stat(rng.binomial(1,D.p_hat.values)) for _ in range(500)]); p=(1+(sims>=obs).sum())/501
    print(f'{lab}: dispersion ratio {obs/sims.mean():.2f}, p={p:.3f}'); rows.append(dict(test=f'antigen-family dispersion, {lab}',OR=obs/sims.mean(),p=p,n=len(D),n_err=int(D.y.sum())))
x=EM[(EM.period=='2020-21')&(~EM.sars2)&EM.labs_acc_12m.notna()].copy(); x['zero']=(x.labs_acc_12m==0).astype(int)
Pp=x.groupby('paper').agg(y=('y','max'),zero=('zero','max'),capenz=('capenz','max'),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size'),exp=('lab_prior_ab','min')).reset_index()
Pp['ln']=np.log2(Pp.n); Pp['lexp']=np.log2(1+Pp.exp)
for f in ['y ~ zero + C(ycat)+res+ln','y ~ zero + capenz + lexp + C(ycat)+res+ln']:
    mod=smf.glm(f,Pp,family=sm.families.Binomial()).fit()
    ci=np.exp(mod.conf_int().loc['zero']); print(f,'zero OR',round(np.exp(mod.params['zero']),2),ci.round(2).tolist(),'p',round(mod.pvalues['zero'],4))
    rows.append(dict(test=f'2020-21 non-SARS2 papers, zero competing labs: {f}',OR=np.exp(mod.params['zero']),lo=ci.iloc[0],hi=ci.iloc[1],p=mod.pvalues['zero'],n=len(Pp),n_err=int(Pp.y.sum())))
print(Pp.groupby(['zero','capenz']).y.agg(['mean','size','sum']).round(3).to_string())
pd.DataFrame(rows).to_csv(f'{OUT}/tables/extra_checks.csv',index=False)
