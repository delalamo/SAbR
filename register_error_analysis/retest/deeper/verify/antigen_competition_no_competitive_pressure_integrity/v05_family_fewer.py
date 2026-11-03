# Is the "fewer errors on crowded antigen families (non-SARS2, q=0.014)" result driven by the class-A GPCR 30% mega-family
# and fiducial-only entries? Also X-ray SARS2 rate.
import warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
INV='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
F=pd.read_csv(f'{INV}/tables/focal_entries_antigen_features.csv.gz'); F['sars2']=F.sars2.astype(bool)
EM=F[(F.method=='EM')&F.ab_status2.isin(['bound','fiducial-only'])&F.labs_acc_12m.notna()].copy()
EM['fam1']=EM.tgt_fam.fillna('none').str.split('|').str[0]
top=EM[~EM.sars2].groupby('fam1').agg(n=('y','size'),err=('y','sum'),labs_med=('labs_fam_12m','median'),cls=('tgt_cls',lambda s:s.mode().iat[0])).sort_values('labs_med',ascending=False)
print('non-SARS2 EM families with highest median other labs (fam, +-12m):'); print(top[top.n>=10].head(10).to_string())
def fit(D,term,extra='C(ycat)+res'):
    mod=smf.glm(f'y ~ {term} + {extra}',D,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(D.paper)[0]})
    ci=mod.conf_int().loc[term]; return 'OR %.3f (%.3f-%.3f) p=%.4f n=%d err=%d'%(np.exp(mod.params[term]),np.exp(ci[0]),np.exp(ci[1]),mod.pvalues[term],len(D),D.y.sum())
N=EM[~EM.sars2]
print('non-SARS2 labs_fam_12m, as investigator     :',fit(N,'l_labs_fam_12m'))
print('  excluding fiducial-only entries           :',fit(N[N.ab_status2=='bound'],'l_labs_fam_12m'))
print('  excluding GPCR targets                    :',fit(N[N.tgt_cls!='GPCR'],'l_labs_fam_12m'))
print('  excluding GPCR + fiducial-only            :',fit(N[(N.tgt_cls!='GPCR')&(N.ab_status2=='bound')],'l_labs_fam_12m'))
print('  excl GPCR+fid, also excl capsid/enzyme    :',fit(N[(N.tgt_cls!='GPCR')&(N.ab_status2=='bound')&~N.tgt_cls.isin(['viral capsid','enzyme','viral other'])],'l_labs_fam_12m'))
print('  accession-level, excl GPCR+fid            :',fit(N[(N.tgt_cls!='GPCR')&(N.ab_status2=='bound')],'l_labs_acc_12m'))
X=F[F.method=='X-ray'].copy(); X['s2']=X.sars2.astype(int)
a=int(X[X.sars2].y.sum()); b=int(X.sars2.sum())-a; c=int(X[~X.sars2].y.sum()); d=int((~X.sars2).sum())-c
print('X-ray SARS2 %d/%d vs non-SARS2 %d/%d, Fisher p=%.3f'%(a,a+b,c,c+d,stats.fisher_exact([[a,b],[c,d]])[1]))
print('X-ray SARS2 adjusted period+res:',fit(X,'s2','C(period)+res'))
