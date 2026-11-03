# Definition sensitivity: is it "enzymes" or the lens's residual rule-order "enzyme" class?
# Broad host-enzyme definition = lens enzyme class OR (target accession has an EC number, non-viral target, not a GPCR construct
# whose EC comes from a fusion partner). Compares enzyme-class vs EC-enzymes classed elsewhere (mostly membrane/receptor enzymes).
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_confounding'
EM=pd.read_csv(f'{OUT}/em_entries_verify.csv.gz')
EM['ec_else']=((EM.enz_ec==1)&(EM.enz==0)&(EM.tgt_org!='viral')&(EM.tgt_cls!='GPCR')).astype(int)
EM['enz_broad']=(EM.enz_host|EM.ec_else).astype(int)
rows=[]
def fit(d,term,extra,label,unit='entry'):
    a=int(((d[term]==1)&(d.y==1)).sum()); n1=int((d[term]==1).sum()); c=int(((d[term]==0)&(d.y==1)).sum()); n0=int((d[term]==0).sum())
    if a<3:
        p=stats.fisher_exact([[a,n1-a],[c,n0-c]])[1]; OR=((a+.5)*(n0-c+.5))/((n1-a+.5)*(c+.5)); lo=hi=np.nan; mdl='Fisher'
    else:
        mod=smf.glm(f'y ~ {term} + {extra}',d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
        OR=np.exp(mod.params[term]); lo,hi=np.exp(mod.conf_int().loc[term]); p=mod.pvalues[term]; mdl=extra
    rows.append(dict(test=label,unit=unit,rate_in=f'{a}/{n1} ({100*a/max(n1,1):.1f}%)',rate_out=f'{c}/{n0} ({100*c/max(n0,1):.1f}%)',OR=OR,lo=lo,hi=hi,p=p,model=mdl))
    print(f'  {label:85s} {a}/{n1} ({100*a/max(n1,1):.1f}%) vs {c}/{n0} ({100*c/max(n0,1):.1f}%)  OR {OR:.2f} ({lo:.2f}-{hi:.2f}) p={p:.2g}')
print('EC-enzymes outside the enzyme class (non-viral, non-GPCR):',int(EM.ec_else.sum()),'entries; classes',EM[EM.ec_else==1].tgt_cls.value_counts().to_dict())
fit(EM,'enz_broad','C(ycat) + res','D1 broad host enzyme (class OR EC, non-viral, non-GPCR) vs other EM')
fit(EM,'ec_else','C(ycat) + res','D2 EC-enzymes classed as transporter/receptor/toxin etc. vs other EM')
H=EM[EM.enz_broad==1].copy(); H['in_cls']=H.enz_host
fit(H,'in_cls','C(period) + res','D3 within broad host enzymes: lens enzyme class vs EC-enzymes classed elsewhere')
# paper level
P=EM.groupby('paper').agg(y=('y','max'),enz_broad=('enz_broad','max'),ec_else=('ec_else','max'),enz_host=('enz_host','max'),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size')).reset_index(); P['ln']=np.log2(P.n)
fit(P,'enz_broad','C(ycat) + res + ln','D4 paper level: broad host enzyme',unit='paper')
fit(P,'ec_else','C(ycat) + res + ln','D5 paper level: EC-enzymes classed elsewhere',unit='paper')
pd.DataFrame(rows).to_csv(f'{OUT}/tests_enzyme_definition.csv',index=False)
