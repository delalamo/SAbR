# Follow-ups: (1) why capsid/enzyme targets carry errors (map quality, fitting, labs); (2) competition adjusted for target class;
# (3) paper-level class rates; (4) leave-one-lab-out robustness
import warnings, collections
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
m=pd.read_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz')
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','lq','lq_z','irm','fit','vhh','role','aux_reagent','v_id','species','L1_atyp','L2_atyp','L3_atyp'])
agg=ch.groupby('pdb').agg(lq=('lq','mean'),lq_z=('lq_z','mean'),irm=('irm','first'),fit=('fit','first'),vhh=('vhh','max'),mouse=('species',lambda s:(s=='mouse').mean()),
    atyp=('L3_atyp','max'),v_id=('v_id','mean')).reset_index()
m=m.merge(agg,on='pdb',how='left')
m['cap']=(m.tgt_cls=='viral capsid').astype(int); m['enz']=m.tgt_cls.isin(['enzyme','viral other']).astype(int)
m['cls2']=m.tgt_cls.replace({'viral other':'enzyme'})
EM=m[m.method=='EM'].copy()
TESTS=[]
def fit(d,f,label,stratum,terms):
    mod=smf.glm(f,d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
    out=[]
    for t in terms:
        k=[x for x in mod.params.index if x.startswith(t)][0]
        ci=mod.conf_int().loc[k]
        out.append(dict(test=label,stratum=stratum,term=k,OR=np.exp(mod.params[k]),lo=np.exp(ci[0]),hi=np.exp(ci[1]),p=mod.pvalues[k],n=len(d),n_err=int(d.y.sum())))
    return out
print('== 1. Capsid and enzyme targets: map quality, resolution, fitting, antibody type (EM)')
for nm,col in [('capsid','cap'),('enzyme (incl. viral enzymes)','enz')]:
    a=EM[EM[col]==1]; b=EM[EM[col]==0]
    print(f'\n{nm}: n={len(a)} vs {len(b)}')
    for v in ['res','lq','lq_z','vhh','mouse','atyp','v_id']:
        x=a[v].dropna(); y=b[v].dropna()
        p=stats.mannwhitneyu(x,y).pvalue if len(x)>3 else np.nan
        print(f'   {v}: {x.mean():.3f} vs {y.mean():.3f} (median {x.median():.3f} vs {y.median():.3f}) p={p:.3g}')
    print('   fit protocol:',a.fit.value_counts(normalize=True).round(2).to_dict(),'| others',b.fit.value_counts(normalize=True).round(2).to_dict())
    print('   starting model:',a.irm.value_counts(normalize=True).round(2).to_dict(),'| others',b.irm.value_counts(normalize=True).round(2).to_dict())
d=EM.dropna(subset=['lq_z','v_id','mouse']).copy()
print('\nEM entries with local map fit:',len(d),'errors',d.y.sum())
for f,lab in [('y ~ cap + enz + C(ycat) + res','class only'),('y ~ cap + enz + lq_z + C(ycat) + res','+ local map fit (lq_z)'),
              ('y ~ cap + enz + lq_z + vhh + C(ycat) + res + C(fit)','+ lq_z + nanobody + fit protocol'),
              ('y ~ cap + enz + lq_z + vhh + mouse + v_id + C(ycat) + res + C(fit)','+ lq_z + nanobody + fit + mouse + germline identity')]:
    r=fit(d,f,f'capsid/enzyme target, {lab}','EM',['cap','enz']); TESTS+=r
    for x in r: print(f'  {lab:40s} {x["term"]}: OR {x["OR"]:.2f} ({x["lo"]:.2f}-{x["hi"]:.2f}) p={x["p"]:.2g}')
# lq_z among capsid entries: errors vs clean
for col in ['cap','enz']:
    a=d[d[col]==1]
    print(f'  within {col}: lq_z errors {a[a.y==1].lq_z.mean():.2f} vs clean {a[a.y==0].lq_z.mean():.2f} (n {int(a.y.sum())}/{len(a)}) p={stats.mannwhitneyu(a[a.y==1].lq_z,a[a.y==0].lq_z).pvalue:.3g}')
print('\n== 2. Paper-level class rates (EM papers; error paper = any first-occurrence error)')
P=EM.groupby('paper').agg(y=('y','max'),cls=('cls2',lambda s:s.mode().iat[0]),lab=('last_author','first'),n=('y','size'),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),sars2=('sars2','max')).reset_index()
P['ln']=np.log2(P.n)
tab=P.groupby('cls').agg(papers=('y','size'),err_papers=('y','sum'),labs=('lab','nunique')).assign(rate=lambda t:(t.err_papers/t.papers).round(3)).sort_values('papers',ascending=False)
print(tab.to_string())
for cl in ['viral capsid','enzyme','channel/transporter','GPCR','viral surface glycoprotein']:
    P['iscls']=(P.cls==cl).astype(int)
    r=fit(P,'y ~ iscls + C(ycat) + res + ln',f'paper-level class: {cl}','EM papers',['iscls']); TESTS+=r
    print(f'  {cl:28s} OR {r[0]["OR"]:.2f} ({r[0]["lo"]:.2f}-{r[0]["hi"]:.2f}) p={r[0]["p"]:.2g}')
print('\n== 3. Leave-one-lab-out (EM entries): capsid and enzyme ORs')
for col in ['cap','enz']:
    labs=EM[(EM[col]==1)&(EM.y==1)].last_author.value_counts()
    ors=[]
    for L in labs.index:
        dd=EM[EM.last_author!=L]
        r=fit(dd,f'y ~ {col} + C(ycat) + res',f'LOLO {col}','EM',[col])[0]; ors.append((L,round(r['OR'],2),round(r['p'],4)))
    print(col,'error labs:',labs.to_dict()); print('   drop-one ORs:',ors)
    dd=EM[~EM.last_author.isin(labs.index[:2])]
    r=fit(dd,f'y ~ {col} + C(ycat) + res',f'{col} target, two largest error labs removed','EM',[col])[0]; TESTS.append(r)
    print(f'   both top-2 labs removed: OR {r["OR"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f}) p={r["p"]:.2g}')
print('\n== 4. Competition adjusted for target class (EM, entries with a target)')
D=EM[EM.ab_status2.isin(['bound','fiducial-only'])&EM.labs_acc_12m.notna()].copy()
for t in ['l_labs_acc_12m','l_labs_fam_12m','first_acc','first_fam','l_prev12_fam']:
    for lab,f in [('year+res',f'y ~ {t} + C(ycat) + res'),('year+res+class',f'y ~ {t} + C(cls2) + C(ycat) + res'),('year+res+class, non-SARS2',None)]:
        dd=D if f else D[~D.sars2]
        f=f or f'y ~ {t} + C(cls2) + C(ycat) + res'
        dd=dd[dd.cls2.map(dd.cls2.value_counts())>=15]
        r=fit(dd,f,f'competition {t} adjusted: {lab}','EM',[t])[0]; TESTS.append(r)
        print(f'  {t:16s} {lab:28s} OR {r["OR"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f}) p={r["p"]:.2g} n={r["n"]}')
# competition among papers in the 2020-21 non-SARS-CoV-2 window, where the binned effect looked largest
print('\n  2020-21 non-SARS2, zero other labs (+-12m) vs >=1, by paper:')
x=D[(D.period=='2020-21')&(~D.sars2)].copy(); x['zero']=(x.labs_acc_12m==0).astype(int)
Pp=x.groupby('paper').agg(y=('y','max'),zero=('zero','max'),cls=('cls2',lambda s:s.mode().iat[0]),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size')).reset_index(); Pp['ln']=np.log2(Pp.n)
print(Pp.groupby('zero').y.agg(['mean','size','sum']).round(3).to_string())
r=fit(Pp,'y ~ zero + C(ycat) + res + ln','2020-21 non-SARS2 papers: zero competing labs','EM papers 2020-21 non-SARS2',['zero'])[0]; TESTS.append(r); print('  paper-level OR',round(r['OR'],2),round(r['lo'],2),round(r['hi'],2),'p',round(r['p'],4))
r=fit(Pp[~Pp.cls.isin(['viral capsid','enzyme'])],'y ~ zero + C(ycat) + res + ln','2020-21 non-SARS2 papers: zero competing labs, capsid/enzyme excluded','EM papers 2020-21 non-SARS2 excl capsid/enzyme',['zero'])[0]; TESTS.append(r); print('  excl capsid/enzyme OR',round(r['OR'],2),round(r['lo'],2),round(r['hi'],2),'p',round(r['p'],4),'n',r['n'],r['n_err'])
pd.DataFrame(TESTS).to_csv(f'{OUT}/tables/followup_tests.csv',index=False)
