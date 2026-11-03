# Error vs clean entries: binding status, antigen class/organism, specific antigens, competition; antibody role (unit level)
import json, re, collections, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
A=pd.read_csv(f'{OUT}/tables/pdbwide_ab_entries_antigen.csv.gz')
e=pd.read_csv(f'{SP}/deeper/entries.csv')
amy=set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb)
m=e.merge(A.drop(columns=['date']),on='pdb',how='left')
m['amyloid']=m.pdb.isin(amy)
m=m[~m.amyloid & (m.err_first | ~m.err_any)].copy()
m['y']=m.err_first.astype(int)
m['dep']=pd.to_datetime(m.deposit); m['dyear']=m.dep.dt.year
m['ycat']=m.dyear.clip(lower=2017).astype(str)
m['period']=pd.cut(m.dyear,[0,2019,2021,2100],labels=['<=2019','2020-21','2022+']).astype(str)
m['paper']=m.doi.fillna(m.last_author.fillna('')+'_'+m.dyear.astype(str))
m['sars2']=m.sars2.astype(bool)
for c in [c for c in m.columns if c.startswith(('labs_','ents_','prev12'))]: m['l_'+c]=np.log2(1+m[c])
m['race_1m']=(m.labs_acc_1m>0).astype(float).where(m.labs_acc_1m.notna())
m['ab_status2']=m.ab_status.replace({'no contact':'other','no interface data':'other'})
m.to_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz',index=False)
print('analysis entries',len(m),'errors',m.y.sum(),'| EM',(m.method=='EM').sum(),m[m.method=='EM'].y.sum())
TESTS=[]
def logit(df,term,form_extra='C(ycat) + res',label='',stratum=''):
    d=df.dropna(subset=[term] if term in df.columns else [])
    if term in d.columns and set(d[term].dropna().unique())<={0,1,0.0,1.0}:
        a=int(((d[term]==1)&(d.y==1)).sum()); b=int(((d[term]==1)&(d.y==0)).sum()); c_=int(((d[term]==0)&(d.y==1)).sum()); dd=int(((d[term]==0)&(d.y==0)).sum())
        if min(a,c_)<3:   # too sparse for a clustered logit: Fisher exact, Haldane-corrected OR, unadjusted
            p=stats.fisher_exact([[a,b],[c_,dd]])[1]; orr=((a+.5)*(dd+.5))/((b+.5)*(c_+.5))
            return [dict(test=label or term,stratum=stratum,term=term+' (Fisher, unadjusted)',OR=orr,lo=np.nan,hi=np.nan,p=p,n=len(d),n_err=int(d.y.sum()))]
    try:
        mod=smf.glm(f'y ~ {term} + {form_extra}',d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
    except Exception as ex:
        return None
    rows=[]
    for k in mod.params.index:
        if k.startswith(term) or (term.startswith('C(') and k.startswith(term)):
            rows.append(dict(test=label or term,stratum=stratum,term=k,OR=np.exp(mod.params[k]),lo=np.exp(mod.conf_int().loc[k,0]),hi=np.exp(mod.conf_int().loc[k,1]),p=mod.pvalues[k],n=len(d),n_err=int(d.y.sum())))
    return rows
def rate(d): return f'{d.y.mean()*100:.1f}% ({int(d.y.sum())}/{len(d)})'
EM=m[m.method=='EM']; XR=m[m.method=='X-ray']
# ---------------- A. binding status
print('\n== A. Binding status (amyloid removed)')
for meth,d in [('EM',EM),('X-ray',XR)]:
    print(meth, {k:rate(g) for k,g in d.groupby('ab_status2')})
d=EM.copy(); d['unbound']=(d.ab_status2=='unbound').astype(int); d['fidonly']=(d.ab_status2=='fiducial-only').astype(int)
for t in ['unbound','fidonly']:
    r=logit(d,t,stratum='EM'); TESTS+=r or []
    print(pd.DataFrame(r).round(4).to_string() if r else t+' failed')
# ---------------- B. antigen class
print('\n== B. Target class (EM): rate, adjusted OR vs all other EM entries')
cls_rows=[]
for meth,D in [('EM',EM),('X-ray',XR)]:
    for cl,g in D.groupby('tgt_cls'):
        if len(g)<15: continue
        d=D.copy(); d['is_cls']=(d.tgt_cls==cl).astype(int)
        r=logit(d,'is_cls',label=f'class: {cl}',stratum=meth)
        if r: 
            r[0].update(rate_in=rate(g),rate_out=rate(D[D.tgt_cls!=cl])); cls_rows+=r; TESTS+=r
print(pd.DataFrame(cls_rows)[['stratum','test','rate_in','rate_out','OR','lo','hi','p']].round(4).to_string())
# global LR test for class in EM
d=EM[EM.tgt_cls.map(EM.tgt_cls.value_counts())>=15]
m0=smf.glm('y ~ C(ycat)+res',d,family=sm.families.Binomial()).fit(); m1=smf.glm('y ~ C(tgt_cls)+C(ycat)+res',d,family=sm.families.Binomial()).fit()
lr=2*(m1.llf-m0.llf); dfree=m1.df_model-m0.df_model; p=stats.chi2.sf(lr,dfree)
print('EM global LR test target class: chi2',round(lr,1),'df',dfree,'p',p)
TESTS.append(dict(test='class: global LR (EM, unclustered)',stratum='EM',term='LR',OR=np.nan,p=p,n=len(d),n_err=int(d.y.sum())))
# excluding SARS-CoV-2
d2=d[~d.sars2]
m0=smf.glm('y ~ C(ycat)+res',d2,family=sm.families.Binomial()).fit(); m1=smf.glm('y ~ C(tgt_cls)+C(ycat)+res',d2,family=sm.families.Binomial()).fit()
lr=2*(m1.llf-m0.llf); dfree=m1.df_model-m0.df_model; p=stats.chi2.sf(lr,dfree)
print('EM non-SARS2 global LR test target class: chi2',round(lr,1),'df',dfree,'p',p)
TESTS.append(dict(test='class: global LR (EM non-SARS2, unclustered)',stratum='EM non-SARS2',term='LR',OR=np.nan,p=p,n=len(d2),n_err=int(d2.y.sum())))
# ---------------- C. organism
print('\n== C. Target organism (EM)')
org_rows=[]
for o,g in EM.groupby('tgt_org'):
    if len(g)<15: continue
    d=EM.copy(); d['is_o']=(d.tgt_org==o).astype(int); r=logit(d,'is_o',label=f'organism: {o}',stratum='EM')
    if r: r[0].update(rate_in=rate(g),rate_out=rate(EM[EM.tgt_org!=o])); org_rows+=r; TESTS+=r
print(pd.DataFrame(org_rows)[['test','rate_in','rate_out','OR','lo','hi','p']].round(4).to_string())
pd.DataFrame(cls_rows+org_rows).to_csv(f'{OUT}/tables/class_organism_tests.csv',index=False)
# ---------------- D. specific antigens (family level), EM
print('\n== D. Specific antigen families (EM, >=15 entries)')
EM2=EM.copy(); EM2['fam1']=EM2.tgt_fam.fillna('none').str.split('|').str[0]
famname=EM2.groupby('fam1').tgt_name.agg(lambda s: collections.Counter(s.fillna('')).most_common(1)[0][0][:45])
fam_rows=[]
for f_,g in EM2.groupby('fam1'):
    if len(g)<15 or f_=='none': continue
    d=EM2.copy(); d['is_f']=(d.fam1==f_).astype(int); r=logit(d,'is_f',label=f'antigen family: {famname[f_]} [{f_}]',stratum='EM')
    if r:
        r[0].update(rate_in=rate(g),rate_out=rate(EM2[EM2.fam1!=f_]),n_papers=g.paper.nunique(),n_err_papers=g[g.y==1].paper.nunique(),n_labs=g.last_author.nunique()); fam_rows+=r; TESTS+=r
F=pd.DataFrame(fam_rows).sort_values('p'); print(F[['test','rate_in','rate_out','n_papers','n_err_papers','OR','lo','hi','p']].round(4).to_string())
F.to_csv(f'{OUT}/tables/antigen_family_tests_em.csv',index=False)
# dispersion of error counts across antigen families beyond year/resolution (simulation under null model)
def dispersion(D,label):
    D=D.copy(); D['fam1']=D.tgt_fam.fillna('none').str.split('|').str[0]
    mod=smf.glm('y ~ C(ycat)+res',D,family=sm.families.Binomial()).fit(); D['p_hat']=mod.predict(D)
    grp=D.groupby('fam1')
    def stat(yv):
        o=pd.Series(yv,index=D.index).groupby(D.fam1).sum(); ex=grp.p_hat.sum(); v=grp.p_hat.apply(lambda q:(q*(1-q)).sum())
        k=ex>0
        return (((o-ex)**2/v.clip(lower=1e-9))[k]).sum()
    obs=stat(D.y.values); rng=np.random.default_rng(1)
    sims=np.array([stat(rng.binomial(1,D.p_hat.values)) for _ in range(500)])
    p=(1+(sims>=obs).sum())/(1+len(sims))
    # paper-level null: permute antigen family labels among papers within year (keeps paper clustering)
    print(f'{label}: families {D.fam1.nunique()}, dispersion obs {obs:.0f} vs null mean {sims.mean():.0f} (ratio {obs/sims.mean():.2f}), p={p:.3f}')
    TESTS.append(dict(test=f'antigen-family dispersion: {label}',stratum=label,term='chi2 ratio',OR=obs/sims.mean(),p=p,n=len(D),n_err=int(D.y.sum())))
dispersion(EM,'EM all'); dispersion(EM[~EM.sars2],'EM non-SARS2')
# same but paper-clustered: aggregate to paper (error paper = any error entry), family of paper = modal family
def disp_paper(D,label):
    D=D.copy(); D['fam1']=D.tgt_fam.fillna('none').str.split('|').str[0]
    P=D.groupby('paper').agg(y=('y','max'),fam1=('fam1',lambda s:s.mode().iat[0]),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size')).reset_index()
    P['ln']=np.log2(P.n)
    mod=smf.glm('y ~ C(ycat)+res+ln',P,family=sm.families.Binomial()).fit(); P['p_hat']=mod.predict(P)
    grp=P.groupby('fam1')
    def stat(yv):
        o=pd.Series(yv,index=P.index).groupby(P.fam1).sum(); ex=grp.p_hat.sum(); v=grp.p_hat.apply(lambda q:(q*(1-q)).sum())
        return (((o-ex)**2/v.clip(lower=1e-9))[ex>0]).sum()
    obs=stat(P.y.values); rng=np.random.default_rng(2); sims=np.array([stat(rng.binomial(1,P.p_hat.values)) for _ in range(500)])
    p=(1+(sims>=obs).sum())/(1+len(sims))
    print(f'{label} (paper level): papers {len(P)}, error papers {P.y.sum()}, families {P.fam1.nunique()}, dispersion ratio {obs/sims.mean():.2f}, p={p:.3f}')
    TESTS.append(dict(test=f'antigen-family dispersion, paper level: {label}',stratum=label,term='chi2 ratio',OR=obs/sims.mean(),p=p,n=len(P),n_err=int(P.y.sum())))
disp_paper(EM,'EM all'); disp_paper(EM[~EM.sars2],'EM non-SARS2')
# ---------------- E. competition
print('\n== E. Competition / novelty (bound or fiducial-only entries with a target)')
comp_terms=['l_labs_acc_6m','l_labs_acc_12m','l_labs_fam_6m','l_labs_fam_12m','l_labs_acc_contact_12m','race_1m','first_acc','first_fam','first_allpdb','l_prev12_acc','l_prev12_fam','l_prev12_allpdb','l_prev12_otherlab_acc']
desc_rows=[]; comp_rows=[]
strata=[('EM',EM),('EM non-SARS2',EM[~EM.sars2]),('EM SARS2',EM[EM.sars2]),('EM <=2019',EM[EM.period=='<=2019']),('EM 2020-21',EM[EM.period=='2020-21']),('EM 2022+',EM[EM.period=='2022+']),('X-ray',XR)]
for sname,D in strata:
    D=D[D.ab_status2.isin(['bound','fiducial-only'])&D.labs_acc_12m.notna()]
    for t in comp_terms:
        raw=t[2:] if t.startswith('l_') else t
        g1=D[D.y==1][raw]; g0=D[D.y==0][raw]
        extra='C(ycat) + res' if not sname.startswith('EM 20') and sname!='EM <=2019' else 'C(ycat) + res'
        if sname=='X-ray': extra='C(period) + res'
        r=logit(D,t,form_extra=extra,label=f'competition: {raw}',stratum=sname)
        if r:
            r[0].update(err_median=g1.median(),clean_median=g0.median(),err_mean=g1.mean(),clean_mean=g0.mean())
            comp_rows+=r
            if sname in ('EM','EM non-SARS2','X-ray','EM 2020-21','EM 2022+','EM <=2019','EM SARS2'): TESTS+=r
C_=pd.DataFrame(comp_rows)
print(C_[['stratum','test','err_median','clean_median','err_mean','clean_mean','OR','lo','hi','p','n','n_err']].round(3).to_string())
C_.to_csv(f'{OUT}/tables/competition_tests.csv',index=False)
# binned display
print('\n EM error rate by number of other labs (same antigen, +-12 months), by SARS2:')
D=EM[EM.labs_acc_12m.notna()].copy(); D['bin']=pd.cut(D.labs_acc_12m,[-1,0,2,9,1e5],labels=['0','1-2','3-9','>=10'])
print(D.groupby(['sars2','period','bin']).y.agg(['mean','size','sum']).round(3).to_string())
print('\n EM first antibody structure of the antigen, by period:')
print(D.groupby(['period','first_acc']).y.agg(['mean','size','sum']).round(3).to_string())
# paper-level sensitivity
print('\n== E2. Paper-level competition (EM, error paper = any error entry)')
P=EM[EM.labs_acc_12m.notna()].groupby('paper').agg(y=('y','max'),ycat=('ycat',lambda s:s.mode().iat[0]),res=('res','median'),n=('y','size'),sars2=('sars2','max'),
    labs12=('labs_acc_12m','median'),labs6=('labs_acc_6m','median'),fam12=('labs_fam_12m','median'),first=('first_acc','max'),prev=('prev12_acc','median'),prevall=('prev12_allpdb','median')).reset_index()
P['ln']=np.log2(P.n); P['paper']=P.paper
for t in ['labs12','labs6','fam12','prev','prevall']: P['l_'+t]=np.log2(1+P[t])
for sname,D in [('EM papers',P),('EM papers non-SARS2',P[~P.sars2])]:
    for t in ['l_labs12','l_labs6','l_fam12','first','l_prev','l_prevall']:
        r=logit(D,t,form_extra='C(ycat)+res+ln',label=f'paper-level competition: {t}',stratum=sname)
        if r:
            r[0].update(err_median=D[D.y==1][t].median(),clean_median=D[D.y==0][t].median()); TESTS+=r
            print(sname,t,{k:round(v,3) if isinstance(v,float) else v for k,v in r[0].items() if k in ('OR','lo','hi','p','n','n_err','err_median','clean_median')})
# ---------------- G. antibody role (unit level)
print('\n== G. Antibody role from contacts (unit = antibody in paper), EM')
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','method','res','year','deposit','unit','err_first','err_any','role','aux_reagent','sars2','vhh','ctype'])
ch=ch[~ch.pdb.isin(amy)]
R=pd.read_csv(f'{OUT}/tables/pdbwide_ab_chain_roles.csv.gz')
ch=ch.merge(R,on=['pdb','chain'],how='left')
U=ch.groupby('unit').agg(err=('err_first','max'),anyrun=('err_any','max'),method=('method','first'),res=('res','median'),dyear=('deposit',lambda s:pd.to_datetime(s).dt.year.min()),
    role_c=('ab_role_contact',lambda s:s.mode().iat[0] if s.notna().any() else 'unknown'),role_old=('role',lambda s:s.mode().iat[0]),aux=('aux_reagent','max'),sars2=('sars2','max'),
    agcls=('ab_ag_cls',lambda s:s.mode().iat[0] if s.notna().any() else 'unknown'),pdb=('pdb','first')).reset_index()
U=U[U.err|~U.anyrun].copy(); U['y']=U.err.astype(int); U['ycat']=U.dyear.clip(lower=2017).astype(str); U['paper']=U.unit.str.split('|').str[0]
UE=U[U.method=='EM']
print({k:rate(g) for k,g in UE.groupby('role_c')})
print(pd.crosstab(UE.role_c,UE.role_old))
for sname,D in [('EM units',UE),('EM units non-SARS2',UE[~UE.sars2]),('X-ray units',U[U.method=='X-ray'])]:
    D=D.copy(); D['fid']=(D.role_c=='fiducial').astype(int); D['nocont']=(D.role_c=='no antigen contact').astype(int)
    for t in ['fid','nocont']:
        r=logit(D,t,form_extra='C(ycat)+res' if 'X-ray' not in sname else 'res',label=f'antibody role: {t}',stratum=sname)
        if r:
            r[0].update(rate_in=rate(D[D[t]==1]),rate_out=rate(D[D[t]==0])); TESTS+=r; print(sname,t,r[0]['rate_in'],r[0]['rate_out'],round(r[0]['OR'],2),round(r[0]['p'],4))
# antibody-level antigen class (EM units, subject antibodies)
print('\n EM units by antigen class contacted:')
print(UE.groupby('agcls').y.agg(['mean','size','sum']).round(3).sort_values('size',ascending=False).to_string())
U.to_csv(f'{OUT}/tables/units_antibody_role.csv.gz',index=False)
# ---------------- BH
T=pd.DataFrame(TESTS); T=T[T.p.notna()].copy()
from statsmodels.stats.multitest import multipletests
T['q']=multipletests(T.p,method='fdr_bh')[1]
T.sort_values('p').to_csv(f'{OUT}/tables/all_tests_bh.csv',index=False)
print('\n== Tests with q<0.1:'); print(T[T.q<0.1].sort_values('p')[['test','stratum','term','OR','lo','hi','p','q','n','n_err']].round(4).to_string())
print('total tests',len(T))
