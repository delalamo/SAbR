import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from statsmodels.discrete.conditional_models import ConditionalLogit
warnings.filterwarnings('ignore'); pd.set_option('display.width',250)
A='/tmp/analysis'; O=f'{A}/out'
d=pd.read_pickle(f'{A}/ecod_grouped.pkl')
ent=pd.read_pickle(f'{A}/entries35.pkl')
q=pd.read_pickle(f'{A}/domain_localq.pkl')[['ecod_domain_id','pdb','chain','Q_SCORE','RSCC']]
d=d.merge(ent.drop(columns=['method','res','year','title']),on='pdb',how='left').merge(q,on=['ecod_domain_id','pdb','chain'],how='left')
d['cls']=np.select([d.group=='Antibody V',d.group=='Antibody C',d.ab=='other'],['Antibody V','Antibody C','Non-antibody'],'TCR/other')
d['era']=pd.cut(d.year,[0,2017,2019,2023],labels=['≤2017','2018–19','2020–22']).astype(str)
d['rb']=d.res_bin.astype(str); d['logL']=np.log(d.dlen.clip(lower=20))
d['yr']=d.era
# local quality z within method
d['lq']=np.where(d.method=='EM',d.Q_SCORE,d.RSCC)
d['lq_z']=d.groupby('method').lq.transform(lambda x:(x-x.mean())/x.std())
d['lq_rel']=d.lq_z-d.groupby('pdb').lq_z.transform('median')
def clfit(formula,data):
    data=data.dropna(subset=[c for c in ['lq_z','lq_rel'] if c in formula])
    return smf.logit(formula,data=data).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(data.pdb)[0]}),len(data)
def ors(m,filt):
    p=m.params[m.params.index.str.contains(filt)]; c=m.conf_int().loc[p.index]
    return pd.DataFrame({'OR':np.exp(p),'lo':np.exp(c[0]),'hi':np.exp(c[1]),'p':m.pvalues[p.index]}).round(3)
rng=np.random.default_rng(0)
def rate_ci(s,B=300):
    g=s.groupby('pdb').flag.agg(['sum','count']); a,c=g['sum'].values,g['count'].values;n=len(g)
    if n==0: return (np.nan,np.nan,np.nan)
    b=[a[i].sum()/c[i].sum() for i in (rng.integers(0,n,n) for _ in range(B))]
    return 100*a.sum()/c.sum(),*np.percentile(b,[2.5,97.5])*100
out={}
############ B: X-ray -> EM transition
print('\n===== B. Transition X-ray -> EM')
ab_ent=set(d.pdb[d.cls=='Antibody V'])
d['cls2']=np.where(d.cls.isin(['Antibody V','Antibody C']),d.cls,np.where(d.pdb.isin(ab_ent)&(d.cls=='Non-antibody'),'Non-antibody, in antibody entries',np.where(d.cls=='Non-antibody','Non-antibody, other entries','TCR')))
rows=[]
for c,s in d[d.cls2!='TCR'].groupby('cls2'):
    for e,se in s.groupby('era'):
        r,lo,hi=rate_ci(se); rows.append(dict(cls=c,era=e,domains=len(se),pct_EM=100*(se.method=='EM').mean(),flag_pct=r,lo=lo,hi=hi,
            em_rate=100*se[se.method=='EM'].flag.mean(),xray_rate=100*se[se.method=='X-ray'].flag.mean(),med_res=se.res.median()))
Bt=pd.DataFrame(rows).round(2); print(Bt.to_string(index=False)); Bt.to_csv(f'{O}/B_rates_by_era_method.csv',index=False)
# decomposition e1 -> e3 with strata method x res bin
dec=[]
for c,s in d[d.cls2!='TCR'].groupby('cls2'):
    s1=s[s.era=='≤2017']; s3=s[s.era=='2020–22']
    w1=s1.groupby(['method','rb']).size()/len(s1)
    r3=s3.groupby(['method','rb']).flag.mean(); r3m=s3.groupby('method').flag.mean()
    r3s=sum(w*(r3.get(k,np.nan) if not np.isnan(r3.get(k,np.nan)) else r3m.get(k[0],s3.flag.mean())) for k,w in w1.items())
    R1,R3=s1.flag.mean(),s3.flag.mean()
    # also: e3 at e1 method mix only
    wm=s1.groupby('method').size()/len(s1); r3mm=sum(w*r3m.get(k,np.nan) for k,w in wm.items())
    dec.append(dict(cls=c,rate_e1=100*R1,rate_e3=100*R3,change=100*(R3-R1),e3_at_e1_method_res_mix=100*r3s,
        within_strata_change=100*(r3s-R1),composition_change=100*(R3-r3s),pct_of_change_from_composition=100*(R3-r3s)/(R3-R1) if R3!=R1 else np.nan,
        pct_EM_e1=100*(s1.method=='EM').mean(),pct_EM_e3=100*(s3.method=='EM').mean()))
Dt=pd.DataFrame(dec).round(2); print(Dt.to_string(index=False)); Dt.to_csv(f'{O}/B_decomposition.csv',index=False)
# yearly series for antibody V
ys=d[d.cls=='Antibody V'].groupby('year').agg(domains=('flag','size'),pct_EM=('method',lambda x:100*(x=='EM').mean()),flag_pct=('flag',lambda x:100*x.mean()))
ys.round(2).to_csv(f'{O}/B_antibodyV_by_year.csv'); print(ys[ys.index>=2010].round(2).T.to_string())
############ C: role
print('\n===== C. Antibody role (auxiliary vs antibody of interest)')
w=d[d.role.isin(['Auxiliary','Antibody of interest'])&(d.cls!='TCR/other')].copy()
w['aux']=(w.role=='Auxiliary').astype(int)
rows=[]
for (ro,c,m),s in w.groupby(['role','cls','method']):
    r,lo,hi=rate_ci(s); rows.append(dict(role=ro,cls=c,method=m,domains=len(s),entries=s.pdb.nunique(),flag_pct=r,lo=lo,hi=hi,mean_lq_z=s.lq_z.mean(),med_res=s.res.median()))
for (ro,c),s in w.groupby(['role','cls']):
    r,lo,hi=rate_ci(s); rows.append(dict(role=ro,cls=c,method='All',domains=len(s),entries=s.pdb.nunique(),flag_pct=r,lo=lo,hi=hi,mean_lq_z=s.lq_z.mean(),med_res=s.res.median()))
Ct=pd.DataFrame(rows).round(2); print(Ct.sort_values(['method','cls','role']).to_string(index=False)); Ct.to_csv(f'{O}/C_rates_by_role.csv',index=False)
print('entries by role x method:'); print(w.groupby(['role','method']).pdb.nunique().unstack())
res={}
for lab,f in [('base',"flag ~ C(cls,Treatment('Non-antibody'))*aux + C(method)*C(rb) + C(yr) + logL"),
              ('+local quality',"flag ~ C(cls,Treatment('Non-antibody'))*aux + C(method)*C(rb) + C(yr) + logL + lq_z")]:
    m,n=clfit(f,w); print(f'\n[{lab}] n={n}'); t=ors(m,'aux|cls|lq'); print(t.to_string()); t.to_csv(f'{O}/C_logit_{lab.replace(" ","_").replace("+","plus")}.csv')
# Ab V only: aux vs interest
for lab,f in [('AbV only',"flag ~ aux + C(method)*C(rb) + C(yr) + logL"),('AbV only +lq',"flag ~ aux + C(method)*C(rb) + C(yr) + logL + lq_z")]:
    m,n=clfit(f,w[w.cls=='Antibody V']); print(f'[{lab}] n={n}'); print(ors(m,'aux|lq').to_string())
# conditional logit within entry
cw=w.dropna(subset=['lq_z']).copy()
cw['isV']=(cw.cls=='Antibody V').astype(int); cw['isC']=(cw.cls=='Antibody C').astype(int)
cw['isV_aux']=cw.isV*cw.aux; cw['isC_aux']=cw.isC*cw.aux
keep=cw.groupby('pdb').flag.transform(lambda x: 0<x.sum()<len(x)).astype(bool); cc=cw[keep]
for lab,cols in [('CL',['isV','isC','isV_aux','isC_aux','logL']),('CL+lq',['isV','isC','isV_aux','isC_aux','logL','lq_z'])]:
    m=ConditionalLogit(cc.flag.values,cc[cols].values,groups=pd.factorize(cc.pdb)[0]).fit(disp=0)
    print(f'[{lab}] informative entries={cc.pdb.nunique()} domains={len(cc)}')
    print(pd.DataFrame({'OR':np.exp(m.params),'lo':np.exp(m.conf_int()[:,0]),'hi':np.exp(m.conf_int()[:,1]),'p':m.pvalues},index=cols).round(3).to_string())
# local quality itself by role/cls
print('\nMean local quality (z) by role x class x method'); print(w.groupby(['method','role','cls']).lq_z.mean().unstack().round(2))
# validation metrics per chain
inst=pd.read_pickle(f'{A}/instances35.pkl').merge(ent[['pdb','role','method','res','year']],on='pdb')
iv=inst[inst.role.isin(['Auxiliary','Antibody of interest'])&~inst.tcr].copy()
iv['ab']=iv.ab.astype(int); iv['aux']=(iv.role=='Auxiliary').astype(int)
print('\nPer-chain validation metrics (means, %):')
print((100*iv.groupby(['method','role','ab'])[['clash_frac','angle_frac','unobs_frac']].mean()).round(2).to_string())
for y in ['clash_frac','angle_frac']:
    iv[y+'_dm']=iv[y]-iv.groupby('pdb')[y].transform('mean')
    iv['ab_dm']=iv.ab-iv.groupby('pdb').ab.transform('mean'); iv['abaux_dm']=iv.ab*iv.aux-iv.groupby('pdb').apply(lambda g:(g.ab*g.aux).mean()).reindex(iv.pdb).values
    m=smf.ols(f'{y}_dm ~ ab_dm + abaux_dm -1',data=iv).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(iv.pdb)[0]})
    print(f'entry-FE OLS {y}: antibody-chain effect (interest)={100*m.params.ab_dm:.2f} pp [p={m.pvalues.ab_dm:.3g}], extra in auxiliary={100*m.params.abaux_dm:.2f} pp [p={m.pvalues.abaux_dm:.3g}]')
# modelling completeness: do Fab C domains get modelled?
fab=d[(d.cls.isin(['Antibody V','Antibody C']))&d.role.isin(['Auxiliary','Antibody of interest'])]
cm=fab.groupby(['pdb','chain','role','method']).cls.apply(lambda x:'Antibody C' in set(x)).reset_index(name='has_C')
print('\nFraction of antibody chains with a modelled constant domain:'); print((100*cm.groupby(['method','role']).has_C.mean()).round(1).unstack())
d.to_pickle(f'{A}/domains_full.pkl'); w.to_pickle(f'{A}/w_roles.pkl')
