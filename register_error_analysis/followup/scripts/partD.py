import re, json, pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings, collections
warnings.filterwarnings('ignore'); pd.set_option('display.width',250)
A='/tmp/analysis'; O=f'{A}/out'
d=pd.read_pickle(f'{A}/domains_full.pkl')
ent=pd.read_pickle(f'{A}/entries35.pkl')
ch=pd.read_pickle(f'{A}/chains.pkl')
inst=pd.read_pickle(f'{A}/instances35.pkl')
rng=np.random.default_rng(0)
import sys; sys.path.insert(0,'/tmp/scripts'); from tiers import tier as TIER
ent['tier']=ent.journal.map(TIER)
d['tier']=d.pdb.map(dict(zip(ent.pdb,ent.tier)))
def fit(f,data,grp='pdb'):
    return smf.logit(f,data=data).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(data[grp])[0]})
def ors(m,filt):
    p=m.params[m.params.index.str.contains(filt)]; c=m.conf_int().loc[p.index]
    return pd.DataFrame({'OR':np.exp(p),'lo':np.exp(c[0]),'hi':np.exp(c[1]),'p':m.pvalues[p.index]}).round(3)
def rate_ci(s,B=300):
    g=s.groupby('pdb').flag.agg(['sum','count']); a,c=g['sum'].values,g['count'].values;n=len(g)
    b=[a[i].sum()/c[i].sum() for i in (rng.integers(0,n,n) for _ in range(B))]
    return round(100*a.sum()/c.sum(),2),*np.round(np.percentile(b,[2.5,97.5])*100,2)
COV="C(method)*C(rb) + C(yr) + logL"
nab=d[d.cls=='Non-antibody']; abV=d[d.cls=='Antibody V']
d['abent']=d.pdb.isin(set(abV.pdb))
################ D1 journal tier
print('===== D1. Journal tier')
rows=[]
for tname,s in d[d.cls!='TCR/other'].groupby('tier'):
    for c,ss in s.groupby('cls'):
        r=rate_ci(ss); rows.append(dict(tier=tname,cls=c,domains=len(ss),entries=ss.pdb.nunique(),flag_pct=r[0],lo=r[1],hi=r[2],pct_EM=round(100*(ss.method=='EM').mean(),1)))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv(f'{O}/D1_rates_by_tier.csv',index=False)
for lab,data in [('all non-antibody domains',nab),('antibody V domains',abV),('all domains in antibody entries',d[d.abent&(d.cls!='TCR/other')])]:
    m=fit(f"flag ~ C(tier,Treatment('Other journal')) + {COV}",data); print(f'[{lab}]'); print(ors(m,'tier').to_string())
# tier x role for antibody domains
w=d[d.role.isin(['Auxiliary','Antibody of interest'])&d.cls.isin(['Antibody V','Antibody C'])].copy()
w['aux']=(w.role=='Auxiliary').astype(int); w['cns']=(w.tier=='CNS').astype(int); w['hi']=w.tier.isin(['CNS','CNS sister journal']).astype(int)
m=fit(f"flag ~ hi*aux + C(cls) + {COV}",w); print('[antibody domains: high-impact x auxiliary]'); print(ors(m,'hi|aux').to_string())
print(w.groupby(['hi','role']).agg(domains=('flag','size'),flagged=('flag','sum'),entries=('pdb','nunique'),flagged_entries=('pdb',lambda x: x[w.loc[x.index,'flag']==1].nunique()),pct=('flag',lambda x:round(100*x.mean(),2))).to_string())
################ D2 entries per paper
print('\n===== D2. Structures per paper (PDB-wide counts of entries sharing the primary citation)')
cit=[]
for l in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    e=json.loads(l); c=e.get('rcsb_primary_citation') or {}
    key=c.get('pdbx_database_id_DOI') or (('PMID'+str(c['pdbx_database_id_PubMed'])) if c.get('pdbx_database_id_PubMed') else None)
    meth=[m['method'] for m in (e.get('exptl') or [])]
    cit.append(dict(pdb=e['rcsb_id'].lower(),key=key,journal=c.get('journal_abbrev'),year=c.get('year'),
                    method='EM' if 'ELECTRON MICROSCOPY' in meth else ('X-ray' if 'X-RAY DIFFRACTION' in meth else 'other')))
cit=pd.DataFrame(cit)
import importlib.util,sys
spec=importlib.util.spec_from_file_location('b2','/tmp/scripts/build2.py')
cit['tier']=cit.journal.map(TIER)
papers=cit.dropna(subset=['key']).groupby('key').agg(n=('pdb','size'),tier=('tier','first'),year=('year','first'),pct_em=('method',lambda x:(x=='EM').mean()))
p2=papers[(papers.year>=2015)&(papers.year<=2025)]
P=p2.groupby('tier').n.describe(percentiles=[.5,.75,.9]).round(2); print(P.to_string()); P.to_csv(f'{O}/D2_entries_per_paper_by_tier_PDBwide_2015_2025.csv')
pe=p2[p2.pct_em>=0.5]; print('EM-majority papers:'); print(pe.groupby('tier').n.describe(percentiles=[.5,.75,.9]).round(2).to_string())
# antibody papers in our set by role
ap=ent[ent.role.isin(['Auxiliary','Antibody of interest'])].dropna(subset=['cite_key']).drop_duplicates('cite_key')
print('Antibody papers (3-5 A set), entries per paper:'); print(ap.groupby(['role','tier']).entries_per_paper.median().unstack().round(1))
# error vs entries per paper
d['log_epp']=np.log2(d.entries_per_paper.clip(lower=1))
for lab,data in [('non-antibody domains',d[d.cls=='Non-antibody']),('antibody V',d[d.cls=='Antibody V'])]:
    m=fit(f"flag ~ log_epp + C(tier,Treatment('Other journal')) + {COV}",data); print(f'[{lab}] per doubling of entries/paper:'); print(ors(m,'log_epp').to_string())
################ D3 race / competition proxies
print('\n===== D3. Competition / time-pressure proxies')
d['comp_bin']=pd.cut(d.competitors_6mo,[-1,0,2,9,1000],labels=['0','1–2','3–9','≥10']).astype(str)
d['hold_short']=(d.hold_days<=60).astype(int)
d['hold_bin']=pd.cut(d.hold_days,[-1,30,90,180,365,1e5],labels=['≤30d','31–90d','91–180d','181–365d','>1y']).astype(str)
for var in ['comp_bin','novel_target','hold_bin','sars2']:
    rows=[]
    for c,s in d[d.cls.isin(['Antibody V','Non-antibody'])].groupby('cls'):
        for k,ss in s.groupby(var):
            r=rate_ci(ss,150); rows.append(dict(cls=c,level=k,domains=len(ss),entries=ss.pdb.nunique(),flag_pct=r[0],lo=r[1],hi=r[2],pct_EM=round(100*(ss.method=='EM').mean(),1)))
    t=pd.DataFrame(rows); print(f'-- {var}'); print(t.to_string(index=False)); t.to_csv(f'{O}/D3_rates_by_{var}.csv',index=False)
for lab,data in [('non-antibody',d[d.cls=='Non-antibody'].dropna(subset=['competitors_6mo'])),('antibody V',d[d.cls=='Antibody V'].dropna(subset=['competitors_6mo']))]:
    m=fit(f"flag ~ C(comp_bin,Treatment('0')) + novel_target + C(hold_bin,Treatment('91–180d')) + C(tier,Treatment('Other journal')) + log_epp + {COV}",data)
    print(f'[{lab}] multivariable'); t=ors(m,'comp|novel|hold|tier|epp'); print(t.to_string()); t.to_csv(f'{O}/D3_multivariable_{lab.replace(" ","_")}.csv')
# SARS-CoV-2 natural experiment: viral antibody-of-interest entries
v=d[(d.role=='Antibody of interest')&(d.viral)&d.cls.isin(['Antibody V','Antibody C','Non-antibody'])].copy()
v['grp']=np.where(v.sars2,'SARS-CoV-2','Other viruses'); v['period']=np.where(v.year<=2019,'≤2019',np.where(v.year==2020,'2020','2021–22'))
print('-- viral antibody-of-interest entries: flag % (domains)')
print(v.groupby(['cls','grp','period']).agg(n=('flag','size'),entries=('pdb','nunique'),pct=('flag',lambda x:round(100*x.mean(),2)),pct_EM=('method',lambda x:round(100*(x=='EM').mean()))).to_string())
vv=v[(v.year>=2020)&(v.cls=='Antibody V')].copy(); vv['s2']=vv.sars2.astype(int); vv['y2020']=(vv.year==2020).astype(int)
for lab,f in [('base',"flag ~ s2*y2020 + C(method) + res + logL"),('+local quality',"flag ~ s2*y2020 + C(method) + res + logL + lq_z")]:
    m=fit(f,vv.dropna(subset=['lq_z']) if 'lq' in f else vv); print(f'[Ab V, viral Ab-of-interest 2020-22: SARS-CoV-2 x 2020; {lab}]'); print(ors(m,'s2|y2020|lq').to_string())
s20=vv[(vv.sars2)&(vv.year==2020)]
fl=s20[s20.flag==1]
print('SARS-CoV-2 2020 Ab V: flagged domains',len(fl),'in',fl.pdb.nunique(),'entries from',ent.set_index('pdb').loc[fl.pdb.unique()].last_author.nunique(),'labs; of',s20.pdb.nunique(),'entries /',ent.set_index('pdb').loc[s20.pdb.unique()].last_author.nunique(),'labs')
print(ent.set_index('pdb').loc[fl.pdb.unique()].groupby('last_author').size().sort_values(ascending=False).head(8).to_string())
print('mean local quality z: SARS2-2020',round(s20.lq_z.mean(),2),' other viruses 2020',round(vv[(~vv.sars2)&(vv.year==2020)].lq_z.mean(),2))
nos=d[(d.cls=='Antibody V')&(~d.sars2)].dropna(subset=['competitors_6mo'])
m=fit(f"flag ~ C(comp_bin,Treatment('0')) + novel_target + C(hold_bin,Treatment('91–180d')) + C(tier,Treatment('Other journal')) + log_epp + {COV}",nos)
print('[antibody V, excluding SARS-CoV-2] multivariable'); t=ors(m,'comp|novel|hold|tier|epp'); print(t.to_string()); t.to_csv(f'{O}/D3_multivariable_antibodyV_noSARS2.csv')
# aux x competition for antibody domains
w=d[d.role.isin(['Auxiliary','Antibody of interest'])&d.cls.isin(['Antibody V','Antibody C'])].dropna(subset=['competitors_6mo']).copy()
w['aux']=(w.role=='Auxiliary').astype(int); w['crowded']=(w.competitors_6mo>=3).astype(int)
m=fit(f"flag ~ aux*crowded + C(cls) + {COV}",w); print('[antibody domains: auxiliary x crowded (≥3 competing groups within 6 months)]'); print(ors(m,'aux|crowded').to_string())
print((100*w.groupby(['crowded','role']).flag.mean()).round(2).unstack())
################ E co-occurrence
print('\n===== E. Co-occurrence of errors across chains')
cf=ch[['pdb','chain','flag']].merge(inst[['pdb','chain','entity','ab','tcr','L']],on=['pdb','chain'])
ef=cf.groupby(['pdb','entity']).agg(flag=('flag','max'),ab=('ab','first'),tcr=('tcr','first'),L=('L','first'),nch=('chain','size')).reset_index()
ef=ef.merge(ent[['pdb','method','res','year','role']],on='pdb')
ef['rb']=pd.cut(ef.res,[2.999,3.25,3.5,3.75,4.0,4.5,5.0]).astype(str); ef['yr']=pd.cut(ef.year,[0,2017,2019,2023]).astype(str)
abE=ef[ef.pdb.isin(set(ef.pdb[ef.ab]))&~ef.tcr]
g=abE.groupby('pdb').apply(lambda x: pd.Series({'ab_flag':int(x[x.ab].flag.max()),'ag_flag':int(x[~x.ab].flag.max()) if (~x.ab).any() else np.nan,
     'ag_res':(x[~x.ab].L*x[~x.ab].nch).sum(),'ab_res':(x[x.ab].L*x[x.ab].nch).sum()})).dropna().reset_index()
g=g.merge(ent[['pdb','method','res','year','role']],on='pdb'); g['rb']=pd.cut(g.res,[2.999,3.25,3.5,3.75,4.0,4.5,5.0]).astype(str); g['yr']=pd.cut(g.year,[0,2017,2019,2023]).astype(str)
ct=pd.crosstab(g.ab_flag,g.ag_flag); print(ct)
print('P(antigen flagged | antibody flagged)=%.1f%%  vs | antibody clean=%.1f%%'%(100*g[g.ab_flag==1].ag_flag.mean(),100*g[g.ab_flag==0].ag_flag.mean()))
from statsmodels.stats.contingency_tables import StratifiedTable
tabs=[pd.crosstab(s.ab_flag,s.ag_flag).reindex(index=[0,1],columns=[0,1],fill_value=0).values for k,s in g.groupby(['method','rb'])]
st=StratifiedTable([t for t in tabs if t.sum()>0]); print('Mantel-Haenszel OR (method x resolution strata)=%.2f [%.2f-%.2f], p=%.2g'%(st.oddsratio_pooled,*st.oddsratio_pooled_confint(),st.test_null_odds().pvalue))
g['log_ag']=np.log(g.ag_res.clip(lower=50)); g['log_ab']=np.log(g.ab_res.clip(lower=50))
m=smf.logit("ag_flag ~ ab_flag + C(method)*C(rb) + C(yr) + log_ag",data=g).fit(disp=0); print('logit OR antigen-flag ~ antibody-flag:',ors(m,'ab_flag').to_string())
for ro,s in g[g.role.isin(['Auxiliary','Antibody of interest'])].groupby('role'):
    print(f'  {ro}: n entries={len(s)}, P(ag|ab flagged)={100*s[s.ab_flag==1].ag_flag.mean():.1f}% (n={int(s.ab_flag.sum())}), P(ag|ab clean)={100*s[s.ab_flag==0].ag_flag.mean():.1f}%')
# general comparison: non-antibody multi-entity entries, distinct entities
nonab=ef[~ef.pdb.isin(set(abE.pdb))]
multi=nonab.groupby('pdb').filter(lambda x: len(x)>=2)
multi=multi.copy(); multi['others_flag']=multi.groupby('pdb').flag.transform('sum')-multi.flag
multi['other_any']=(multi.others_flag>0).astype(int); multi['logL']=np.log(multi.L.clip(lower=20)*multi.nch)
m=fit("flag ~ other_any + C(method)*C(rb) + C(yr) + logL",multi); print('[non-antibody multi-entity entries] entity flagged ~ any other entity flagged:'); print(ors(m,'other_any').to_string())
abE2=abE.copy(); abE2['others_flag']=abE2.groupby('pdb').flag.transform('sum')-abE2.flag; abE2['other_any']=(abE2.others_flag>0).astype(int); abE2['logL']=np.log(abE2.L.clip(lower=20)*abE2.nch)
abE2['other_ab_flag']=(abE2.groupby('pdb').apply(lambda g:(g.flag*g.ab).sum()).reindex(abE2.pdb).values-abE2.flag*abE2.ab>0).astype(int)
abE2['ag_flag']=(abE2.groupby('pdb').apply(lambda g:(g.flag*(~g.ab)).sum()).reindex(abE2.pdb).values>0).astype(int)
m=fit("flag ~ other_ab_flag + ag_flag + C(method)*C(rb) + C(yr) + logL",abE2[abE2.ab]); print('[antibody entities] flagged ~ other antibody entity (e.g. partner H/L) flagged + antigen flagged:'); print(ors(m,'flag').to_string())
# null by permutation within method x res-bin strata (entry-level)
obs=((g.ab_flag==1)&(g.ag_flag==1)).sum(); sims=[]
for _ in range(2000):
    gg=g.copy(); gg['ag_p']=gg.groupby(['method','rb']).ag_flag.transform(lambda x: rng.permutation(x.values)); sims.append(((gg.ab_flag==1)&(gg.ag_p==1)).sum())
print(f'entries with both flagged: observed={obs}, expected under within-stratum independence={np.mean(sims):.1f}, p(sim>=obs)={np.mean(np.array(sims)>=obs):.4f}')
################ Extras
print('\n===== X1. Lab-level clustering (last author)')
ef2=ch.groupby('pdb').agg(flag=('flag','max'),nres=('length','sum')).reset_index().merge(ent[['pdb','method','res','year','last_author','n_entities']],on='pdb')
ef2['rb']=pd.cut(ef2.res,[2.999,3.25,3.5,3.75,4.0,4.5,5.0]).astype(str); ef2['yr']=pd.cut(ef2.year,[0,2014,2017,2019,2023]).astype(str); ef2['lognres']=np.log(ef2.nres)
m=smf.logit("flag ~ C(method)*C(rb) + C(yr) + lognres + np.log(n_entities)",data=ef2).fit(disp=0); ef2['p']=m.predict(ef2)
big=ef2.groupby('last_author').filter(lambda x: len(x)>=5)
def disp(df,col):
    g=df.groupby('last_author').agg(o=(col,'sum'),e=('p','sum'),v=('p',lambda x:(x*(1-x)).sum()))
    return ((g.o-g.e)**2/g.v).sum()/len(g)
obsd=disp(big,'flag'); sims=[]
for _ in range(500):
    b=big.copy(); b['sim']=rng.random(len(b))<b.p; sims.append(disp(b,'sim'))
print(f'labs with ≥5 entries: {big.last_author.nunique()} labs, {len(big)} entries; dispersion observed={obsd:.2f} vs simulated={np.mean(sims):.2f} (95% {np.percentile(sims,2.5):.2f}-{np.percentile(sims,97.5):.2f}); p={np.mean(np.array(sims)>=obsd):.3f}')
gl=big.groupby('last_author').agg(o=('flag','sum'),e=('p','sum'),n=('flag','size'))
gl['oe']=gl.o/gl.e; top=gl.sort_values('oe',ascending=False)
print('share of flagged entries from the 10%% of labs with highest O/E: %.0f%% (expected under model %.0f%%)'%(float(100*top.head(len(top)//10).o.sum()/gl.o.sum()),float(100*top.head(len(top)//10).e.sum()/gl.e.sum())))
print('\n===== X2. Were flagged entries later revised (coordinates replaced)?')
x=ent.merge(ef2[['pdb','flag']],on='pdb'); print((100*x.groupby('flag').atom_model_revised.mean()).round(2))
print('\n===== X3. Auxiliary subtypes (3–5 Å)')
desc={}
for l in open('/tmp/supp/meta2/meta_35.jsonl'):
    e=json.loads(l); desc[e['rcsb_id'].lower()]=' '.join((p['rcsb_polymer_entity']['pdbx_description'] or '') for p in e['polymer_entities'] or [])
d['gstab']=d.pdb.map(lambda p: bool(re.search(r'nanobody[- ]?35|\bnb[- ]?35\b|scfv[- ]?16|\bfv[- ]?16|svfv16',desc.get(p,''),re.I)))
w=d[d.role.isin(['Auxiliary','Antibody of interest'])&d.cls.isin(['Antibody V','Antibody C','Non-antibody'])].copy()
w['role2']=np.where(w.role=='Auxiliary',np.where(w.gstab,'Aux: G-protein stabiliser','Aux: other'),'Antibody of interest')
print(w.groupby(['role2','cls']).agg(domains=('flag','size'),entries=('pdb','nunique'),pct=('flag',lambda x:round(100*x.mean(),2))).unstack().to_string())
d.to_pickle(f'{A}/domains_full2.pkl')
print('\n===== X4. Who contributes the high-impact auxiliary antibody flags?')
hw=d[(d.role=='Auxiliary')&d.cls.isin(['Antibody V','Antibody C'])&(d.flag==1)]
hx=ent.set_index('pdb').loc[hw.pdb.unique(),['tier','journal','last_author','method','res','year','title']]
print(hx.sort_values('tier').to_string())
# paper-level test
from scipy.stats import fisher_exact
aw2=d[d.role=='Auxiliary'].copy(); aw2['hi']=aw2.tier.isin(['CNS','CNS sister journal'])
pp=aw2[aw2.cls.isin(['Antibody V','Antibody C'])].groupby(aw2.cite_key.fillna(aw2.pdb)).agg(hi=('hi','first'),ab_flag=('flag','max'))
tab=pd.crosstab(pp.hi,pp.ab_flag); print('Auxiliary-antibody papers: rows high-impact, cols any antibody domain flagged'); print(tab); print('Fisher p=%.3g'%fisher_exact(tab.values)[1])
ip=d[(d.role=='Antibody of interest')&d.cls.isin(['Antibody V','Antibody C'])].copy(); ip['hi']=ip.tier.isin(['CNS','CNS sister journal'])
pp2=ip.groupby(ip.cite_key.fillna(ip.pdb)).agg(hi=('hi','first'),ab_flag=('flag','max'))
tab2=pd.crosstab(pp2.hi,pp2.ab_flag); print('Antibody-of-interest papers:'); print(tab2); print('Fisher p=%.3g'%fisher_exact(tab2.values)[1])
print('auxiliary entries, antigen (non-antibody) domain flag % by tier:')
aw=d[(d.role=='Auxiliary')&(d.cls=='Non-antibody')]; print(aw.groupby(aw.tier.isin(['CNS','CNS sister journal'])).agg(domains=('flag','size'),flagged=('flag','sum'),pct=('flag',lambda x:round(100*x.mean(),2))))
