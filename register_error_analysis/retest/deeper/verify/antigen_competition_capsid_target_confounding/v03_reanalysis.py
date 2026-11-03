# Adversarial re-analysis of "capsid_target" (antigen_competition lens): capsid-binding antibodies have more errors.
# Confounders checked: period, resolution, method, chain type, SARS-CoV-2, paper and LAB-NETWORK clustering (co-author merged),
# number of V chains per entry, antibody source organism (PDB-annotated), germline identity / CDR lengths / nanobody,
# unbuilt residues, local map fit, and the error definition itself (CDR-internal NaN-delta runs; long-CDR3 SAbR failures).
import warnings, re, json, collections
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_confounding'
CC=f'{SP}/deeper/cache/verify_capsid_confounding'
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)

m=pd.read_csv(f'{INV}/tables/focal_entries_antigen_features.csv.gz')     # error (first-occurrence) vs clean entries, amyloid removed
m['cap']=(m.tgt_cls=='viral capsid').astype(int)
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','ctype','method','species','v_gene','v_id','L1','L2','L3','L1_atyp','L2_atyp','L3_atyp','vhh','lq_z','irm','fit','unit','paper','err_first','err_any','err_copied_or_artifact_only','sars2','deposit','res'])
org=pd.read_csv(f'{CC}/chain_ab_org.csv')
gap=pd.read_csv(f'{CC}/chain_gaps.csv.gz')
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
au=pd.read_csv(f'{CC}/entry_authors.csv').fillna('')
def simp(s):
    s=str(s).lower()
    for k,v in [('homo sapiens','human'),('mus musculus','mouse'),('bos taurus','cow'),('lama','camelid'),('vicugna','camelid'),('camel','camelid'),
                ('oryctolagus','rabbit'),('macaca','macaque'),('rattus','rat'),('gallus','chicken'),('synthetic','synthetic'),('sus scrofa','pig')]:
        if k in s: return v
    return 'other'
org['aborg']=org.ab_org.map(simp)
ch=ch.merge(org[['pdb','chain','aborg']],on=['pdb','chain'],how='left').merge(gap[['pdb','chain','n_gaps','nterm_trunc','any_gap']],on=['pdb','chain'],how='left')
ch['aborg']=ch.aborg.fillna('other')

# ---------- lab networks: last author merged with co-author-linked groups behind the capsid error papers (manual, from author lists)
def labnet(r):
    la=r.last_author if isinstance(r.last_author,str) else ''
    names=(r.audit+'|'+r.cit_authors)
    if la in ('Hafenstein, S.','Hafenstein, S.L.','Charnesky, A.J.') or 'Hafenstein' in names: return 'NET Hafenstein'
    if la in ('Lou, Z.','Kun, L.') : return 'NET Lou Z./Li K. (FMDV, bovine)'
    if la in ('Feng, R.','Fu, W.') or (la=='Wang, X.' and ('Rao, Z.' in names or 'Zhu, L.' in names)): return 'NET Wang X./Zhu L. (IBP)'
    if la in ('Cheng, T.','Li, S.W.','Xia, N.','Xia, N.S.') or 'Xia, N' in names: return 'NET Xiamen (Xia N.)'
    return la
m=m.merge(au,on='pdb',how='left'); m[['audit','cit_authors']]=m[['audit','cit_authors']].fillna('')
m['lab']=m.apply(labnet,axis=1)
# a second, broader merge: Rao Z. co-authored both the Lou and Wang X. networks
m['lab_rao']=np.where(m.lab.str.startswith('NET Lou')|m.lab.str.startswith('NET Wang')|((m.cit_authors+m.audit).str.contains(r'Rao, Z\.')),'NET Rao Z. wider',m.lab)

# ---------- entry-level covariates from chains
c2=ch.copy()
for o in ['human','mouse','cow','rabbit','rat','camelid','macaque','synthetic']: c2['o_'+o]=(c2.aborg==o).astype(int)
c2['nonhuman_nonmouse']=(~c2.aborg.isin(['human','mouse'])).astype(int)
c2['longL3']=(c2.L3>=25).astype(int)
agg=c2.groupby('pdb').agg(n_ch=('chain','size'),frac_human=('o_human','mean'),any_mouse=('o_mouse','max'),any_cow=('o_cow','max'),any_rabbit=('o_rabbit','max'),
    any_rat=('o_rat','max'),any_camelid=('o_camelid','max'),any_nhnm=('nonhuman_nonmouse','max'),all_human=('o_human','min'),
    vid_min=('v_id','min'),vid_mean=('v_id','mean'),l1a=('L1_atyp','max'),l2a=('L2_atyp','max'),l3a=('L3_atyp','max'),longL3=('longL3','max'),vhh=('vhh','max'),
    lqz_min=('lq_z','min'),lqz_mean=('lq_z','mean'),any_gap=('any_gap','max'),nterm_trunc=('nterm_trunc','max'),n_gaps=('n_gaps','sum'),
    irm=('irm','first'),fit=('fit','first'),err_art_only=('err_copied_or_artifact_only','max')).reset_index()
m=m.merge(agg,on='pdb',how='left')
m['ln_ch']=np.log2(m.n_ch); m['ln_V']=np.log2(m.n_V_chains)
for c in ['any_gap','nterm_trunc','any_cow','any_rabbit','any_rat','any_nhnm','all_human','any_mouse','vhh','l1a','l2a','l3a','longL3']: m[c]=m[c].astype(float)
m['res_c']=m.res-3.5

# ---------- error-definition variants (run level)
runs['nan_delta']=runs.delta.isna()
L3=ch.set_index(['pdb','chain']).L3
runs['L3']=[L3.get((p,c),np.nan) for p,c in zip(runs.pdb,runs.chain)]
# confirmed numbering artifacts (rare-indel patterns reproduced by SAbR on <=2.5 A X-ray structures; case_files_ABCF lens, read-only)
art=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF/tables/artifact_runs.csv')
artk=set(zip(art.pdb,art.chain,art.imgt0,art.imgt1))
runs['confirmed_artifact']=[(p,c,a,b) in artk for p,c,a,b in zip(runs.pdb,runs.chain,runs.imgt0,runs.imgt1)]
# NOTE: long-CDR3 runs are NOT treated as artifacts: v04 shows SAbR agrees with ANARCI on 15/15 high-res bovine VH incl. ultralong CDR3
runs['canon']=~runs.nan_delta & ~runs.confirmed_artifact
runs['canon_nogap']=runs.canon & ~runs.next_to_gap_origin.astype(bool)
ent_canon=runs.groupby('pdb').canon.max(); ent_canon_ng=runs.groupby('pdb').canon_nogap.max()
ent_real=(~runs.confirmed_artifact).groupby(runs.pdb).max()
m['y_art']=np.where(m.y==0,0.0,np.where(m.pdb.map(ent_real).fillna(False),1.0,np.nan))
m['y_canon']=np.where(m.y==0,0.0,np.where(m.pdb.map(ent_canon).fillna(False),1.0,np.nan))
m['y_canon_nogap']=np.where(m.y==0,0.0,np.where(m.pdb.map(ent_canon_ng).fillna(False),1.0,np.nan))

EM=m[m.method=='EM'].copy()
print('EM entries',len(EM),'errors',int(EM.y.sum()),'| capsid',int(EM.cap.sum()),'capsid errors',int(EM[EM.cap==1].y.sum()))
TESTS=[]
def rate(d,y='y'):
    d=d[d[y].notna()]; return f'{d[y].mean()*100:.1f}% ({int(d[y].sum())}/{len(d)})'
def fitlog(d,form,term,label,group='paper',y='y',record=True,extra=None):
    d=d.dropna(subset=[y]).copy(); d['yy']=d[y]
    vars_=[v for v in re.findall(r'[A-Za-z_][A-Za-z0-9_]*',form) if v in d.columns]
    d=d.dropna(subset=vars_)
    f='yy ~ '+form
    try:
        mod=smf.glm(f,d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[group])[0]})
        k=[x for x in mod.params.index if x==term or x.startswith(term+'[')][0]
        ci=mod.conf_int().loc[k]; res=dict(test=label,term=k,OR=np.exp(mod.params[k]),lo=np.exp(ci[0]),hi=np.exp(ci[1]),p=mod.pvalues[k],
            n=len(d),n_err=int(d.yy.sum()),n_cap=int(d.cap.sum()) if 'cap' in d else np.nan,n_cap_err=int(d[d.cap==1].yy.sum()) if 'cap' in d else np.nan,cluster=group)
    except Exception as ex:
        res=dict(test=label,term=term,OR=np.nan,p=np.nan,n=len(d),n_err=int(d.yy.sum()),note=str(ex)[:80],cluster=group)
    if extra: res.update(extra)
    if record: TESTS.append(res)
    print(f"  {label:95s} OR {res.get('OR',np.nan):6.2f} ({res.get('lo',np.nan):.2f}-{res.get('hi',np.nan):.2f}) p={res.get('p',np.nan):.2g} n={res['n']} err={res['n_err']} cap_err={res.get('n_cap_err')}/{res.get('n_cap')}")
    return res
def fisher(a,b,c,d,label):
    orr,p=stats.fisher_exact([[a,b],[c,d]]); hal=((a+.5)*(d+.5))/((b+.5)*(c+.5))
    TESTS.append(dict(test=label,term='Fisher',OR=hal,p=p,n=a+b+c+d,n_err=a+c)); print(f'  {label:95s} {a}/{a+b} vs {c}/{c+d}  OR(Haldane) {hal:.2f} p={p:.2g}')

print('\n== 1. Reproduction (EM, entry level)')
fitlog(EM,'cap + C(ycat) + res','cap','R1 reproduce: year cat + res, cluster=paper')
print('\n== 2. Clustering by lab network instead of paper')
fitlog(EM,'cap + C(ycat) + res','cap','R2a cluster=lab network (co-author merged)',group='lab')
fitlog(EM,'cap + C(ycat) + res','cap','R2b cluster=wider Rao Z. network',group='lab_rao')
print('\n== 3. Number of V chains per entry (more copies -> more chances)')
print('   n_V_chains capsid vs other EM: mean',EM[EM.cap==1].n_V_chains.mean().round(2),EM[EM.cap==0].n_V_chains.mean().round(2),
      'MW p',stats.mannwhitneyu(EM[EM.cap==1].n_V_chains,EM[EM.cap==0].n_V_chains).pvalue)
fitlog(EM,'cap + ln_V + C(ycat) + res','cap','R3 + log2 n V chains, cluster=lab',group='lab')
print('\n== 4. Period / resolution / SARS-CoV-2 strata (cluster=lab)')
EM['period']=pd.cut(EM.dyear,[0,2019,2021,2100],labels=['<=2019','2020-21','2022+']).astype(str)
for per in ['<=2019','2020-21','2022+']:
    d=EM[EM.period==per]; print('   ',per,'capsid',rate(d[d.cap==1]),'other',rate(d[d.cap==0]))
    fitlog(d,'cap + C(ycat) + res','cap',f'R4 period {per}',group='lab')
EM['rbin']=np.where(EM.res<3.5,'<3.5','>=3.5')
for rb in ['<3.5','>=3.5']:
    d=EM[EM.rbin==rb]; print('   res',rb,'capsid',rate(d[d.cap==1]),'other',rate(d[d.cap==0]))
    fitlog(d,'cap + C(ycat) + res','cap',f'R4 resolution {rb}',group='lab')
d=EM[~EM.sars2.astype(bool)]; fitlog(d,'cap + C(ycat) + res','cap','R4 non-SARS-CoV-2 EM only',group='lab')
d=EM[(~EM.sars2.astype(bool))&(EM.tgt_org=='viral')]; print('   viral non-SARS2: capsid',rate(d[d.cap==1]),'other viral',rate(d[d.cap==0]))
fitlog(d,'cap + C(ycat) + res','cap','R4 vs other viral non-SARS-CoV-2 targets only',group='lab')
print('\n== 5. Antibody source organism (PDB entity annotation)')
print(EM.groupby('cap')[['frac_human','all_human','any_mouse','any_cow','any_rabbit','any_rat','any_camelid','any_nhnm','vhh','vid_min','longL3','l1a','l3a']].mean().round(3).T.to_string())
for nm,d in [('all-human-antibody entries',EM[EM.all_human==1]),('entries with any mouse antibody',EM[EM.any_mouse==1]),('entries with any non-human non-mouse antibody',EM[EM.any_nhnm==1])]:
    print('   ',nm,': capsid',rate(d[d.cap==1]),'other',rate(d[d.cap==0]))
d=EM[EM.all_human==1]; a=int(d[d.cap==1].y.sum()); b=int((d.cap==1).sum())-a; c=int(d[d.cap==0].y.sum()); dd=int((d.cap==0).sum())-c
fisher(a,b,c,dd,'R5a capsid vs other, all-human-antibody EM entries (Fisher)')
fitlog(EM[EM.any_mouse==1],'cap + C(period) + res','cap','R5b capsid vs other, entries with a mouse antibody (period+res)',group='lab')
d=EM[EM.any_nhnm==1]; a=int(d[d.cap==1].y.sum()); b=int((d.cap==1).sum())-a; c=int(d[d.cap==0].y.sum()); dd=int((d.cap==0).sum())-c
fisher(a,b,c,dd,'R5c capsid vs other, entries with a non-human non-mouse antibody (Fisher)')
fitlog(EM,'cap + all_human + any_cow + any_rabbit + any_rat + vhh + C(ycat) + res','cap','R5d + antibody organism indicators + VHH',group='lab')
fitlog(EM,'cap + all_human + any_cow + any_rabbit + any_rat + vhh + vid_min + l1a + l3a + longL3 + C(ycat) + res','cap','R5e + organism + VHH + min germline id + atypical L1/L3 + long L3',group='lab')
print('\n== 6. Unbuilt residues and local map fit')
print(EM.groupby('cap')[['any_gap','nterm_trunc','lqz_min','lqz_mean']].mean().round(3).T.to_string())
fitlog(EM,'cap + any_gap + nterm_trunc + C(ycat) + res','cap','R6a + any V-region gap + N-term truncation',group='lab')
fitlog(EM,'cap + any_gap + nterm_trunc + lqz_min + C(ycat) + res','cap','R6b + gaps + worst-chain lq_z',group='lab')
fitlog(EM,'cap + ln_V + all_human + any_cow + any_rabbit + any_rat + vhh + vid_min + l1a + l3a + longL3 + any_gap + nterm_trunc + lqz_min + C(irm) + C(fit) + C(ycat) + res','cap','R6c full: chains+organism+antibody+gaps+lq_z+start model+fit',group='lab')
print('\n== 7. Error definition: confirmed numbering artifacts; CDR-internal NaN-delta runs')
x=EM[EM.y==1]; print('   EM error entries that are confirmed-artifact-only: capsid',int(x[x.cap==1].y_art.isna().sum()),'of',int((x.cap==1).sum()),'| other',int(x[x.cap==0].y_art.isna().sum()),'of',int((x.cap==0).sum()))
fitlog(EM,'cap + C(ycat) + res','cap','R7-0 confirmed numbering artifacts removed (FR1 gap 26, FR3 gap 84/84+85)',group='lab',y='y_art')
fitlog(EM,'cap + ln_V + all_human + any_cow + any_rabbit + any_rat + vhh + vid_min + l1a + l3a + longL3 + any_gap + nterm_trunc + lqz_min + C(ycat) + res','cap','R7-0b artifacts removed + full covariates',group='lab',y='y_art')
x=EM[EM.y==1]
print('   EM error entries with only non-canonical runs: capsid',int(x[(x.cap==1)].y_canon.isna().sum()),'of',int((x.cap==1).sum()),
      '| other',int(x[(x.cap==0)].y_canon.isna().sum()),'of',int((x.cap==0).sum()))
nc=runs[~runs.canon]; print('   non-canonical first-occurrence runs (all methods):',len(nc),'entries',nc.pdb.nunique(),'of which capsid entries',nc[nc.pdb.isin(set(m[m.cap==1].pdb))].pdb.nunique())
a=int(((x.cap==1)&x.y_canon.isna()).sum()); b=int(((x.cap==1)&x.y_canon.notna()).sum()); c=int(((x.cap==0)&x.y_canon.isna()).sum()); dd=int(((x.cap==0)&x.y_canon.notna()).sum())
fisher(a,b,c,dd,'R7a among EM error entries: share whose errors are only non-canonical, capsid vs other')
fitlog(EM,'cap + C(ycat) + res','cap','R7b artifacts removed AND CDR-internal NaN-delta-only errors dropped',group='lab',y='y_canon')
fitlog(EM,'cap + C(ycat) + res','cap','R7c R7b AND not next to an unbuilt residue',group='lab',y='y_canon_nogap')
fitlog(EM,'cap + ln_V + all_human + any_cow + any_rabbit + any_rat + vhh + vid_min + l1a + l3a + longL3 + any_gap + nterm_trunc + lqz_min + C(ycat) + res','cap','R7d R7b + full covariates',group='lab',y='y_canon')
# artifact-only entries (excluded from the main analysis): are capsid entries also enriched for copied/artifact-only runs?
e_all=pd.read_csv(f'{SP}/deeper/entries.csv'); ae=e_all[(e_all.method=='EM')].copy()
ae['art_only']=(ae.err_any & ~ae.err_first).astype(int); ae['cap']=ae.pdb.isin(set(m[m.cap==1].pdb)).astype(int)
ae=ae[ae.pdb.isin(set(m.pdb))|ae.art_only.astype(bool)]
print('   EM copied/artifact-only entries: capsid',int(ae[ae.cap==1].art_only.sum()),'of',int((ae.cap==1).sum()),' other',int(ae[ae.cap==0].art_only.sum()),'of',int((ae.cap==0).sum()))
print('\n== 8. Paper level (one row per EM paper), cluster=lab network')
P=EM.groupby('paper').agg(y=('y','max'),yc=('y_canon',lambda s: np.nan if s.isna().all() else s.max()),ya=('y_art',lambda s: np.nan if s.isna().all() else s.max()),cap=('cap','max'),lab=('lab','first'),lab_rao=('lab_rao','first'),
    dyear=('dyear','min'),res=('res','median'),n=('y','size'),all_human=('all_human','min'),sars2=('sars2','max')).reset_index()
# papers with an error only via non-canonical runs -> excluded for yc; papers with both canonical and non-canonical -> 1
P['period']=pd.cut(P.dyear,[0,2019,2021,2100],labels=['<=2019','2020-21','2022+']).astype(str); P['ln']=np.log2(P.n)
print('   capsid papers',rate(P[P.cap==1]),'labs',P[P.cap==1].lab.nunique(),'| other',rate(P[P.cap==0]))
print('   capsid error papers by lab network:',P[(P.cap==1)&(P.y==1)].lab.value_counts().to_dict())
fitlog(P,'cap + C(period) + res + ln','cap','R8a paper level, cluster=lab network',group='lab')
fitlog(P,'cap + C(period) + res + ln','cap','R8b paper level, confirmed artifacts removed',group='lab',y='ya')
fitlog(P,'cap + C(period) + res + ln','cap','R8b2 paper level, artifacts and NaN-delta-only removed',group='lab',y='yc')
nets_a=P[(P.cap==1)&(P.ya==1)].lab.value_counts(); print('   capsid error papers after artifact removal, by network:',nets_a.to_dict())
for k in range(1,4):
    drop=list(nets_a.index[:k]); d=P[~P.lab.isin(drop)]
    fitlog(d,'cap + C(period) + res + ln','cap',f'R8c-art paper level, artifacts removed, top {k} networks dropped ({"; ".join(drop)})',group='lab',y='ya')
nets=P[(P.cap==1)&(P.y==1)].lab.value_counts()
for k in range(1,5):
    drop=list(nets.index[:k]); d=P[~P.lab.isin(drop)]
    fitlog(d,'cap + C(period) + res + ln','cap',f'R8c paper level, top {k} capsid-error lab networks dropped ({"; ".join(drop)})',group='lab')
for L in nets.index:
    d=P[P.lab!=L]; fitlog(d,'cap + C(period) + res + ln','cap',f'   LONO paper level without {L}',group='lab',record=False)
d=P[P.all_human==1]; a=int(d[d.cap==1].y.sum()); b=int((d.cap==1).sum())-a; c=int(d[d.cap==0].y.sum()); dd=int((d.cap==0).sum())-c
fisher(a,b,c,dd,'R8d paper level, all-human-antibody papers (Fisher)')
d=P[P.all_human<1]; fitlog(d,'cap + C(period) + res + ln','cap','R8e paper level, papers with any non-human antibody',group='lab')
print('\n== 9. Lab-network level: does each network err more on capsids than on its other antigens? (networks with both)')
N=EM.groupby(['lab','cap']).agg(y=('y','max'),n=('y','size'),papers=('paper','nunique')).reset_index()
both=N.groupby('lab').cap.nunique(); both=both[both==2].index
nb=N[N.lab.isin(both)].pivot(index='lab',columns='cap',values='y')
print('   networks with capsid AND other EM entries:',len(nb),'| error on capsid only',int(((nb[1]==1)&(nb[0]==0)).sum()),'other only',int(((nb[1]==0)&(nb[0]==1)).sum()),'both',int(((nb[1]==1)&(nb[0]==1)).sum()),'neither',int(((nb[1]==0)&(nb[0]==0)).sum()))
b_=int(((nb[1]==1)&(nb[0]==0)).sum()); c_=int(((nb[1]==0)&(nb[0]==1)).sum())
p=stats.binomtest(b_,b_+c_).pvalue if b_+c_>0 else np.nan
TESTS.append(dict(test='R9 within-network McNemar: capsid-only vs other-only error networks',term='McNemar exact',OR=(b_+.5)/(c_+.5),p=p,n=len(nb),n_err=b_+c_)); print('   exact McNemar p',p)
print('   per-network capsid vs other error rates (networks with a capsid error):')
for L in nets.index:
    g=EM[EM.lab==L]; print(f'     {L:40s} capsid {rate(g[g.cap==1])}  other {rate(g[g.cap==0])}')
print('\n== 10. Unit level (antibody-in-paper; EM): chain type, organism, germline, CDR length, nanobody')
u=c2[(c2.method=='EM')].copy(); u['cap']=u.pdb.isin(set(m[m.cap==1].pdb)).astype(int)
u=u[u.pdb.isin(set(EM.pdb))]
U=u.groupby('unit').agg(y=('err_first','max'),cap=('cap','max'),ctype=('ctype','first'),aborg=('aborg',lambda s:s.mode().iat[0]),v_id=('v_id','mean'),
    l1a=('L1_atyp','max'),l3a=('L3_atyp','max'),L3=('L3','max'),vhh=('vhh','max'),paper=('paper','first'),pdb=('pdb','first'),res=('res','median'),deposit=('deposit','min'),
    lqz=('lq_z','min'),any_gap=('any_gap','max'),nterm=('nterm_trunc','max'),ncopies=('chain','size')).reset_index()
U=U.merge(EM[['pdb','lab']],on='pdb',how='left')
U['dyear']=pd.to_datetime(U.deposit).dt.year; U['ycat']=U.dyear.clip(lower=2017).astype(str)
U['org2']=U.aborg.where(U.aborg.isin(['human','mouse','camelid','macaque','synthetic']),'other non-human')
U['longL3']=(U.L3>=25).astype(int); U['y']=U.y.astype(int); U['any_gap']=U.any_gap.astype(float); U['nterm']=U.nterm.astype(float); U['ln_cp']=np.log2(U.ncopies)
print('   units',len(U),'errors',U.y.sum(),'capsid units',U.cap.sum(),'capsid error units',U[U.cap==1].y.sum())
print(pd.crosstab([U.org2],[U.cap,U.y]).to_string())
fitlog(U,'cap + C(ycat) + res','cap','R10a unit level, year+res, cluster=lab',group='lab')
fitlog(U,'cap + C(ctype) + C(ycat) + res','cap','R10b unit level + chain type',group='lab')
fitlog(U,'cap + C(ctype) + C(org2) + C(ycat) + res','cap','R10c unit level + chain type + antibody organism',group='lab')
fitlog(U,'cap + C(ctype) + C(org2) + v_id + l1a + l3a + longL3 + vhh + ln_cp + C(ycat) + res','cap','R10d + germline id + atypical L1/L3 + long L3 + VHH + copies',group='lab')
fitlog(U,'cap + C(ctype) + C(org2) + v_id + l1a + l3a + longL3 + vhh + ln_cp + any_gap + nterm + lqz + C(ycat) + res','cap','R10e + gaps + lq_z',group='lab')
for o in ['human','mouse','other non-human']:
    d=U[U.org2==o]; a=int(d[d.cap==1].y.sum()); b=int((d.cap==1).sum())-a; c=int(d[d.cap==0].y.sum()); dd=int((d.cap==0).sum())-c
    fisher(a,b,c,dd,f'R10f unit level within {o} antibodies: capsid vs other (Fisher)')
for ct in ['H','K','L']:
    d=U[U.ctype==ct]; print(f'   chain type {ct}: capsid',rate(d[d.cap==1]),'other',rate(d[d.cap==0]))
print('\n== 11. Within-lab (conditional logit, lab network strata), journal tier, cap x human interaction')
from statsmodels.discrete.conditional_models import ConditionalLogit
tier=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','tier']).drop_duplicates('pdb')
EM2=EM.merge(tier,on='pdb',how='left')
fitlog(EM2,'cap + C(tier) + C(ycat) + res','cap','R11a + journal tier',group='lab')
for yv,lab_ in [('y','all errors'),('y_art','artifacts removed')]:
    d=EM2.dropna(subset=[yv]).copy()
    v=d.groupby('lab').agg(vy=(yv,'nunique'),vc=('cap','nunique')); keep=v[(v.vy>1)&(v.vc>1)].index; dd=d[d.lab.isin(keep)].copy()
    X=pd.get_dummies(dd[['cap','res']].assign(per=dd.dyear.clip(lower=2019,upper=2023).astype(str)),columns=['per'],drop_first=True).astype(float)
    try:
        cl=ConditionalLogit(dd[yv].astype(int).values,X.values,groups=pd.factorize(dd.lab)[0]).fit(disp=0)
        b=cl.params[0]; se=cl.bse[0]; pv=cl.pvalues[0]
        r=dict(test=f'R11b within-lab conditional logit ({lab_}); informative labs={len(keep)}',term='cap',OR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=pv,n=len(dd),n_err=int(dd[yv].sum()),n_cap=int(dd.cap.sum()),n_cap_err=int(dd[dd.cap==1][yv].sum()))
    except Exception as ex:
        r=dict(test=f'R11b within-lab conditional logit ({lab_})',term='cap',OR=np.nan,p=np.nan,n=len(dd),n_err=int(dd[yv].sum()),note=str(ex)[:80])
    TESTS.append(r); print('  ',r)
    print('     informative labs:',list(keep))
U['human']=(U.org2=='human').astype(int)
fitlog(U,'cap*human + C(ctype) + C(ycat) + res','cap:human','R11c unit level interaction capsid x human antibody (OR of interaction)',group='lab')
# power: how many human capsid error units would OR 4 predict?
d=U[U.human==1]; base=d[d.cap==0].y.mean(); n=int((d.cap==1).sum()); exp_p=4*base/(1-base+4*base)
print(f'   human units: baseline {base:.3f}; capsid human units n={n}, observed errors {int(d[d.cap==1].y.sum())}, expected under OR 4: {n*exp_p:.1f}; P(X<=obs | OR 4) = {stats.binom.cdf(int(d[d.cap==1].y.sum()),n,exp_p):.4f}')
TESTS.append(dict(test='R11d human capsid units: observed vs expected under OR 4 (binomial lower tail)',term='binom',OR=np.nan,p=stats.binom.cdf(int(d[d.cap==1].y.sum()),n,exp_p),n=n,n_err=int(d[d.cap==1].y.sum())))
# BH over this verifier's tests
T=pd.DataFrame(TESTS); ok=T.p.notna(); T.loc[ok,'q']=multipletests(T.loc[ok,'p'],method='fdr_bh')[1]
T.to_csv(f'{OUT}/tests_capsid_confounding.csv',index=False)
print('\n== All tests with BH q'); print(T[['test','OR','lo','hi','p','q','n','n_err','n_cap','n_cap_err']].round(4).to_string())
EM[['pdb','paper','lab','lab_rao','cap','y','y_art','y_canon','y_canon_nogap','n_V_chains','all_human','any_cow','any_rabbit','any_rat','any_gap','nterm_trunc','lqz_min','dyear','res','modes']].to_csv(f'{OUT}/em_entries_verify.csv.gz',index=False)
