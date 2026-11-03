# Adversarial confounding check of the "enzyme_target" finding (antigen_competition lens).
# Inputs are read-only: the lens's focal entry table (antigen class per entry), shared chains.csv, runs, lab experience,
# and the per-chain unbuilt-residue features from v01_chain_gaps.py.
import json, re, warnings, collections
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
AC='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_confounding'
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)

m=pd.read_csv(f'{AC}/focal_entries_antigen_features.csv.gz').merge(pd.read_csv(f'{AC}/lab_experience.csv'),on='pdb',how='left')
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz')
G=pd.read_csv(f'{SP}/deeper/cache/verify_enzyme_confounding/chain_gaps.csv.gz')
uni=json.load(open(f'{SP}/deeper/cache/antigen_competition/uniprot_info.json'))
ch=ch.merge(G,on=['pdb','chain'],how='left')
ch['mega']=ch.chain_desc.fillna('').str.contains('megabody',case=False).astype(int)
ch['atyp']=ch[['L1_atyp','L2_atyp','L3_atyp']].max(axis=1)
ch['mouse']=(ch.species=='mouse').astype(float)
for c in ['any_gap','nterm_trunc','n_gaps']: ch[c]=ch[c].fillna(0).astype(float)
agg=ch.groupby('pdb').agg(vhh=('vhh','max'),mega=('mega','max'),mouse=('mouse','mean'),v_id=('v_id','mean'),lq_z=('lq_z','mean'),
    irm=('irm','first'),fit=('fit','first'),atyp=('atyp','max'),light=('ctype',lambda s:s.isin(['K','L']).any()),
    gap_frac=('any_gap','mean'),gap_any=('any_gap','max'),nterm_any=('nterm_trunc','max'),n_gaps=('n_gaps','sum'),n_ch=('chain','size')).reset_index()
m=m.merge(agg,on='pdb',how='left')
m['enz_host']=(m.tgt_cls=='enzyme').astype(int); m['enz_vir']=(m.tgt_cls=='viral other').astype(int); m['enz']=m.enz_host|m.enz_vir
m['cap']=(m.tgt_cls=='viral capsid').astype(int)
# alternative, rule-order-free enzyme definition: any target accession with an EC number in UniProt
def has_ec(keys):
    if not isinstance(keys,str): return 0
    return int(any(bool((uni.get(k) or {}).get('ec')) for k in re.split(r'[|;,]',keys)))
m['enz_ec']=m.tgt_keys.apply(has_ec)
m['res_bin']=pd.cut(m.res,[0,3.25,3.75,4.25,9],labels=['<=3.25','3.25-3.75','3.75-4.25','>4.25']).astype(str)
m['lexp']=np.log2(1+m.lab_prior_ab.fillna(0))
m['nonfid_bound']=(m.ab_status2=='bound').astype(int)
EM=m[m.method=='EM'].copy(); XR=m[m.method=='X-ray'].copy()
print('analysis entries',len(m),'EM',len(EM),'EM errors',EM.y.sum())
TESTS=[]
def rate(d): return f'{d.y.mean()*100:.1f}% ({int(d.y.sum())}/{len(d)})'
def logit(d,term,extra='C(ycat) + res',label='',cl='paper'):
    d=d.copy()
    a=int(((d[term]==1)&(d.y==1)).sum()); b=int(((d[term]==1)&(d.y==0)).sum()); c_=int(((d[term]==0)&(d.y==1)).sum()); dd=int(((d[term]==0)&(d.y==0)).sum())
    row=dict(test=label,n=len(d),n_err=int(d.y.sum()),rate_in=f'{a}/{a+b} ({100*a/max(a+b,1):.1f}%)',rate_out=f'{c_}/{c_+dd} ({100*c_/max(c_+dd,1):.1f}%)',cluster=cl)
    if a<3:
        p=stats.fisher_exact([[a,b],[c_,dd]])[1]; row.update(OR=((a+.5)*(dd+.5))/((b+.5)*(c_+.5)),lo=np.nan,hi=np.nan,p=p,model='Fisher (unadjusted)')
        TESTS.append(row); return row
    try:
        mod=smf.glm(f'y ~ {term} + {extra}',d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[cl])[0]})
        ci=mod.conf_int().loc[term]
        row.update(OR=np.exp(mod.params[term]),lo=np.exp(ci[0]),hi=np.exp(ci[1]),p=mod.pvalues[term],model=f'logit {extra}')
    except Exception as ex:
        row.update(OR=np.nan,p=np.nan,model=f'failed {ex}')
    TESTS.append(row); return row
def show(r): print(f"  {r['test'][:95]:95s} {r['rate_in']:>16s} vs {r['rate_out']:>18s} OR {r['OR']:.2f} ({r.get('lo',np.nan):.2f}-{r.get('hi',np.nan):.2f}) p={r['p']:.2g}")

print('\n== 1. Reproduce (EM entries, year+res, paper-clustered)')
for t,lab in [('enz_host','host enzyme vs all other EM'),('enz_vir','viral enzyme vs all other EM'),('enz','host+viral enzyme vs all other EM')]:
    show(logit(EM,t,label=f'1 reproduce: {lab}'))

print('\n== 2. Control choice (EM entries)')
D=EM[EM.ab_status2=='bound']; show(logit(D,'enz',label='2a enzyme vs other antigen-bound entries (fiducial-only/unbound/no-contact removed)'))
D=EM[~EM.sars2]; show(logit(D,'enz',label='2b enzyme vs other non-SARS-CoV-2 EM entries'))
D=EM[(EM.ab_status2=='bound')&~EM.sars2]; show(logit(D,'enz',label='2c enzyme vs non-SARS2 antigen-bound entries'))
D=EM[(EM.ab_status2=='bound')&~EM.sars2&(EM.cap==0)]; show(logit(D,'enz',label='2d enzyme vs non-SARS2 antigen-bound, capsid removed'))
nonvir=EM[(EM.ab_status2=='bound')&(EM.tgt_org!='viral')]
show(logit(nonvir,'enz_host',label='2e host enzyme vs other non-viral protein targets (bound)'))
print('   rates by class among non-viral bound EM entries:',{k:rate(g) for k,g in nonvir.groupby('tgt_cls') if len(g)>=10})
show(logit(EM,'enz',label='2f enzyme, lab-clustered SEs (all EM)',cl='last_author'))

print('\n== 3. Period x resolution stratification (EM entries)')
for per,g in EM.groupby('period'): print(f'   {per}: enzyme {rate(g[g.enz==1])} vs other {rate(g[g.enz==0])}; host enzyme {rate(g[g.enz_host==1])}')
def mh(D,term,strata,label):
    tabs=[]
    for k,g in D.groupby(strata):
        a=((g[term]==1)&(g.y==1)).sum(); b=((g[term]==1)&(g.y==0)).sum(); c_=((g[term]==0)&(g.y==1)).sum(); d_=((g[term]==0)&(g.y==0)).sum()
        if (a+b)>0 and (c_+d_)>0: tabs.append(np.array([[a,b],[c_,d_]],dtype=float))
    st=StratifiedTable(tabs); p=st.test_null_odds(correction=True).pvalue; lo,hi=st.oddsratio_pooled_confint()
    row=dict(test=label,n=len(D),n_err=int(D.y.sum()),OR=st.oddsratio_pooled,lo=lo,hi=hi,p=p,model=f'Mantel-Haenszel over {strata}',rate_in=f'{int(sum(t[0,0] for t in tabs))}/{int(sum(t[0].sum() for t in tabs))}',rate_out=f'{int(sum(t[1,0] for t in tabs))}/{int(sum(t[1].sum() for t in tabs))}',cluster='none')
    TESTS.append(row); show(row); return row
mh(EM,'enz',['period','res_bin'],'3a enzyme, MH over period x resolution bin (entries; ignores clustering)')
for per in ['<=2019','2020-21','2022+']:
    D=EM[EM.period==per]; show(logit(D,'enz',extra='C(ycat) + res' if per!='<=2019' else 'res',label=f'3b enzyme within period {per}'))

print('\n== 4. Paper level (one unit per paper; EM)')
def paper_tab(D,flag):
    P=D.groupby('paper').agg(y=('y','max'),f=(flag,'max'),lab=('last_author','first'),n=('y','size'),ycat=('ycat',lambda s:s.mode().iat[0]),
        period=('period',lambda s:s.mode().iat[0]),res=('res','median'),sars2=('sars2','max'),bound=('nonfid_bound','max'),
        vhh=('vhh','max'),gap=('gap_any','max'),nterm=('nterm_any','max'),irm=('irm','first'),fit=('fit','first'),lexp=('lexp','min'),mega=('mega','max')).reset_index()
    P['ln']=np.log2(P.n); P=P.rename(columns={'f':flag}); return P
P=paper_tab(EM,'enz')
print('   enzyme papers',int(P.enz.sum()),'error',int(P[P.enz==1].y.sum()),'| other papers',int((P.enz==0).sum()),'error',int(P[P.enz==0].y.sum()))
show(logit(P,'enz',extra='C(ycat) + res + ln',label='4a paper level: enzyme (any entry) vs other EM papers'))
show(logit(P,'enz',extra='C(ycat) + res + ln',label='4b paper level, lab-clustered',cl='lab'))
show(logit(P[P.bound==1],'enz',extra='C(ycat) + res + ln',label='4c paper level, antigen-bound papers only'))
show(logit(P[(P.bound==1)&~P.sars2.astype(bool)],'enz',extra='C(ycat) + res + ln',label='4d paper level, antigen-bound non-SARS2 papers'))
for per,g in P.groupby('period'): print(f'   paper level {per}: enzyme {rate(g[g.enz==1])} vs other {rate(g[g.enz==0])}')
mh(P,'enz',['period'],'4e paper level, MH over period')
Ph=paper_tab(EM,'enz_host'); show(logit(Ph,'enz_host',extra='C(ycat) + res + ln',label='4f paper level: host enzyme only'))
# lab-collapsed: one row per lab x class (lab has any error in its enzyme / non-enzyme papers)
L=P.groupby(['lab','enz']).agg(y=('y','max'),n=('y','size'),res=('res','median'),ycat=('ycat',lambda s:s.mode().iat[0])).reset_index(); L['ln']=np.log2(L.n)
show(logit(L,'enz',extra='res + ln',label='4g lab x class level (one row per lab and class), adjusted for n papers',cl='lab'))

print('\n== 5. Cluster bootstrap over labs (paper-level OR, year+res+size)')
rng=np.random.default_rng(7); labs=P.lab.unique(); byl={l:g for l,g in P.groupby('lab')}; ors=[]
for i in range(400):
    s=rng.choice(labs,len(labs),replace=True); B=pd.concat([byl[l] for l in s],ignore_index=True)
    if B[B.enz==1].y.sum()<2: ors.append(np.nan); continue
    try:
        mod=smf.glm('y ~ enz + C(period) + res + ln',B,family=sm.families.Binomial()).fit(); ors.append(np.exp(mod.params['enz']))
    except Exception: ors.append(np.nan)
ors=np.array(ors); ok=ors[np.isfinite(ors)&(ors<1e3)]
print(f'   bootstrap reps {len(ok)}; median OR {np.median(ok):.2f}; 95% interval {np.percentile(ok,2.5):.2f}-{np.percentile(ok,97.5):.2f}; P(OR<=1)={np.mean(ok<=1):.3f}')
TESTS.append(dict(test='5 paper-level enzyme OR, lab cluster bootstrap (period+res+size)',OR=np.median(ok),lo=np.percentile(ok,2.5),hi=np.percentile(ok,97.5),p=2*min(np.mean(ok<=1),0.5)+1/len(ok),n=len(P),n_err=int(P.y.sum()),model='lab bootstrap'))

print('\n== 6. Simpler known factors (EM entries): unbuilt residues, map fit, nanobody/megabody, starting model, fitting, atypical CDRs, lab experience')
for v in ['gap_any','gap_frac','nterm_any','n_gaps','lq_z','vhh','mega','atyp','v_id','mouse','lexp']:
    a=EM[EM.enz==1][v].dropna(); b=EM[EM.enz==0][v].dropna()
    print(f'   {v:10s} enzyme {a.mean():.3f} vs other {b.mean():.3f}  p={stats.mannwhitneyu(a,b).pvalue:.2g}')
print('   error rate by gap_any x enzyme:'); print(EM.groupby(['enz','gap_any']).y.agg(['mean','sum','size']).round(3).to_string())
d=EM.dropna(subset=['lq_z','v_id','gap_frac']).copy()
for f,lab in [('C(ycat) + res','base (same sample)'),
              ('gap_any + nterm_any + C(ycat) + res','+ unbuilt residues (any gap, N-terminal truncation)'),
              ('gap_any + nterm_any + lq_z + C(ycat) + res','+ unbuilt + local map fit'),
              ('gap_any + nterm_any + lq_z + vhh + mega + atyp + v_id + mouse + C(irm) + C(fit) + C(ycat) + res','+ unbuilt + map + VHH + megabody + atypical CDR + germline id + mouse + starting model + fit'),
              ('gap_any + nterm_any + lq_z + vhh + mega + atyp + v_id + mouse + C(irm) + C(fit) + lexp + sars2 + cap + C(ycat) + res','full + lab experience + SARS2 + capsid')]:
    show(logit(d,'enz',extra=f,label=f'6 enzyme adjusted: {lab}'))
show(logit(d,'enz',extra='gap_any + nterm_any + lq_z + vhh + mega + atyp + v_id + mouse + C(irm) + C(fit) + lexp + sars2 + cap + C(ycat) + res',label='6 enzyme full model, lab-clustered',cl='last_author'))

print('\n== 7. Chain-type / antibody-unit level (unit = antibody V region in one paper; EM chains)')
c=ch[(ch.method=='EM')&~ch.err_copied_or_artifact_only].merge(m[['pdb','enz','enz_host','cap','ab_status2','ycat','period','tgt_cls']],on='pdb',how='inner')
c['kind']=np.where(c.vhh==1,'VHH',np.where(c.ctype=='H','Fab heavy','light'))
U=c.groupby('unit').agg(y=('err_first','max'),enz=('enz','max'),kind=('kind','first'),paper=('paper','first'),lab=('last_author','first'),res=('res','median'),
    ycat=('ycat',lambda s:s.mode().iat[0]),gap=('any_gap','max'),nterm=('nterm_trunc','max'),bound=('ab_status2',lambda s:(s=='bound').any()),
    mega=('mega','max'),atyp=('atyp','max'),v_id=('v_id','mean'),sars2=('sars2','max')).reset_index()
U['y']=U.y.astype(int)
print('   units',len(U),'error units',U.y.sum(),'| enzyme units',int(U.enz.sum()),'error',int(U[U.enz==1].y.sum()))
print(U.groupby(['kind','enz']).y.agg(['mean','sum','size']).round(3).to_string())
show(logit(U,'enz',extra='C(kind) + C(ycat) + res',label='7a unit level, adjusted for chain kind'))
for k in ['VHH','Fab heavy','light']:
    show(logit(U[U.kind==k],'enz',extra='C(ycat) + res',label=f'7b unit level within {k}'))
show(logit(U,'enz',extra='C(kind) + gap + nterm + mega + atyp + v_id + C(ycat) + res',label='7c unit level, + chain kind + unbuilt + megabody + atypical CDR + germline id'))
show(logit(U[U.bound],'enz',extra='C(kind) + C(ycat) + res',label='7d unit level, antigen-bound only, + chain kind'))

print('\n== 8. Alternative enzyme definition (any target accession with an EC number, regardless of rule order)')
print('   crosstab tgt_cls-enzyme vs EC-based:'); print(pd.crosstab(EM.enz,EM.enz_ec))
print('   EC-positive entries outside the enzyme class, by class:',EM[(EM.enz_ec==1)&(EM.enz==0)].tgt_cls.value_counts().to_dict())
show(logit(EM,'enz_ec',label='8a EC-based enzyme (any class) vs other EM'))
X=EM[(EM.enz_ec==1)&(EM.enz==0)]; print('   EC-positive but not enzyme class:',rate(X))
EM['enz_ec_only']=((EM.enz_ec==1)&(EM.enz==0)).astype(int)
show(logit(EM[EM.enz==0],'enz_ec_only',label='8b EC-positive targets classed elsewhere (e.g. membrane enzymes) vs other non-enzyme EM'))

print('\n== 9. Leave-one-paper-out and leave-k-labs-out (paper level)')
errp=P[(P.enz==1)&(P.y==1)]
print('   enzyme error papers:',len(errp)); print(errp[['paper','lab','n','period']].to_string())
lo=[]
for pp in errp.paper:
    r=smf.glm('y ~ enz + C(ycat) + res + ln',P[P.paper!=pp],family=sm.families.Binomial()).fit(); lo.append(np.exp(r.params['enz']))
print(f'   leave-one-error-paper-out OR range {min(lo):.2f}-{max(lo):.2f}')
labs_by=errp.lab.value_counts()
for k in [2,3,4,6]:
    drop=list(labs_by.index[:k]); D=P[~P.lab.isin(drop)]
    show(logit(D,'enz',extra='C(ycat) + res + ln',label=f'9 paper level, top-{k} enzyme-error labs removed ({", ".join(drop)})'))
# treat same-lab repeats of the same antibody as one paper (Zimmer Nb881 2021/2023; Naismith megabody 2025/2026)
P2=P.copy(); P2.loc[P2.lab=='Zimmer, J.','paper']='Zimmer'; P2.loc[P2.lab=='Naismith, J.H.','paper']='Naismith'
P2=P2.groupby('paper').agg(y=('y','max'),enz=('enz','max'),lab=('lab','first'),n=('n','sum'),ycat=('ycat','first'),res=('res','median'),period=('period','first')).reset_index(); P2['ln']=np.log2(P2.n)
show(logit(P2,'enz',extra='C(ycat) + res + ln',label='9b paper level, same-lab reused antibody papers merged (Zimmer Nb881, Naismith megabody)'))

R=pd.DataFrame(TESTS)
R['q_BH']=multipletests(R.p.fillna(1),method='fdr_bh')[1]
R.to_csv(f'{OUT}/tests_enzyme_confounding.csv',index=False)
EM.to_csv(f'{OUT}/em_entries_verify.csv.gz',index=False)
print('\nall tests:'); print(R[['test','rate_in','rate_out','OR','lo','hi','p','q_BH','n','n_err']].round(4).to_string())
