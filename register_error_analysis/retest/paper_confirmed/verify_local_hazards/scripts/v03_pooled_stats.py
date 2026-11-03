"""Own-code pooled statistics for the 'unbuilt nearby / N-term unbuilt' hazard (both the investigator's SAbR-label
definition, taken from lh_frame, and the independent alignment definition from v02), bottom-quartile fit, and 'either'.
MH OR (RGB CI), one case per paper, paper-cluster bootstrap, heterogeneity vs rest, control-set variants, and
leave-one-paper-out. Output: tables/v03.log, tables/v03_pooled.csv"""
import numpy as np, pandas as pd, json
from scipy.stats import norm, chi2, fisher_exact, binomtest
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_local_hazards/tables'
LH='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/local_hazards/tables'
LED='/home/user/SAbR/register_error_analysis/retest/deeper/synthesis/artifact_ledger_runs.csv'
log=open(f'{OUT}/v03.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); log.write(s+'\n')
MODEMAP={'A_FR1Nterm':'A','F_CDR1':'F','B_CDR2FR3':'B','C_FR3':'C','Dplus_CDR3short':'D+','Dminus_CDR3long':'D-','E_FR4bulge':'E'}
WIN={'A':(1,14),'F':(20,45),'B':(50,75),'C':(71,100),'D+':(104,124),'D-':(104,124),'E':(115,128)}
STRICT={'amyloid_fibril','indel_gap_artifact_confirmed','DE_chain_end_artifact','near_copy','not_a_register_shift'}
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
C=C.merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz'),on=['pdb','chain'])
C['light']=C.ctype.isin(['K','L']); C['EM']=C.method.eq('EM')
G=pd.read_pickle(f'{OUT}/v02_gap_frame.pkl.gz'); C=C.merge(G,on=['pdb','chain'],how='left')
LHF=pd.read_pickle(f'{LH}/lh_frame.pkl.gz')[['role','pdb','chain','mode','imgt0','imgt1','unmod_near','nterm_missing','win_fit_pct']]
# ---- case runs
R=pd.read_csv(f'{SP}/deeper/runs_all.csv'); R=R.merge(pd.read_csv(LED)[['pdb','chain','imgt0','imgt1','flag']],on=['pdb','chain','imgt0','imgt1'],how='left')
R['m']=R['mode'].map(MODEMAP); R=R[R.m.notna()]
V=pd.read_csv(f'{OUT}/v01_cases_rederived.csv')
st=V[V.n_overlap_runs>0][['pdb','chain','imgt0','imgt1','status_v']].drop_duplicates(['pdb','chain','imgt0','imgt1'])
R=R.merge(st,on=['pdb','chain','imgt0','imgt1'],how='left')
R['first_clean']=(~R.copied_or_artifact.astype(bool))&~R.flag.isin(STRICT)
R=R.drop(columns=['paper','sars2','year']).merge(C[['pdb','chain','paper','vseq','unit','EM','light','in_paper_window','paper_control','release','year','sars2']+[c for c in G.columns if c not in('pdb','chain')]],on=['pdb','chain'],how='left')
pu=set(R[R.status_v.notna()&(R.status_v!='not confirmed by SAbR/ANARCI')].unit)
sets={'strict':R[R.status_v.eq('confirmed first occurrence')],
      'incl':R[R.status_v.isin(['confirmed first occurrence','confirmed, loose-ledger doubt','copy of an earlier model','near copy (ledger)'])],
      'rest':R[R.first_clean&~R.unit.isin(pu)]}
sets['rest_win']=sets['rest'][sets['rest'].in_paper_window]
def featurise(d,m_col):
    d=d.copy()
    d['gap_v']=[(r[f'unb_{m}']>0) or (m=='A' and r.first_built_v>3) for m,(_,r) in zip(d[m_col],d.iterrows())]
    return d
for k in sets:
    s=featurise(sets[k],'m').merge(LHF[LHF.role=='case'].drop(columns='role'),left_on=['pdb','chain','m','imgt0','imgt1'],right_on=['pdb','chain','mode','imgt0','imgt1'],how='left',suffixes=('','_lh'))
    s['gap_lh']=(s.unmod_near.fillna(0)>0)|((s.m=='A')&s.nterm_missing.fillna(False).astype(bool))
    s['case']=1; sets[k]=s
# ---- controls (one row per chain x mode)
def controls(mask):
    k=C[mask]; rows=[]
    for m in WIN:
        x=k.copy(); x['m']=m; rows.append(x)
    x=featurise(pd.concat(rows,ignore_index=True),'m'); x['case']=0
    lh=LHF[LHF.role=='control'].drop(columns=['role','imgt0','imgt1'])
    x=x.merge(lh,left_on=['pdb','chain','m'],right_on=['pdb','chain','mode'],how='left')
    x['gap_lh']=(x.unmod_near.fillna(0)>0)|((x.m=='A')&x.nterm_missing.fillna(False).astype(bool))
    return x
s1=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx'); s1_entries=set(s1.PDB_CHAIN_ID.str.split(':').str[0].str.lower())
CTL={'paper':controls(C.paper_control),'full':controls(~C.err_any.astype(bool))}
# variant: in window, entry nowhere in Table S1, no run overlapping the mode window +-5 (deepdive-style), any other runs allowed
Rall=pd.read_csv(f'{SP}/deeper/runs_all.csv')
x=controls(C.in_paper_window&~C.pdb.isin(s1_entries))
bad={m:set(zip(*Rall[(Rall.imgt0<=hi+5)&(Rall.imgt1>=lo-5)][['pdb','chain']].values.T)) for m,(lo,hi) in WIN.items()}
x=x[[ (p,c) not in bad[m] for p,c,m in zip(x.pdb,x.chain,x.m)]]
CTL['win_local']=x
P('paper_control chains in entries that ARE in Table S1 (error elsewhere, e.g. antigen or <50% V):',int(C[C.paper_control].pdb.isin(s1_entries).sum()),'of',int(C.paper_control.sum()))
P('in-window chains excluded from paper controls because of a non-case run (err_any, not a Table S1 entry):',int((C.in_paper_window&~C.paper_flagged_entry&C.err_any.astype(bool)).sum()))
def units(d): return d.sort_values(['pdb','chain']).drop_duplicates(['m','paper','vseq'])
def mh(df,x):
    num=den=0; sP=sQR=sS=0; O=E=Vv=0; tabs=[]
    for s,g in df.groupby('stratum'):
        a=((g.case==1)&g[x]).sum(); b=((g.case==1)&~g[x]).sum(); c=((g.case==0)&g[x]).sum(); d=((g.case==0)&~g[x]).sum(); n=a+b+c+d
        if a+b==0 or c+d==0: continue
        tabs.append((a,b,c,d))
        R_=a*d/n; S_=b*c/n; num+=R_; den+=S_
        Pp=(a+d)/n; Q=(b+c)/n; sP+=Pp*R_; sQR+=(Pp*S_+Q*R_); sS+=Q*S_
        O+=a; E+=(a+b)*(a+c)/n; Vv+=(a+b)*(c+d)*(a+c)*(b+d)/(n*n*(n-1)) if n>1 else 0
    if den==0 or num==0: return dict(OR=np.nan,lo=np.nan,hi=np.nan,p=np.nan,tabs=tabs)
    OR=num/den; se=np.sqrt(sP/(2*num**2)+sQR/(2*num*den)+sS/(2*den**2))
    p=chi2.sf((abs(O-E)-0.5)**2/Vv,1) if Vv>0 else np.nan
    return dict(OR=OR,lo=OR*np.exp(-1.96*se),hi=OR*np.exp(1.96*se),p=p,se=se,tabs=tabs)
def std_pct(tabs):
    nc=sum(a+b for a,b,c,d in tabs); return 100*sum(a for a,b,c,d in tabs)/nc, 100*sum((a+b)*c/(c+d) for a,b,c,d in tabs)/nc, nc
def boot(df,x,B=2000,seed=1):
    rng=np.random.default_rng(seed)
    df=df.copy(); df['cell']=df.case.astype(str)+np.where(df[x],'1','0')
    tab=df.groupby(['paper','stratum','cell']).size().unstack(fill_value=0)
    for c in ['11','10','01','00']:
        if c not in tab: tab[c]=0
    tab=tab.reset_index(); papers=tab.paper.unique(); idx={p:i for i,p in enumerate(papers)}; tab['pi']=tab.paper.map(idx)
    strata=tab.stratum.unique(); si={s:i for i,s in enumerate(strata)}; tab['si']=tab.stratum.map(si)
    M=np.zeros((len(papers),len(strata),4))
    for r in tab.itertuples(): M[r.pi,r.si]=[r._asdict()['_3'] if False else getattr(r,'_'+str(list(tab.columns).index('11')+1)),0,0,0]
    # simpler fill
    M=np.zeros((len(papers),len(strata),4))
    for _,r in tab.iterrows(): M[r.pi,r.si]=[r['11'],r['10'],r['01'],r['00']]
    out=[]
    for _ in range(B):
        w=rng.multinomial(len(papers),np.ones(len(papers))/len(papers))
        T=np.tensordot(w,M,axes=1); a,b,c,d=T[:,0],T[:,1],T[:,2],T[:,3]; n=T.sum(1); ok=(a+b>0)&(c+d>0)
        num=(a*d/np.where(n>0,n,1))[ok].sum(); den=(b*c/np.where(n>0,n,1))[ok].sum()
        if num>0 and den>0: out.append(np.log(num/den))
    out=np.array(out); return np.exp(np.percentile(out,[2.5,97.5])), len(out)/B
def test(cset,ctl,x,meth,log_=True,opp=False,extra=''):
    cs=sets[cset]; k=CTL[ctl]
    if meth=='EM': cs=cs[cs.EM==True]; k=k[k.EM==True]
    elif meth=='X-ray': cs=cs[cs.EM==False]; k=k[k.EM==False]
    if x=='poor': cs=cs[cs.win_fit_pct.notna()]; k=k[k.win_fit_pct.notna()]
    cu=units(cs); ku=units(k)
    if opp: cu=cu.sort_values(['pdb','chain']).drop_duplicates('paper')
    d=pd.concat([cu,ku]); d['stratum']=d.m+np.where(d.light,'_L','_H'); d=d[d.stratum.isin(set(d[d.case==1].stratum))]
    d[x]=d[x].astype(bool)
    r=mh(d,x); pc,pk,nc=std_pct(r['tabs'])
    r.update(n_case=nc,n_ctrl=int((d.case==0).sum()),pct_case=pc,pct_ctrl=pk,n_papers=d[d.case==1].paper.nunique(),d=d)
    return r
def fmt(r): return f"n={r['n_case']} ({r['n_papers']} papers) {r['pct_case']:.1f}% vs {r['pct_ctrl']:.1f}% (ctrl units {r['n_ctrl']}) MH OR {r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.2g}"
# bottom-quartile threshold from full controls, per mode x method (as investigator), applied to all
fc=CTL['full']; fc['meth']=np.where(fc.EM,'EM','X')
q=fc.groupby(['m','meth']).win_fit_pct.quantile(0.25)
def add_poor(d):
    d['meth']=np.where(d.EM,'EM','X'); d['poor']=[ (v<q.get((m,me),np.nan)) if pd.notna(v) else False for v,m,me in zip(d.win_fit_pct,d.m,d.meth)]
    d['either_lh']=d.gap_lh|d.poor; d['either_v']=d.gap_v|d.poor; return d
for k in sets: sets[k]=add_poor(sets[k])
for k in CTL: CTL[k]=add_poor(CTL[k])
rows=[]
P('\n=== agreement of gap definitions on case runs (strict):'); s=sets['strict']; P(pd.crosstab(s.gap_lh,s.gap_v).to_string())
s=sets['rest']; P('rest:'); P(pd.crosstab(s.gap_lh,s.gap_v).to_string())
k=CTL['paper']; P('paper controls (chain x mode):'); P(pd.crosstab(k.gap_lh,k.gap_v).to_string())
for x in ['gap_lh','gap_v','poor','either_lh','either_v']:
    P(f'\n######## {x}')
    for meth in ['EM','X-ray','all']:
        for cset,ctl in [('strict','paper'),('incl','paper'),('rest','full'),('rest_win','paper'),('strict','win_local'),('strict','full')]:
            r=test(cset,ctl,x,meth)
            if r['n_case']==0: continue
            row=dict(feature=x,method=meth,cases=cset,controls=ctl,**{k:r[k] for k in ['n_case','n_papers','n_ctrl','pct_case','pct_ctrl','OR','lo','hi','p']})
            if cset=='strict' and ctl=='paper':
                ro=test(cset,ctl,x,meth,opp=True); row.update(opp_n=ro['n_case'],opp_OR=ro['OR'],opp_lo=ro['lo'],opp_hi=ro['hi'],opp_p=ro['p'])
                (bl,bh),okf=boot(r['d'],x); row.update(boot_lo=bl,boot_hi=bh,boot_ok=okf)
                rr=test('rest','full',x,meth); z=(np.log(r['OR'])-np.log(rr['OR']))/np.sqrt(r['se']**2+rr['se']**2) if np.isfinite(r['OR']) and np.isfinite(rr['OR']) else np.nan
                row.update(het_p=2*norm.sf(abs(z)))
                # leave-one-paper-out
                d=r['d']; cp=d[d.case==1].paper.value_counts()
                lo_or=[]
                for pp in cp.index:
                    rr2=mh(d[~((d.case==1)&(d.paper==pp))],x); lo_or.append((pp,cp[pp],rr2['OR'],rr2['lo'],rr2['p']))
                row['lopo_min']=min(o[2] for o in lo_or); row['lopo_maxp']=max(o[4] for o in lo_or)
                top=sorted(lo_or,key=lambda o:-o[1])[:3]
                P(f'  [{meth}] top papers by units:',[(o[0][-25:],o[1],round(o[2],2),round(o[3],2),f"{o[4]:.2g}") for o in top])
            rows.append(row)
            P(f'{meth:5s} {cset:8s} vs {ctl:9s}', fmt(r), (f"| opp n={row['opp_n']} OR {row['opp_OR']:.2f} ({row['opp_lo']:.2f}-{row['opp_hi']:.2f}) p={row['opp_p']:.2g} | paper-boot CI {row['boot_lo']:.2f}-{row['boot_hi']:.2f} | het vs rest p={row['het_p']:.2f} | LOPO min OR {row['lopo_min']:.2f}, max p {row['lopo_maxp']:.2g}" if 'opp_n' in row else ''))
pd.DataFrame(rows).to_csv(f'{OUT}/v03_pooled.csv',index=False)
# strict EM case units: listing
s=units(sets['strict'][sets['strict'].EM==True])
P('\nstrict EM case units:'); P(s[['pdb','chain','m','imgt0','imgt1','paper','gap_lh','gap_v','unb_'+'A','first_built_v','poor']].assign(paper=s.paper.str[-22:]).to_string(index=False))
s=units(sets['strict'][sets['strict'].EM==False])
P('\nstrict X-ray case units:'); P(s[['pdb','chain','m','imgt0','imgt1','paper','year','gap_lh','gap_v','poor','win_fit_pct']].assign(paper=s.paper.str[-22:]).to_string(index=False))
log.close()
