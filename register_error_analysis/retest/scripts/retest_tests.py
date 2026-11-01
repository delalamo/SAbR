import json, re, collections, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from statsmodels.discrete.conditional_models import ConditionalLogit
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',70)
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
M=pd.read_pickle(f'{SP}/retest/master.pkl')
M['single']=M.ctype.isin(['H','K','L']) if 'ctype' in M else M.chain_type.isin(['H','K','L'])
M['ctype']=M.chain_type.where(M.chain_type.isin(['H','K','L']),'multi')
M['logV']=np.log(M.nV); M['new']=M.era.str.startswith('new').astype(int)
S=M[M.single&M.res.notna()].copy()
def fit(f,d):
    try: return smf.logit(f,data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.pdb)[0]})
    except Exception as e: print('   (not estimable:',str(e)[:70],')'); return None
def ors(m,filt):
    if m is None: return pd.DataFrame()
    p=m.params[m.params.index.str.contains(filt)]; c=m.conf_int().loc[p.index]
    return pd.DataFrame({'OR':np.exp(p),'lo':np.exp(c[0]),'hi':np.exp(c[1]),'p':m.pvalues[p.index]}).round(3)
BASE="C(method)*C(rb) + C(ybin) + C(ctype) + logV"
def rate(x): return f'{100*x.mean():.1f}% (n={len(x)})'
def H(t): print('\n'+'='*100+'\n'+t+'\n'+'='*100)
# ---------------------------------------------------------------- T2/T17 timing
H('T2/T17. Timing of the EM decline (half-years) — does it line up with AlphaFold2 (Jul 2021)?')
E=M[(M.method=='EM')&M.single].copy(); d=pd.to_datetime(E.deposit); E['half']=d.dt.year.astype(str)+np.where(d.dt.month<=6,'H1','H2')
print(E[E.half>='2018'].groupby('half').flag.agg(rate).to_string())
# ---------------------------------------------------------------- T3 decomposition
H('T3. How much of each change is composition (method x resolution mix) vs within-stratum change?')
S['period']=np.select([S.year<=2019,S.year<=2021],['≤2019','2020–21'],'2022–26')
for a,b in [('≤2019','2020–21'),('2020–21','2022–26')]:
    s1=S[S.period==a]; s2=S[S.period==b]
    w1=s1.groupby(['method','rb']).size()/len(s1); r2=s2.groupby(['method','rb']).flag.mean()
    r2s=sum(w*r2.get(k,s2.flag.mean()) for k,w in w1.items())
    print(f'{a} -> {b}: {100*s1.flag.mean():.2f}% -> {100*s2.flag.mean():.2f}%  (EM share {100*(s1.method=="EM").mean():.0f}% -> {100*(s2.method=="EM").mean():.0f}%); within-stratum {100*(r2s-s1.flag.mean()):+.2f} pp, composition {100*(s2.flag.mean()-r2s):+.2f} pp')
# ---------------------------------------------------------------- T4 EM vs X-ray with local quality
H('T4. EM vs X-ray at matched resolution, with and without local map quality')
for era,s in S.groupby('era'):
    for lab,f in [('base',"flag ~ C(method,Treatment('X-ray')) + C(rb) + C(ctype) + logV"),('+lq',"flag ~ C(method,Treatment('X-ray')) + C(rb) + C(ctype) + logV + lq_z")]:
        print(era,lab); print(ors(fit(f,s.dropna(subset=['lq_z'])),'method|lq').to_string())
# ---------------------------------------------------------------- T5 role
H('T5. Auxiliary vs antibody-of-interest (entry-level role), with prior template and local quality')
R=S[S.role.isin(['Auxiliary','Antibody of interest'])].copy(); R['aux']=(R.role=='Auxiliary').astype(int)
for era,s in R.groupby('era'):
    print(era); print(s.groupby(['method','role']).flag.agg(rate).unstack().to_string())
    print(ors(fit("flag ~ aux + prior_hires + lq_z + "+BASE,s.dropna(subset=['lq_z'])),'aux|prior|lq').to_string())
print('\n-- by SARS-CoV-2 status (antibody of interest only), EM:')
print(R[(R.method=='EM')&(R.role=='Antibody of interest')].groupby(['era','sars2']).flag.agg(rate).unstack().to_string())
print('\n-- Within entries containing both an auxiliary reagent chain (Nb35/scFv16/BAG2/NabFab/sybody/...) and other antibody chains')
w=M[M.pdb.isin(M[M.aux_reagent].pdb)&M.pdb.isin(M[~M.aux_reagent].pdb)].copy()
print(' entries',w.pdb.nunique(),'| reagent chains',w.aux_reagent.sum(),'flag',rate(w[w.aux_reagent].flag),'| other antibody chains',(~w.aux_reagent).sum(),'flag',rate(w[~w.aux_reagent].flag))
inf=w.groupby('pdb').flag.transform(lambda x: 0<x.sum()<len(x)).astype(bool)
if inf.sum()>3:
    cw=w[inf]; m=ConditionalLogit(cw.flag.values,cw[['aux_reagent']].astype(float).values,groups=pd.factorize(cw.pdb)[0]).fit(disp=0)
    print(' within-entry conditional OR (reagent vs other antibody): %.2f [%.2f-%.2f], informative entries %d'%(np.exp(m.params[0]),*np.exp(m.conf_int()[0]),cw.pdb.nunique()))
print(' local map quality z: reagent %.2f vs other %.2f'%(w[w.aux_reagent].lq_z.mean(),w[~w.aux_reagent].lq_z.mean()))
# ---------------------------------------------------------------- T6-T9 publication pressure
H('T6-T9. Journal tier, structures per paper, target novelty, competition, hold time')
S['log_epp']=np.log2(S.entries_per_paper.clip(lower=1)); S['comp_bin']=pd.cut(S.competitors,[-1,0,2,9,1e4],labels=['0','1–2','3–9','≥10']).astype(str)
S['hold_bin']=pd.cut(S.hold_days,[-1,30,90,180,365,1e5],labels=['≤30d','31–90d','91–180d','181–365d','>1y']).astype(str)
for era,s in S.groupby('era'):
    print(era); print(s.groupby('tier').flag.agg(rate).to_string())
    m=fit("flag ~ C(tier,Treatment('Other journal')) + log_epp + novel_target + C(comp_bin,Treatment('0')) + C(hold_bin,Treatment('91–180d')) + lq_z + sars2 + "+BASE,s.dropna(subset=['lq_z','novel_target']))
    print(ors(m,'tier|epp|novel|comp|hold|sars2|lq').to_string())
# ---------------------------------------------------------------- T10 SARS-CoV-2 2020 with local quality
H('T10. SARS-CoV-2 vs other viral antigens, antibody of interest, EM, 2020 and 2021 — adjusted for local map quality')
V=S[(S.method=='EM')&(S.role=='Antibody of interest')&S.year.isin([2020,2021])].copy()
V['viral']=V.title.fillna('').str.contains('virus|viral|HIV|Env|influenza|hemagglutinin|spike|SARS|RSV|Ebola|dengue|zika|glycoprotein',case=False)
V=V[V.viral|V.sars2]; V['s2']=V.sars2.astype(int); V['y2020']=(V.year==2020).astype(int)
print(V.groupby(['year','s2']).flag.agg(rate).unstack().to_string())
print(ors(fit("flag ~ s2*y2020 + C(rb) + C(ctype) + logV + lq_z",V.dropna(subset=['lq_z'])),'s2|y2020|lq').to_string())
# ---------------------------------------------------------------- T11 co-occurrence
H('T11. Co-occurrence: does a flag on one antibody chain predict flags on other chains in the same entry?')
M['seqkey']=M.seq.str[:105]
g=M.groupby('pdb')
M['other_distinct_flag']=[int(((grp.flag==1)&(grp.seqkey!=k)).any()) for k,(_,grp) in zip(M.seqkey,[(p,g.get_group(p)) for p in M.pdb])]
M['has_other_distinct']=[int((grp.seqkey!=k).any()) for k,(_,grp) in zip(M.seqkey,[(p,g.get_group(p)) for p in M.pdb])]
X=M[(M.has_other_distinct==1)&M.single&M.res.notna()]
print(X.groupby('other_distinct_flag').flag.agg(rate).to_string())
print(ors(fit("flag ~ other_distinct_flag + lq_z + "+BASE,X.dropna(subset=['lq_z'])),'other|lq').to_string())
segs=pd.read_pickle(f'{SP}/retest/segs.pkl')
SG=pd.read_pickle(f'{SP}/sabr/out/segments.pkl'); SG=SG[SG.seqkey.str.len()>=50]
cop=M.groupby(['pdb','seqkey']).agg(n=('chain','size'),nflag=('flag','sum')).reset_index()
cop=cop[(cop.n>=2)&(cop.nflag>=1)]
print('identical copies in the same entry with ≥1 flagged copy: %d groups; all copies flagged in %d (%.0f%%)'%(len(cop),(cop.nflag==cop.n).sum(),100*(cop.nflag==cop.n).mean()))
SG['k']=SG.seqkey.astype(object)+'|'+SG.imgt_start.astype(int).astype(str)
same=SG.groupby(['pdb','k']).chain.nunique().reset_index().merge(M.groupby(['pdb','seqkey']).size().rename('ncopies').reset_index().assign(k0=lambda d:d.seqkey),how='left',left_on='pdb',right_on='pdb')
# ---------------------------------------------------------------- T12 lab clustering and persistence
H('T12. Lab-level clustering and persistence across eras')
ent=M.groupby('pdb').agg(flag=('flag','max'),last_author=('last_author','first'),method=('method','first'),rb=('rb','first'),ybin=('ybin','first'),n=('chain','size'),era=('era','first'),lq_z=('lq_z','mean')).reset_index()
ent['logn']=np.log(ent.n)
mm=smf.logit("flag ~ C(method)*C(rb) + C(ybin) + logn",data=ent).fit(disp=0); ent['p']=mm.predict(ent)
big=ent.groupby('last_author').filter(lambda x: len(x)>=5); rng=np.random.default_rng(0)
def disp(df,col):
    gg=df.groupby('last_author').agg(o=(col,'sum'),e=('p','sum'),v=('p',lambda x:(x*(1-x)).sum())); return ((gg.o-gg.e)**2/gg.v).sum()/len(gg)
obs=disp(big,'flag'); sims=[disp(big.assign(sim=rng.random(len(big))<big.p),'sim') for _ in range(300)]
print('labs ≥5 entries: %d; dispersion observed %.2f vs simulated %.2f (95%% %.2f-%.2f)'%(big.last_author.nunique(),obs,np.mean(sims),*np.percentile(sims,[2.5,97.5])))
labs=ent.groupby(['last_author','era']).agg(o=('flag','sum'),e=('p','sum'),n=('flag','size')).unstack()
both=labs.dropna(); both=both[(both[('n','snapshot (released ≤ Apr 2022)')]>=3)&(both[('n','new (released after Apr 2022)')]>=3)]
oe_old=both[('o','snapshot (released ≤ Apr 2022)')]/both[('e','snapshot (released ≤ Apr 2022)')]; 
newd=ent[ent.era.str.startswith('new')&ent.last_author.isin(both.index)].copy(); newd['old_oe']=newd.last_author.map(oe_old); newd['old_bad']=(newd.old_oe>1.5).astype(int)
print('labs with ≥3 entries in both eras:',len(both))
print(' new-era entry flag rate by lab\'s snapshot O/E >1.5:'); print(newd.groupby('old_bad').flag.agg(rate).to_string())
print(ors(fit("flag ~ old_bad + C(method)*C(rb) + logn",newd.rename(columns={'last_author':'la'}).assign(pdb=newd.last_author)),'old_bad').to_string())
# ---------------------------------------------------------------- T13 corrections
H('T13. Were the conkit-flagged antibody V errors ever corrected? (SAbR on today\'s coordinates)')
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx'); t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
segmap=dict(zip(zip(segs.pdb,segs.chain),segs.segs))
tt=t.merge(M[['pdb','chain','atom_revised','flag']],on=['pdb','chain'])
tt['still']=[any(set(range(r.rs,r.re+1)) & set(range(s['start'],s['end']+1)) for s in segmap.get((r.pdb,r.chain),[])) for r in tt.itertuples()]
print('conkit-flagged antibody chains (V/C):',len(tt),'| entries with an "Atomic model" revision since release: %d (%.0f%%)'%(tt.drop_duplicates('pdb').atom_revised.sum(),100*tt.drop_duplicates('pdb').atom_revised.mean()))
print(' of revised entries, flagged segment still disagreeing today: %s; of unrevised: %s'%(rate(tt[tt.atom_revised==1].still),rate(tt[tt.atom_revised==0].still)))
print(' SAbR-flagged V domains (all eras): coordinates revised since release in %s'%rate(M[M.flag==1].drop_duplicates('pdb').atom_revised))
print(' vs unflagged: %s'%rate(M[M.flag==0].drop_duplicates('pdb').atom_revised))
# ---------------------------------------------------------------- T17 starting model / fitting protocol
H('T17. Starting model type and EM fitting protocol')
for era,s in S.groupby('era'):
    print(era); print(s.groupby(['method','irm']).flag.agg(rate).unstack().to_string())
s=S[S.era.str.startswith('new')].dropna(subset=['lq_z'])
print(ors(fit("flag ~ C(irm,Treatment('none recorded')) + lq_z + "+BASE,s),'irm|lq').to_string())
e=S[(S.method=='EM')].copy()
print(e.groupby(['era','fit']).flag.agg(rate).unstack().to_string())
print(ors(fit("flag ~ C(fit,Treatment('not recorded')) + lq_z + C(rb) + C(ybin) + C(ctype) + logV",e.dropna(subset=['lq_z'])),'fit|lq').to_string())
print('-- EM resolution paradox (new era): rate and starting model by resolution')
ee=S[(S.method=='EM')&S.era.str.startswith('new')]
print(ee.groupby('rb').agg(rate=('flag',rate),AF_model=('irm',lambda x:round(100*(x=='AlphaFold model').mean(),1)),exp_model=('irm',lambda x:round(100*(x=='experimental (PDB) model').mean(),1)),rigid=('fit',lambda x:round(100*(x=='rigid body only').mean(),1)),lq=('lq',lambda x: round(x.mean(),3))).to_string())
print('-- software (where recorded)'); print(S.groupby('era')[[c for c in S.columns if c.startswith('sw_')]].mean().round(3).to_string())
for sw in ['sw_ISOLDE','sw_Rosetta','sw_Coot']:
    print(sw,ors(fit(f"flag ~ {sw} + lq_z + "+BASE,S.dropna(subset=['lq_z'])),sw).to_string())
M.to_pickle(f'{SP}/retest/master2.pkl')
