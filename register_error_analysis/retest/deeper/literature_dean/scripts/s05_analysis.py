"""Compare the error set with SAbDab/PyIgClassify-style reference statistics.
Unit = paper|V-region (one antibody chain type in one paper). Main tests within cryo-EM units, adjusted for
deposition-year bin, best resolution and chain type. BH-FDR across all tests in this script."""
import pickle, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/literature_dean/tables'
R=pickle.load(open(f'{SP}/deeper/cache/literature_dean/ref.pkl','rb'))
c=pd.read_pickle(f'{SP}/deeper/cache/literature_dean/chains_aug.pkl')
c['dep']=pd.to_datetime(c.deposit); c['rel']=pd.to_datetime(c.release)
c['modes']=c.err_modes.fillna('')
log=open(f'{OUT}/analysis.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); log.write(s+'\n')
# ---------------- reference length distributions (SAbDab-style: unique V domains, X-ray <=3.0 A, first released <=2019)
ref=R[(R.hq30)&(R.first_year<=2019)]
P('reference set: unique V domains with an X-ray <=3.0 A structure, first released <=2019:',len(ref),ref.ctype.value_counts().to_dict())
freq={}
rows=[]
for ct in 'HKL':
    for k,col in [('L1','len_CDR1'),('L2','len_CDR2'),('L3','len_CDR3')]:
        v=ref[ref.ctype==ct][col]; vc=v.value_counts(); n=len(v)
        for L,cnt in vc.items(): freq[(ct,k,int(L))]=cnt/n
        freq[(ct,k,'n')]=n
        for L,cnt in vc.sort_index().items(): rows.append(dict(ctype=ct,cdr=k,length=int(L),ref_n=int(cnt),ref_frac=cnt/n))
REF=pd.DataFrame(rows)
def f(ct,k,L):
    if pd.isna(L): return np.nan
    return freq.get((ct,k,int(L)),0.5/freq[(ct,k,'n')])  # pseudocount for lengths unseen in the reference
# ---------------- unit table
c['is_em']=c.method=='EM'
g=c.groupby('unit')
U=pd.DataFrame({'ctype':g.ctype.first(),'all_em':g.is_em.all(),'year':g.dep.min().dt.year,
   'res':g.res.min(),'err':g.err_first.any(),'modes':g.modes.agg(lambda s:';'.join(sorted({m for x in s for m in x.split(';') if m}))),
   'rL1':g.rL1.first(),'rL2':g.rL2.first(),'rL3':g.rL3.first(),'vhh':g.vhh.max(),'sars2':g.sars2.any(),'species':g.species.first(),
   'v_id':g.v_id.first(),'dbrow':g.dbrow.first(),'n_chains':g.size(),'rel_min':g.rel.min(),
   'err_any':g.err_any.any(),'copied_only':g.err_copied_or_artifact_only.any()})
U['ybin']=pd.cut(U.year,[0,2019,2020,2021,2022,2100],labels=['<=2019','2020','2021','2022','>=2023']).astype(str)
for k in ['L1','L2','L3']:
    U['f'+k]=[f(ct,k,L) for ct,L in zip(U.ctype,U['r'+k])]
    U['rare'+k]=(U['f'+k]<0.05).astype(int)
U['rare_any']=((U.rareL1+U.rareL2+U.rareL3)>0).astype(int)
U['logf_min']=np.log10(U[['fL1','fL2','fL3']].min(1))
for m in ['A_FR1Nterm','F_CDR1','B_CDR2FR3','C_FR3','Dplus_CDR3short','Dminus_CDR3long','E_FR4bulge','D0','other']:
    U['m_'+m]=U.modes.str.contains(m).astype(int)
U['m_D_E']=((U.m_Dplus_CDR3short+U.m_Dminus_CDR3long+U.m_E_FR4bulge+U.m_D0)>0).astype(int)
# modal-pull quantities: reference frequency of the length the error would 'snap' to
U['pull_short3']=[np.log2(f(ct,'L3',L-1)/f(ct,'L3',L)) if pd.notna(L) else np.nan for ct,L in zip(U.ctype,U.rL3)]
U['pull_long3']=[np.log2(f(ct,'L3',L+1)/f(ct,'L3',L)) if pd.notna(L) else np.nan for ct,L in zip(U.ctype,U.rL3)]
# V domain has an X-ray <=3.0 A structure anywhere in the PDB (any date), i.e. would enter a SAbDab-style curated set
U['has_hq']=[bool(R.hq30.iat[int(r)]) if (r>=0 and r<len(R)) else False for r in U.dbrow]
U.to_csv(f'{OUT}/units_literature.csv.gz')
E=U[U.all_em & U.rL3.notna()].copy()
P('EM units',len(E),'error units',int(E.err.sum()),'| X-ray-only units',int((~U.all_em).sum()))
tests=[]
def logit(df,y,x,adj="C(ybin)+res+C(ctype)",label=None,extra=''):
    d=df.dropna(subset=[y,x,'res']).copy(); d[y]=d[y].astype(int)
    if d[x].dtype==bool: d[x]=d[x].astype(int)
    try:
        m=smf.logit(f'{y} ~ {x} + {adj}',d).fit(disp=0,maxiter=200)
        orr=np.exp(m.params[x]); ci=np.exp(m.conf_int().loc[x]).values; p=m.pvalues[x]
    except Exception as e:
        orr,ci,p=np.nan,(np.nan,np.nan),np.nan
    a=d[d[x]==1] if d[x].nunique()<=2 else None
    if a is not None:
        b=d[d[x]==0]; ev=f'{a[y].mean():.3f} ({int(a[y].sum())}/{len(a)})'; cv=f'{b[y].mean():.3f} ({int(b[y].sum())}/{len(b)})'
    else:
        ev=f'mean x in y=1: {d[d[y]==1][x].mean():.3f}'; cv=f'mean x in y=0: {d[d[y]==0][x].mean():.3f}'
    tests.append(dict(test=label or f'{y}~{x}',y=y,x=x,n=len(d),n_y=int(d[y].sum()),exposed=ev,unexposed=cv,OR=orr,lo=ci[0],hi=ci[1],p=p,adjusted=adj,note=extra))
    P(f'{label or y+"~"+x}: n={len(d)} y={int(d[y].sum())} exposed {ev} vs {cv}  OR {orr:.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={p:.3g}')
P('\n== Rare CDR lengths (ref freq <5% among X-ray<=3.0 A unique V domains, <=2019) vs first-occurrence errors, EM units')
logit(E,'err','rare_any',label='T1 any CDR length rare -> any error')
logit(E,'err','rareL1',label='T2 rare CDR1 length -> any error')
logit(E,'err','rareL2',label='T3 rare CDR2 length -> any error')
logit(E,'err','rareL3',label='T4 rare CDR3 length -> any error')
logit(E,'m_F_CDR1','rareL1',label='T5 rare CDR1 length -> F_CDR1 mode')
logit(E,'m_B_CDR2FR3','rareL2',label='T6 rare CDR2 length -> B_CDR2FR3 mode')
logit(E,'m_D_E','rareL3',label='T7 rare CDR3 length -> CDR3/FR4 modes (D+,D-,D0,E)')
logit(E,'m_A_FR1Nterm','rare_any',label='T8 rare any length -> A_FR1Nterm mode (location mismatch control)')
E['neg_logf_min']=-E.logf_min
logit(E,'err','neg_logf_min',label='T9 rarity score (-log10 min ref freq) -> any error (per decade)')
P('\n== Modal pull: does the slipped register correspond to a more common CDR3 length in the reference?')
logit(E,'m_Dplus_CDR3short','pull_short3',label='T10 log2 freq(L3-1)/freq(L3) -> D+ (CDR3 built short)')
logit(E,'m_Dminus_CDR3long','pull_long3',label='T11 log2 freq(L3+1)/freq(L3) -> D- (CDR3 built long)')
E['short_more_common']=(E.pull_short3>0).astype(int); E['long_more_common']=(E.pull_long3>0).astype(int)
logit(E,'m_Dplus_CDR3short','short_more_common',label='T12 L3-1 more common than L3 -> D+')
logit(E,'m_Dminus_CDR3long','long_more_common',label='T13 L3+1 more common than L3 -> D-')
for ct in 'HKL':
    d=E[E.ctype==ct]
    P(f'  {ct}: D+ units {int(d.m_Dplus_CDR3short.sum())}, of which L3-1 more common: {int(d[d.m_Dplus_CDR3short==1].short_more_common.sum())}; controls with L3-1 more common: {d[d.err==0].short_more_common.mean():.3f}')
P('\n== Has an X-ray <=3.0 A structure of the same V domain anywhere in the PDB (enters SAbDab-style curated sets)')
logit(E,'err','has_hq',label='T14 V domain has X-ray<=3.0 A structure (any date) -> any error')
# ---------------- nanobody / species / germline identity relative to reference
E['lowid']=(E.v_id<0.85).astype(int)
logit(E,'err','lowid',label='T15 germline identity <85% -> any error (context)')
P('\n== Specific canonical lengths (North nomenclature in brackets)')
H=E[(E.ctype=='H')&E.rL2.isin([7,8])].copy(); H['h2_7']=(H.rL2==7).astype(int)
logit(H,'err','h2_7',adj='C(ybin)+res',label='T16 heavy CDR2 IMGT 7 [H2-9] vs 8 [H2-10] -> any error')
logit(H,'m_B_CDR2FR3','h2_7',adj='C(ybin)+res',label='T17 heavy CDR2 IMGT 7 vs 8 -> B_CDR2FR3 mode')
Lm=E[(E.ctype=='L')&E.rL1.isin([8,9])].copy(); Lm['l1_8']=(Lm.rL1==8).astype(int)
logit(Lm,'err','l1_8',adj='C(ybin)+res',label='T18 lambda CDR1 IMGT 8 [L1-13] vs 9 [L1-14] -> any error (replication)')
Km=E[(E.ctype=='K')&E.rL1.isin([6,12])].copy(); Km['k1_12']=(Km.rL1==12).astype(int)
logit(Km,'err','k1_12',adj='C(ybin)+res',label='T19 kappa CDR1 IMGT 12 [L1-17] vs 6 [L1-11] -> any error (replication)')
# ---------------- BH
T=pd.DataFrame(tests); T['q']=multipletests(T.p.fillna(1),method='fdr_bh')[1]
T.to_csv(f'{OUT}/tests_bh.csv',index=False)
P('\n',T[['test','n','n_y','exposed','unexposed','OR','lo','hi','p','q']].to_string())
# ---------------- reference length table with study comparison
rows=[]
for ct in 'HKL':
    for k in ['L1','L2','L3']:
        sub=E[E.ctype==ct]
        ctrl=sub[(sub.err==0)&(~sub.err_any)]; er=sub[sub.err]
        for L in sorted(set(REF[(REF.ctype==ct)&(REF.cdr==k)].length)|set(sub['r'+k].dropna().astype(int))):
            rr=REF[(REF.ctype==ct)&(REF.cdr==k)&(REF.length==L)]
            rows.append(dict(ctype=ct,cdr=k,length=L,ref_frac=float(rr.ref_frac.sum()),ref_n=int(rr.ref_n.sum()),
                em_clean_units_frac=(ctrl['r'+k]==L).mean(),em_clean_n=int((ctrl['r'+k]==L).sum()),
                em_error_units_frac=(er['r'+k]==L).mean(),em_error_n=int((er['r'+k]==L).sum()),
                em_error_rate=(sub[sub['r'+k]==L].err.mean() if (sub['r'+k]==L).sum() else np.nan)))
LT=pd.DataFrame(rows); LT.to_csv(f'{OUT}/cdr_length_vs_reference.csv',index=False)
# distribution distances
P('\n== Jensen-Shannon distance of CDR-length distributions vs reference (EM clean units / EM error units)')
from scipy.spatial.distance import jensenshannon
for (ct,k),d in LT.groupby(['ctype','cdr']):
    P(f'  {ct}{k[1]}: clean {jensenshannon(d.ref_frac+1e-9,d.em_clean_units_frac+1e-9):.3f}  error {jensenshannon(d.ref_frac+1e-9,d.em_error_units_frac+1e-9):.3f}  n_err={int(d.em_error_n.sum())}')
log.close()
