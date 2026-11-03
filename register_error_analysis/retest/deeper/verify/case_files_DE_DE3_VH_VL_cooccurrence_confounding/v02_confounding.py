"""Confounding / clustering checks for DE3 (VH D/E slip -> VL first-occurrence error, cryo-EM Fab entries)."""
import pandas as pd, numpy as np, warnings
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_DE3_confounding'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE3_VH_VL_cooccurrence_confounding'
T=pd.read_pickle(f'{CACHE}/T_em.pkl'); TX=pd.read_pickle(f'{CACHE}/T_xray.pkl')
res_rows=[]
def fisher(df,a,b):
    tab=pd.crosstab(df[a],df[b]).reindex(index=[False,True],columns=[False,True],fill_value=0).values
    o,p=fisher_exact(tab); return o,p,int(tab[1,1]),int(tab[1].sum()),int(tab[0,1]),int(tab[0].sum())
def logit(df,form,x='x',cluster='paper'):
    try:
        m=smf.logit(form,data=df).fit(disp=0,maxiter=200,cov_type='cluster',cov_kwds={'groups':pd.factorize(df[cluster])[0]})
        return float(np.exp(m.params[x])),float(m.pvalues[x]),np.exp(m.conf_int().loc[x]).tolist()
    except Exception as e:
        return np.nan,np.nan,[np.nan,np.nan]
def add(label,df,a='H_DE',b='L_oth',form=None,note=''):
    o,p,k1,n1,k0,n0=fisher(df,a,b)
    row=dict(analysis=label,exposure=a,outcome=b,n=len(df),n_exposed=n1,events_exposed=k1,rate_exposed=k1/max(n1,1),
             n_unexposed=n0,events_unexposed=k0,rate_unexposed=k0/max(n0,1),fisher_OR=o,fisher_p=p,
             n_papers_exposed=df[df[a]].paper.nunique(),n_papers_exposed_event=df[df[a]&df[b]].paper.nunique(),note=note)
    if form:
        z=df.assign(y=df[b].astype(int),x=df[a].astype(int))
        aor,ap,ci=logit(z,form); row.update(adj_OR=aor,adj_p=ap,adj_lo=ci[0],adj_hi=ci[1],model=form)
    res_rows.append(row); return row
for D in (T,TX):
    D['yb']=pd.cut(D.year,[0,2019,2021,2023,2030]).astype(str)
    D['multiFab']=(D.nH>1).astype(int)
    D['Fab_lqz']=D[['H_lqz','L_lqz']].mean(1)
    D['lqz_low']=(D.Fab_lqz< -1).astype(int)
    D['sars2i']=D.sars2.astype(int)
    D['H_oth_i']=D.H_oth.astype(int)
    D['ctypeK']=(D.ctypeL.str.contains('K')).astype(int)
base='y ~ x + C(yb) + res'
# 1. reproduction
add('1 entry: reproduce (year bin+res, cluster paper)',T,form=base)
add('1b entry: + sars2, nH, kappa',T,form=base+' + sars2i + np.log(nH) + ctypeK')
# local map fit (Fab mean lq_z), complete cases
Tq=T.dropna(subset=['Fab_lqz'])
add('1c entry: + Fab lq_z (map-model fit)',Tq,form=base+' + sars2i + Fab_lqz')
add('1d entry: + H has other (non-D/E) error',T,form=base+' + H_oth_i')
add('1e entry: + all of the above',Tq,form=base+' + sars2i + np.log(nH) + ctypeK + Fab_lqz + H_oth_i')
# 2. specificity: is it D/E-specific or any heavy error?
T['H_DE_only']=T.H_DE&~T.H_oth
T['H_oth_only']=T.H_oth&~T.H_DE
add('2a entry: H D/E only (no other H error) -> L_oth',T,a='H_DE_only',form=base)
add('2b entry: H other-only (no D/E) -> L_oth',T,a='H_oth_only',form=base)
add('2c entry: H any error -> L any error',T,a='H_any',b='L_any',form=base)
# 3. Fab-in-paper units (same paper, same H and L V sequences collapsed)
T['fabunit']=T.paper.astype(str)+'||'+T.Hseqs+'||'+T.Lseqs
agg={c:'max' for c in ['H_DE','L_DE','H_oth','L_oth','H_any','L_any','sars2i','multiFab','H_oth_i','ctypeK']}
agg.update(year='min',res='median',Fab_lqz='mean',nH='max',paper='first',pdb='first')
U=T.groupby('fabunit').agg(agg).reset_index()
U['yb']=pd.cut(U.year,[0,2019,2021,2023,2030]).astype(str)
for c in ['H_DE','L_DE','H_oth','L_oth','H_any','L_any']: U[c]=U[c].astype(bool)
add('3 Fab-in-paper unit',U,form=base)
Uq=U.dropna(subset=['Fab_lqz'])
add('3b Fab unit: + sars2, Fab lq_z, H_oth',Uq,form=base+' + sars2i + Fab_lqz + H_oth_i')
# 4. paper level: collapse to paper (any H_DE entry; any entry with H_DE & L_oth) -> per paper: does an H_DE paper have L errors?
# 4a one random entry per paper, repeated
rng=np.random.default_rng(1)
ors=[];ps=[];k1s=[]
G=[g.index.values for _,g in T.groupby('paper')]
for it in range(2000):
    idx=[rng.choice(g) for g in G]; S=T.loc[idx]
    o,p,k1,n1,k0,n0=fisher(S,'H_DE','L_oth'); ors.append(o if np.isfinite(o) else np.nan); ps.append(p); k1s.append((k1,n1,k0,n0))
ors=np.array(ors); ps=np.array(ps)
k=np.array(k1s)
res_rows.append(dict(analysis='4a one random entry per paper (2000 draws)',exposure='H_DE',outcome='L_oth',n=len(G),
   n_exposed=float(np.median(k[:,1])),events_exposed=float(np.median(k[:,0])),rate_exposed=float(np.median(k[:,0]/np.maximum(k[:,1],1))),
   n_unexposed=float(np.median(k[:,3])),events_unexposed=float(np.median(k[:,2])),rate_unexposed=float(np.median(k[:,2]/k[:,3])),
   fisher_OR=float(np.nanmedian(ors)),fisher_p=float(np.median(ps)),note=f'share of draws p<0.05: {np.mean(ps<0.05):.2f}; median OR (finite)'))
# 4b paper-collapsed exposure: papers with any H_DE entry; outcome = any L_oth in those same entries vs in non-H_DE entries of the same papers
# 5. paper-level propensity: leave-one-Fab-unit-out rate of VL errors in the same paper
def loo(df,col):
    s=df.groupby('paper')[col].transform('sum'); n=df.groupby('paper')[col].transform('count')
    return ((s-df[col].astype(int))/(n-1)).where(n>1)
U['paper_Lerr_other']=loo(U,'L_any'); U['paper_Herr_other']=loo(U,'H_any')
Up=U.dropna(subset=['paper_Lerr_other'])
add('5a Fab unit, papers with >=2 Fab units (base)',Up,form=base)
add('5b Fab unit: + paper VL/VH error rate in OTHER Fab units',Up,form=base+' + paper_Lerr_other + paper_Herr_other')
# 5c spillover: VL error rate in OTHER Fab units of papers that have an H_DE unit vs papers that do not
P_HDE=set(U[U.H_DE].paper)
O=U[~U.H_DE].copy(); O['paper_has_HDE']=O.paper.isin(P_HDE)
add('5c spillover: L_oth in non-H_DE Fab units, paper has an H_DE unit elsewhere',O,a='paper_has_HDE',b='L_oth',form=base)
add('5d spillover: L_any in non-H_DE Fab units, paper has an H_DE unit elsewhere',O,a='paper_has_HDE',b='L_any',form=base)
# 6. within-paper conditional logit on Fab units (papers with variation in outcome)
W=U.copy(); W['y']=W.L_oth.astype(int); W['x']=W.H_DE.astype(int)
g=W.groupby('paper').agg(vy=('y','std'),vx=('x','std'))
keep=g[(g.vy>0)&(g.vx>0)].index
Wk=W[W.paper.isin(keep)]
try:
    cm=ConditionalLogit(Wk.y.values,Wk[['x']].values,groups=pd.factorize(Wk.paper)[0]).fit(disp=0)
    res_rows.append(dict(analysis='6 within-paper conditional logit (Fab units; papers with variation in both)',exposure='H_DE',outcome='L_oth',
        n=len(Wk),n_papers_exposed=len(keep),fisher_OR=np.nan,adj_OR=float(np.exp(cm.params[0])),adj_p=float(cm.pvalues[0]),
        adj_lo=float(np.exp(cm.conf_int()[0][0])),adj_hi=float(np.exp(cm.conf_int()[0][1])),note=f'{len(keep)} informative papers'))
except Exception as e:
    print('clogit fail',e)
# 7. drop the dominant paper(s)
dom='10.1038/s41422-021-00487-9'
add('7a entry: drop Zhou Cell Res 2021 (7CZx)',T[T.paper!=dom],form=base)
add('7b Fab unit: drop Zhou Cell Res 2021',U[U.paper!=dom],form=base)
dom2=['10.1038/s41422-021-00487-9','10.1016/j.celrep.2025.115334']
add('7c Fab unit: drop Zhou 2021 + Zhu 2025 (N-terminal-slip papers)',U[~U.paper.isin(dom2)],form=base)
# 8. outcome excluding chain-end (origin 3) VL errors
T['L_oth_noend']=[bool(o) and any(t not in ('3',) for c in str(o).split() for t in c.split(':')[1].split(';')) if o else False for o in T.L_orig]
# simpler: VL runs whose origin is not '3'
add('8 entry: outcome = VL non-D/E error not at chain end/unbuilt (origin != 3)',T,b='L_oth_noend',form=base)
# 9. X-ray replication
add('9 X-ray entries (same definition, entry level)',TX,form=base)
TX['fabunit']=TX.paper.astype(str)+'||'+TX.Hseqs+'||'+TX.Lseqs
UX=TX.groupby('fabunit').agg({**{c:'max' for c in ['H_DE','L_oth','H_any','L_any']},'year':'min','res':'median','paper':'first'}).reset_index()
UX['yb']=pd.cut(UX.year,[0,2019,2021,2023,2030]).astype(str)
for c in ['H_DE','L_oth','H_any','L_any']: UX[c]=UX[c].astype(bool)
add('9b X-ray Fab units',UX,form=base)
R=pd.DataFrame(res_rows)
from statsmodels.stats.multitest import multipletests
pcol=R.adj_p.where(R.adj_p.notna(),R.fisher_p)
R['p_used']=pcol; m=pcol.notna(); R.loc[m,'BH_q']=multipletests(pcol[m],method='fdr_bh')[1]
R.to_csv(f'{OUT}/tables/confounding_tests.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',70)
print(R[['analysis','exposure','outcome','n','n_exposed','events_exposed','rate_exposed','events_unexposed','rate_unexposed','fisher_OR','fisher_p','adj_OR','adj_p','adj_lo','adj_hi','n_papers_exposed','n_papers_exposed_event','BH_q','note']].round(4).to_string())
U.to_pickle(f'{CACHE}/U_em.pkl')
print(U[U.H_DE][['paper','pdb','L_oth','L_any','paper_Lerr_other','Fab_lqz']].to_string())
