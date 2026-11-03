"""Re-test the secondary claims of the amyloid finding:
(1) lambda L1=8 vs L1=9 error rate with/without amyloid (chain and unit level; all methods and EM-only; adjusted for
    method, resolution, deposition-year bin; cluster-robust by paper);
(2) 'antibody alone' excess with/without amyloid;
(3) how much the amyloid runs inflate the 'chain end / next to unbuilt loop' origin and mode-A counts;
(4) expected number of error entries among the amyloid entries under a year/res/nV model fitted to non-amyloid EM entries."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_confounding'
PM='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
c=pd.read_csv(f'{SP}/chains.csv.gz'); e=pd.read_csv(f'{SP}/entries.csv'); r=pd.read_csv(f'{SP}/runs_first_occurrence.csv')
rx=r'amyloid(?!-beta)'
c['amy']=c.title.fillna('').str.contains(rx,case=False); e['amy']=e.title.fillna('').str.contains(rx,case=False)
AMY=set(e[e.amy].pdb)
c['dep_year']=pd.to_datetime(c.deposit).dt.year
c['ybin']=pd.cut(c.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=['<=2018','2019','2020','2021','2022','2023','>=2024']).astype(str)
c['cluster']=c.doi.fillna(c.paper)
rows=[]
def add(**k): rows.append(k); print(k)
def fisher(a,n1,b,n2):
    return stats.fisher_exact([[a,n1-a],[b,n2-b]])[1]
lam=c[(c.ctype=='L')&(c.L1.isin([8,9]))].copy()
lam['L1is8']=(lam.L1==8).astype(int)
for meth in ['all','EM']:
    for excl in [False,True]:
        d=lam if meth=='all' else lam[lam.method=='EM']
        if excl: d=d[~d.amy]
        for lvl in ['chain','unit']:
            if lvl=='chain':
                g8=d[d.L1==8]; g9=d[d.L1==9]
                a,n1,b,n2=g8.err_first.sum(),len(g8),g9.err_first.sum(),len(g9)
            else:
                u=d.groupby('unit').agg(err=('err_first','max'),L1=('L1','first'))
                a,n1=u[u.L1==8].err.sum(),(u.L1==8).sum(); b,n2=u[u.L1==9].err.sum(),(u.L1==9).sum()
            add(test='lambda L1=8 vs 9 (raw)',method=meth,amyloid_excluded=excl,level=lvl,err_n=f'{a}/{n1}',ctrl_n=f'{b}/{n2}',
                rate8=round(100*a/n1,2),rate9=round(100*b/n2,2),RR=round((a/n1)/(b/n2),2) if b else np.nan,p=fisher(a,n1,b,n2))
# adjusted logistic, chain level, cluster by paper
for meth in ['all','EM']:
    for excl in [False,True]:
        d=lam if meth=='all' else lam[lam.method=='EM']
        if excl: d=d[~d.amy]
        d=d.dropna(subset=['res']).copy(); d['y']=d.err_first.astype(int)
        f='y ~ L1is8 + res + C(ybin)'+(' + C(method)' if meth=='all' else '')
        try:
            m=smf.glm(f,data=d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.cluster)[0]})
            ci=np.exp(m.conf_int().loc['L1is8'])
            add(test='lambda L1=8 vs 9 (adjusted chain-level logistic, cluster by paper)',method=meth,amyloid_excluded=excl,level='chain',
                n=len(d),n_err=int(d.y.sum()),OR=round(float(np.exp(m.params['L1is8'])),2),lo=round(ci[0],2),hi=round(ci[1],2),p=float(m.pvalues['L1is8']))
        except Exception as ex: print('fail',ex)
        # unit-level (one row per paper|V-region), adjusted
        u=d.groupby('unit').agg(y=('y','max'),L1is8=('L1is8','first'),res=('res','min'),ybin=('ybin','first'),method=('method','first'),cluster=('cluster','first')).reset_index()
        try:
            m=smf.glm(f,data=u,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(u.cluster)[0]})
            ci=np.exp(m.conf_int().loc['L1is8'])
            add(test='lambda L1=8 vs 9 (adjusted unit-level logistic, cluster by paper)',method=meth,amyloid_excluded=excl,level='unit',
                n=len(u),n_err=int(u.y.sum()),OR=round(float(np.exp(m.params['L1is8'])),2),lo=round(ci[0],2),hi=round(ci[1],2),p=float(m.pvalues['L1is8']))
        except Exception as ex: print('fail',ex)
# amyloid units among lambda L1=8 error units
le=lam[lam.err_first]
print('lambda L1=8 error units',le[le.L1==8].unit.nunique(),' of which amyloid',le[(le.L1==8)&le.amy].unit.nunique())
print('lambda L1=8 error papers (DOI or entry)',le[le.L1==8].cluster.nunique(),' amyloid',le[(le.L1==8)&le.amy].cluster.nunique())
print(le[(le.L1==8)&~le.amy].groupby('unit').agg(pdb=('pdb','first'),v=('v_gene','first'),modes=('err_modes','first'),method=('method','first'),year=('dep_year','first')).to_string())
# (2) antibody alone
fr=pd.read_csv(f'{PM}/em_entry_analysis_frame_with_amyloid.csv.gz')
for excl in [False,True]:
    d=fr[~fr.amyloid] if excl else fr
    a=int(d[d.y==1].ab_alone.sum()); n1=int((d.y==1).sum()); b=int(d[d.y==0].ab_alone.sum()); n2=int((d.y==0).sum())
    add(test='antibody alone (EM entries)',amyloid_excluded=excl,level='entry',err_n=f'{a}/{n1}',ctrl_n=f'{b}/{n2}',rate_err=round(100*a/n1,2),rate_ctrl=round(100*b/n2,2),p=fisher(a,n1,b,n2))
print(fr[(fr.y==1)&(fr.ab_alone==1)&(~fr.amyloid)][['pdb','title']].to_string())
# (3) origin / mode inflation
r['amy']=r.pdb.isin(AMY)
ru=r.merge(c[['pdb','chain','unit']],on=['pdb','chain'],how='left')
for lab,sub in [('runs',ru),('units',ru.drop_duplicates(['unit','origin']))]:
    o3=sub.origin.str.startswith('3')
    add(test=f'share of first-occurrence {lab} in origin 3 (chain end/next to gap)',amyloid_excluded=False,level=lab,err_n=f'{o3.sum()}/{len(sub)}',rate_err=round(100*o3.mean(),1))
    s2=sub[~sub.amy]; o3=s2.origin.str.startswith('3')
    add(test=f'share of first-occurrence {lab} in origin 3 (chain end/next to gap)',amyloid_excluded=True,level=lab,err_n=f'{o3.sum()}/{len(s2)}',rate_err=round(100*o3.mean(),1))
# runs next_to_gap flag
for excl in [False,True]:
    s=ru[~ru.amy] if excl else ru
    add(test='first-occurrence runs flagged next_to_gap_origin',amyloid_excluded=excl,level='run',err_n=f'{int(s.next_to_gap_origin.sum())}/{len(s)}',rate_err=round(100*s.next_to_gap_origin.mean(),1))
# (4) expected errors among amyloid entries given year/res/nV (model on non-amyloid EM entries: err_first vs clean)
em=e[(e.method=='EM')&((e.err_first)|(~e.err_any))].copy()
em['y']=em.err_first.astype(int); em['dep_year']=pd.to_datetime(em.deposit).dt.year
em['ybin']=pd.cut(em.dep_year,[0,2018,2019,2020,2021,2022,2023,2100]).astype(str); em['l2nV']=np.log2(em.n_V_chains)
m=smf.glm('y ~ C(ybin) + res + l2nV',data=em[~em.amy],family=sm.families.Binomial()).fit()
pa=m.predict(em[em.amy]); exp_=pa.sum()
# Poisson-binomial P(all 15 flagged)
p_all=float(np.prod(pa))
add(test='amyloid EM entries flagged vs expected from year/res/nV model',level='entry',err_n=f'{int(em[em.amy].y.sum())}/{int(em.amy.sum())}',expected=round(float(exp_),2),
    p=p_all,note='p = P(all flagged | model probabilities)')
# per-paper version: 7 independent papers
add(test='amyloid papers flagged',level='paper',err_n='7/7 papers (6 labs)',note='Radamaker/Fandrich 2019, Fandrich 2021a, 2021b, Saelices 2024, Ricagno 2024, Li D 2025, Huang H 2025')
R=pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
mask=R.p.notna()
R.loc[mask,'q_BH']=multipletests(R.loc[mask,'p'],method='fdr_bh')[1]
R.to_csv(f'{OUT}/secondary_claims.csv',index=False)
