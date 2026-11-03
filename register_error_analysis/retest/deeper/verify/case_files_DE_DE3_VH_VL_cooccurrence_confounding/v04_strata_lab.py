"""Stratified (Mantel-Haenszel) and lab-clustered versions at the Fab-in-paper unit level; one unit per lab."""
import pandas as pd, numpy as np, warnings
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_DE3_confounding'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE3_VH_VL_cooccurrence_confounding'
T=pd.read_pickle(f'{CACHE}/T_em.pkl')
T['fabunit']=T.paper.astype(str)+'||'+T.Hseqs+'||'+T.Lseqs
U=T.groupby('fabunit').agg(H_DE=('H_DE','max'),L_oth=('L_oth','max'),year=('year','min'),res=('res','median'),sars2=('sars2','max'),
                           paper=('paper','first'),lab=('last_author','first')).reset_index()
U['yb']=pd.cut(U.year,[0,2019,2021,2023,2030]).astype(str); U['rb']=pd.cut(U.res,[0,3.5,4.0,9]).astype(str)
U['stratum']=U.yb+'|'+U.rb+'|'+U.sars2.astype(str)
tabs=[]
for s,g in U.groupby('stratum'):
    t=pd.crosstab(g.H_DE,g.L_oth).reindex(index=[True,False],columns=[True,False],fill_value=0).values.astype(float)
    if t[0].sum()>0 and t[1].sum()>0: tabs.append(t)
st=StratifiedTable(tabs)
out=[dict(analysis='MH over year-bin x res-bin x SARS2 strata (Fab units)',OR=st.oddsratio_pooled,p=st.test_null_odds().pvalue,
          n_strata=len(tabs),n=int(sum(t.sum() for t in tabs)),exposed_events=int(sum(t[0,0] for t in tabs)),exposed=int(sum(t[0].sum() for t in tabs)))]
z=U.assign(y=U.L_oth.astype(int),x=U.H_DE.astype(int))
m=smf.logit('y ~ x + C(yb) + res + sars2',data=z).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(z.lab)[0]})
out.append(dict(analysis='Fab units, logit, SE clustered by last author (lab)',OR=float(np.exp(m.params['x'])),p=float(m.pvalues['x']),n=len(z),
                exposed_events=int((z.x&z.y).sum()),exposed=int(z.x.sum())))
# one random Fab unit per lab, 2000 draws
rng=np.random.default_rng(7); G=[g.index.values for _,g in U.groupby('lab')]; ors=[];ps=[];ev=[]
for i in range(2000):
    S=U.loc[[rng.choice(g) for g in G]]
    t=pd.crosstab(S.H_DE,S.L_oth).reindex(index=[False,True],columns=[False,True],fill_value=0).values
    o,p=fisher_exact(t); ors.append(o); ps.append(p); ev.append((t[1,1],t[1].sum()))
ev=np.array(ev); ors=np.array(ors,dtype=float)
out.append(dict(analysis='one random Fab unit per lab (2000 draws)',OR=float(np.median(ors[np.isfinite(ors)])),p=float(np.median(ps)),n=len(G),
     exposed_events=float(np.median(ev[:,0])),exposed=float(np.median(ev[:,1])),note=f'share p<0.05 {np.mean(np.array(ps)<0.05):.2f}'))
O=pd.DataFrame(out); O.to_csv(f'{OUT}/tables/strata_lab.csv',index=False); print(O.to_string())
