"""Re-test SN1 (SEQRES starts after IMGT 1 -> mode A) with confounder adjustment, model-start conditioning,
cluster-robust errors and strata. EM units, mode-A error units vs clean units of the same chain types."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
U=pd.read_csv(f'{SN}/units.csv.gz',low_memory=False)
CN=pd.read_csv(f'{OUT}/chain_nterm.csv.gz')
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
runsall=pd.read_csv(f'{SP}/deeper/runs_all.csv')
chd=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','deposit','species','vhh'])
CN=CN.merge(chd,on=['pdb','chain'],how='left'); CN['deposit']=pd.to_datetime(CN.deposit)
Akeys=set(zip(runs.loc[runs['mode']=='A_FR1Nterm','pdb'],runs.loc[runs['mode']=='A_FR1Nterm','chain']))
CN['is_A']=[k in Akeys for k in zip(CN.pdb,CN.chain)]
EM=U.method_u=='EM'
U['caseA']=EM&U.err_first_u&U.modes_u.str.contains('A_FR1Nterm')
U['ctrl']=EM&U.clean&U.ctype.isin(['H','K','L'])
D=U[U.caseA|U.ctrl].copy()
# representative chain: investigator's (earliest) for controls; for cases the earliest mode-A error chain
rep=CN.sort_values(['unit','deposit','pdb','chain'])
repA=rep[rep.is_A].groupby('unit').first()
rep0=rep.groupby('unit').first()
cols=['pdb','chain','seq_first_imgt','model_first_anarci','model_first_sabr','v_offset','n_mod_1_10']
r=rep0[cols].add_prefix('r0_'); D=D.join(r,on='unit')
ra=repA[cols].add_prefix('rA_'); D=D.join(ra,on='unit')
for c in cols[2:]:
    D['x_'+c]=np.where(D.caseA,D['rA_'+c],D['r0_'+c])
D['y']=D.caseA.astype(int)
D['seq_trunc']=(D.x_seq_first_imgt>1).astype(int)
D['seq_trunc_inv']=D.seq_nt_trunc.astype(int)
D['mstart']=pd.cut(D.x_model_first_anarci,[0,1,2,3,100],labels=['1','2','3','4+']).astype(str)
D['model_trunc']=(D.x_model_first_anarci>1).astype(int)
D['yb']=pd.cut(D.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=['<=2018','2019','2020','2021','2022','2023','>=2024']).astype(str)
D['nonhm']=(~D.q_species_all.isin(['human','mouse'])).astype(int)
D['vhh_i']=D.vhh_u.fillna(0).astype(int); D['sars_i']=D.sars2_u.astype(int)
D['lab']=D.last_author.fillna(D.paper.astype(str))
D.to_csv(f'{OUT}/units_modeA_analysis.csv.gz',index=False)
L=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); L.append(s)
P('cases',D.y.sum(),'controls',(D.y==0).sum())
P('investigator seq_nt_trunc: cases %.3f controls %.3f'%(D[D.y==1].seq_trunc_inv.mean(),D[D.y==0].seq_trunc_inv.mean()))
P('error-chain seq_trunc: cases %.3f (%d) controls %.3f'%(D[D.y==1].seq_trunc.mean(),D[D.y==1].seq_trunc.sum(),D[D.y==0].seq_trunc.mean()))
P('agreement investigator vs error-chain-based:',pd.crosstab(D.seq_trunc_inv,D.seq_trunc).values.tolist())
P('\nCross-tab: mode-A rate by model start (ANARCI IMGT of first built residue) x SEQRES truncated')
t=D.groupby(['mstart','seq_trunc']).y.agg(['sum','count']); t['rate']=t['sum']/t['count']; P(t.to_string())
P('\nModel start distribution, cases vs controls'); P(pd.crosstab(D.mstart,D.y).to_string())
rows=[]
def fit(formula,d,label,feat='seq_trunc',cluster=None):
    try:
        m=smf.logit(formula,data=d)
        if cluster is None: f=m.fit(disp=0,maxiter=300)
        else:
            g=pd.factorize(d[cluster])[0]; f=m.fit(disp=0,maxiter=300,cov_type='cluster',cov_kwds={'groups':g})
        OR=np.exp(f.params[feat]); lo,hi=np.exp(f.conf_int().loc[feat]); p=f.pvalues[feat]
    except Exception as e:
        OR=lo=hi=p=np.nan; P('fail',label,e)
    n1=int(d.y.sum()); n0=int((d.y==0).sum()); e1=d.loc[d.y==1,feat].mean(); e0=d.loc[d.y==0,feat].mean()
    rows.append(dict(model=label,feature=feat,n_err=n1,n_ctrl=n0,err_rate=e1,ctrl_rate=e0,OR=OR,lo=lo,hi=hi,p=p))
    P(f'{label:70s} n={n1}/{n0} {e1:.3f} vs {e0:.3f}  OR {OR:.2f} [{lo:.2f}-{hi:.2f}] p={p:.2g}')
M1=' + C(yb) + res_med + C(ctype)'; M2=M1+' + vhh_i + nonhm + sars_i'
P('\n--- Models (feature = SEQRES starts after IMGT 1, error chain for cases)')
fit('y ~ seq_trunc',D,'crude')
fit('y ~ seq_trunc'+M2,D,'M2 (investigator covariates)')
fit('y ~ seq_trunc'+M2,D,'M2 cluster-robust by paper',cluster='paper')
fit('y ~ seq_trunc'+M2,D,'M2 cluster-robust by last author',cluster='lab')
fit('y ~ seq_trunc'+M2+' + C(mstart)',D,'M2 + model start category (1/2/3/4+)')
fit('y ~ seq_trunc'+M2+' + C(mstart)',D,'M2 + model start, cluster paper',cluster='paper')
fit('y ~ seq_trunc'+M2+' + model_trunc',D,'M2 + model N-term not at IMGT 1 (binary)')
fit('y ~ model_trunc'+M2,D,'M2: model starts after IMGT 1 (feature = model_trunc)',feat='model_trunc')
d=D[D.x_model_first_anarci>1]
fit('y ~ seq_trunc'+M1.replace(' + C(ctype)','')+' + C(ctype) + vhh_i + nonhm + sars_i',d,'Only units whose model starts after IMGT 1: SEQRES trunc vs SEQRES full (unbuilt)')
fit('y ~ seq_trunc + C(mstart)'+' + res_med + C(ctype)',d,'  same, simpler covariates + model start')
d=D[D.x_model_first_anarci==D.x_seq_first_imgt]
P('\nUnits where SEQRES start == model start (the deposited record matches the built model):')
fit('y ~ seq_trunc'+M2,d,'  M2, built model and record start together')
P('\n--- Strata (crude Fisher, feature = seq_trunc)')
def strat(name,mask):
    d=D[mask]; t=pd.crosstab(d.y,d.seq_trunc).reindex(index=[0,1],columns=[0,1],fill_value=0)
    orr,p=stats.fisher_exact(t.values)
    P(f'{name:40s} cases {t.loc[1,1]}/{t.loc[1].sum()} ({t.loc[1,1]/max(1,t.loc[1].sum()):.2f})  ctrl {t.loc[0,1]}/{t.loc[0].sum()} ({t.loc[0,1]/max(1,t.loc[0].sum()):.3f})  OR {orr:.2f} p={p:.2g}')
    rows.append(dict(model='stratum: '+name,feature='seq_trunc',n_err=int(t.loc[1].sum()),n_ctrl=int(t.loc[0].sum()),err_rate=t.loc[1,1]/max(1,t.loc[1].sum()),ctrl_rate=t.loc[0,1]/max(1,t.loc[0].sum()),OR=orr,p=p))
for yb in [(0,2019),(2020,2021),(2022,2023),(2024,2100)]: strat(f'deposit {yb[0]}-{yb[1]}',D.dep_year.between(*yb))
for ct in ['H','K','L']: strat('chain '+ct,D.ctype==ct)
strat('light (K+L)',D.ctype.isin(['K','L']))
strat('res < 3.5',D.res_med<3.5); strat('res >= 3.5',D.res_med>=3.5)
strat('SARS-CoV-2',D.sars_i==1); strat('not SARS-CoV-2',D.sars_i==0)
strat('human/mouse non-nanobody',(D.nonhm==0)&(D.vhh_i==0))
strat('first structure of antibody (n_prior_same_ab=0)',D.n_prior_same_ab==0)
strat('model starts at IMGT 1',D.x_model_first_anarci==1)
strat('model starts at IMGT 2',D.x_model_first_anarci==2)
strat('model starts at IMGT 3',D.x_model_first_anarci==3)
strat('model starts at IMGT >=4',D.x_model_first_anarci>=4)
strat('model starts at IMGT >1',D.x_model_first_anarci>1)
pd.DataFrame(rows).to_csv(f'{OUT}/v03_models.csv',index=False)
open(f'{OUT}/v03_strata.log','w').write('\n'.join(L)+'\n')
