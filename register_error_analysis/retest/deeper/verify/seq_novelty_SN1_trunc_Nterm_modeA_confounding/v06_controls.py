"""Control-definition sensitivity: 'clean' controls exclude units with ANY disagreement run (incl. copied/artifact).
Check truncation rates in the excluded groups and refit with broader controls (all EM units without a first-occurrence
mode-A error). Also residual SEQRES effect inside model-start strata 2-3 with covariates."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf, json
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
U=pd.read_csv(f'{SN}/units.csv.gz',low_memory=False)
CN=pd.read_csv(f'{OUT}/chain_nterm.csv.gz')
rep=CN.merge(pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','deposit']),on=['pdb','chain'])
rep['deposit']=pd.to_datetime(rep.deposit); rep=rep.sort_values(['unit','deposit','pdb','chain']).groupby('unit').first()
U=U.join(rep[['model_first_anarci']],on='unit')
runsall=pd.read_csv(f'{SP}/deeper/runs_all.csv')
EM=U.method_u=='EM'
U['caseA']=EM&U.err_first_u&U.modes_u.str.contains('A_FR1Nterm')
U['grp']=np.select([U.caseA, EM&U.clean, EM&U.err_first_u, EM&U.err_any_u&~U.err_first_u],['A case','clean','other first-occ error','copied/artifact only'],'other')
L=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); L.append(s)
g=U[EM].groupby('grp').agg(n=('unit','size'),seq_trunc=('seq_nt_trunc','mean'),model_start_gt1=('model_first_anarci',lambda x:(x>1).mean()))
P('EM units by group: SEQRES truncation and model-start>1 rates'); P(g.to_string())
# does the excluded copied/artifact-only group carry N-terminal runs?
ra=runsall.copy(); ra['nterm']=ra.imgt0<=10
ex=U[EM&(U.grp=='copied/artifact only')]
P('\ncopied/artifact-only EM units with N-terminal (IMGT<=10) runs:')
chunit=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','unit'])
ra=ra.merge(chunit,on=['pdb','chain'])
nt_units=set(ra[ra.nterm].unit)
ex2=ex.assign(nterm_run=ex.unit.isin(nt_units))
P(ex2.groupby('nterm_run').agg(n=('unit','size'),seq_trunc=('seq_nt_trunc','mean')).to_string())
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=['<=2018','2019','2020','2021','2022','2023','>=2024']).astype(str)
U['nonhm']=(~U.q_species_all.isin(['human','mouse'])).astype(int); U['vhh_i']=U.vhh_u.fillna(0).astype(int); U['sars_i']=U.sars2_u.astype(int)
U['seq_trunc']=U.seq_nt_trunc.astype(int); U['y']=U.caseA.astype(int)
U['mstart']=pd.cut(U.model_first_anarci,[0,1,2,3,100],labels=['1','2','3','4+']).astype(str)
M2=' + C(yb) + res_med + C(ctype) + vhh_i + nonhm + sars_i'
def fit(d,f,label,feat='seq_trunc'):
    m=smf.logit(f,data=d).fit(disp=0,maxiter=300,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper.astype(str))[0]})
    OR=np.exp(m.params[feat]); lo,hi=np.exp(m.conf_int().loc[feat])
    P(f'{label:75s} n={int(d.y.sum())}/{int((d.y==0).sum())} {d.loc[d.y==1,feat].mean():.3f} vs {d.loc[d.y==0,feat].mean():.3f} OR {OR:.2f} [{lo:.2f}-{hi:.2f}] p={m.pvalues[feat]:.2g}')
    return dict(model=label,n_err=int(d.y.sum()),n_ctrl=int((d.y==0).sum()),OR=OR,lo=lo,hi=hi,p=m.pvalues[feat])
rows=[]
P('\n--- broader controls: all EM units without a first-occurrence mode-A error (cluster-robust by paper)')
d=U[EM&U.ctype.isin(['H','K','L'])&U.model_first_anarci.notna()].copy()
rows.append(fit(d,'y ~ seq_trunc'+M2,'broad controls, M2'))
rows.append(fit(d,'y ~ seq_trunc'+M2+' + C(mstart)','broad controls, M2 + model start'))
P('\n--- clean controls, cluster-robust by paper, M2 + model start')
d=U[(U.caseA|(EM&U.clean))&U.ctype.isin(['H','K','L'])&U.model_first_anarci.notna()].copy()
rows.append(fit(d,'y ~ seq_trunc'+M2+' + C(mstart)','clean controls, M2 + model start'))
rows.append(fit(d,'y ~ seq_trunc'+M2+' + model_first_anarci','clean controls, M2 + model start (linear IMGT)'))
d2=d[d.model_first_anarci.isin([2,3])]
rows.append(fit(d2,'y ~ seq_trunc + C(mstart) + res_med + C(ctype) + sars_i','model start 2-3 only (residual SEQRES effect)'))
d3=d[d.model_first_anarci>=2]
rows.append(fit(d3,'y ~ seq_trunc + C(mstart) + res_med + C(ctype) + sars_i + vhh_i','model start >=2 (SEQRES-trunc vs unbuilt-but-in-SEQRES)'))
pd.DataFrame(rows).to_csv(f'{OUT}/v06_models.csv',index=False)
open(f'{OUT}/v06_controls.log','w').write('\n'.join(L)+'\n')
