"""Re-fit the investigator's M1/M2/M3 for seq_nt_trunc (mode A), then challenge it:
(1) adjust for / stratify on 'N-terminus not modelled' (first modelled residue IMGT>1, the known prior finding);
(2) leave-one-lab-out / leave-one-paper-out; (3) period / chain-type strata; (4) alternative control set."""
import pandas as pd, numpy as np, warnings, statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{T}/units.csv.gz',low_memory=False)
R=pd.read_csv(f'{SP}/deeper/cache/verify_SN1/first_modelled.csv.gz')
U=U.merge(R[['pdb','chain','first_mod_anarci','first_mod_sabr','first_mod_seqidx']],on=['pdb','chain'],how='left')
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=['<=2018','2019','2020','2021','2022','2023','>=2024']).astype(str)
U['nonhm']=(~U.q_species_all.isin(['human','mouse'])).astype(int)
U['vhh_i']=U.vhh_u.fillna(0).astype(int); U['sars_i']=U.sars2_u.astype(int)
U['trunc']=U.seq_nt_trunc.astype(int)
U['unbuilt']=(U.first_mod_anarci>1).astype(float); U.loc[U.first_mod_anarci.isna(),'unbuilt']=np.nan
# 'unbuilt but present in SEQRES' = first modelled residue > first SEQRES residue position
U['unbuilt_in_seqres']=((U.first_mod_anarci>U.seq_first_imgt)).astype(float); U.loc[U.first_mod_anarci.isna(),'unbuilt_in_seqres']=np.nan
EM=U.method_u=='EM'
A=EM&U.err_first_u&U.modes_u.fillna('').str.contains('A_FR1Nterm')
Cm=EM&U.clean&U.ctype.isin(U[A].ctype.unique())
d=U[A|Cm].copy(); d["y"]=A[A|Cm].astype(int)
out=[]
def fit(df,form,term='trunc',label=''):
    try:
        f=smf.logit('y ~ '+form,data=df).fit(disp=0,maxiter=300)
        OR=np.exp(f.params[term]); lo,hi=np.exp(f.conf_int().loc[term]); p=f.pvalues[term]
    except Exception as e:
        OR=lo=hi=p=np.nan
    r=dict(test=label,term=term,n_err=int(df.y.sum()),n_ctrl=int((df.y==0).sum()),OR=OR,lo=lo,hi=hi,p=p); out.append(r); return r
M1='trunc + C(yb) + res_med + C(ctype)'
M2=M1+' + vhh_i + nonhm + sars_i'
print('raw', pd.crosstab(d.y,d.trunc), stats.fisher_exact(pd.crosstab(d.y,d.trunc).values))
print(fit(d,M1,label='M1 reproduce'))
print(fit(d,M2,label='M2 reproduce'))
d3=d[(d.n_prior_same_ab==0)&(d.nonhm==0)&(d.vhh_i==0)]
print(fit(d3,M2.replace(' + vhh_i + nonhm',''),label='M3 reproduce'))
# --- overlap with N-terminus-not-modelled
dd=d.dropna(subset=['unbuilt'])
print('\nN-term not modelled x trunc x case:'); print(pd.crosstab([dd.unbuilt,dd.trunc],dd.y,margins=True))
print('rate by group:'); print(dd.groupby(['unbuilt','trunc']).y.agg(['mean','sum','size']))
print(fit(dd,'unbuilt + C(yb) + res_med + C(ctype) + vhh_i + nonhm + sars_i',term='unbuilt',label='M2 unbuilt N-term (any cause)'))
print(fit(dd,M2+' + unbuilt',label='M2 trunc + unbuilt'))
s=dd[dd.unbuilt==1]
print(fit(s,M2,label='M2 within N-term-not-modelled: trunc vs residues in SEQRES but unmodelled'))
print('fisher within unbuilt', stats.fisher_exact(pd.crosstab(s.y,s.trunc).values), pd.crosstab(s.y,s.trunc).values.tolist())
# matched on first-modelled IMGT position (2,3,4+)
s=s.copy(); s['fm']=s.first_mod_anarci.clip(upper=5).astype(int).astype(str)
print(fit(s,M2+' + C(fm)',label='M2 within unbuilt + first-modelled position'))
for k,g in s.groupby('fm'):
    t=pd.crosstab(g.y,g.trunc).reindex(index=[0,1],columns=[0,1],fill_value=0)
    print('first modelled IMGT',k,'ctrl [untrunc,trunc]',t.loc[0].tolist(),'case [untrunc,trunc]',t.loc[1].tolist(),'rate trunc %.3f untrunc %.3f'%(t.loc[1,1]/max(1,t[1].sum()),t.loc[1,0]/max(1,t[0].sum())))
# CMH over first-modelled strata
from statsmodels.stats.contingency_tables import StratifiedTable
tabs=[pd.crosstab(g.trunc,g.y).reindex(index=[1,0],columns=[1,0],fill_value=0).values+0.0 for k,g in s.groupby('fm')]
st=StratifiedTable(tabs); print('CMH OR within unbuilt, stratified by first-modelled pos: %.2f p=%.3g'%(st.oddsratio_pooled,st.test_null_odds().pvalue))
# --- leave-one-lab-out / paper-out on M2
cases=d[d.y==1]
for key in ['last_author','paper']:
    ors=[];ps=[]
    for v in cases[key].dropna().unique():
        r=smf.logit('y ~ '+M2,data=d[d[key]!=v]).fit(disp=0,maxiter=300); ors.append(np.exp(r.params['trunc'])); ps.append(r.pvalues['trunc'])
    print(f'leave-one-{key}-out: OR range {min(ors):.2f}-{max(ors):.2f}, max p {max(ps):.2g}')
    out.append(dict(test=f'leave-one-{key}-out M2',OR=min(ors),hi=max(ors),p=max(ps)))
# leave-two-labs: drop the labs of truncated cases two at a time (worst case)
labs=cases[cases.trunc==1].last_author.unique()
print('labs among truncated cases',len(labs),'papers',cases[cases.trunc==1].paper.nunique())
# --- strata
for lab,sub in [('heavy',d[d.ctype=='H']),('kappa',d[d.ctype=='K']),('lambda',d[d.ctype=='L']),('dep<=2021',d[d.dep_year<=2021]),('dep>=2022',d[d.dep_year>=2022]),('non-SARS2',d[d.sars_i==0])]:
    t=pd.crosstab(sub.y,sub.trunc).reindex(index=[0,1],columns=[0,1],fill_value=0)
    fe=stats.fisher_exact(t.values)
    print(f'{lab}: case {t.loc[1,1]}/{t.loc[1].sum()} ({t.loc[1,1]/t.loc[1].sum():.3f}) ctrl {t.loc[0,1]}/{t.loc[0].sum()} ({t.loc[0,1]/t.loc[0].sum():.3f}) OR {fe[0]:.2f} p {fe[1]:.2g}')
    out.append(dict(test='stratum '+lab,n_err=int(t.loc[1].sum()),n_ctrl=int(t.loc[0].sum()),err=t.loc[1,1]/t.loc[1].sum(),ctrl=t.loc[0,1]/t.loc[0].sum(),OR=fe[0],p=fe[1]))
# --- alternative control: EM units of same ctypes with no mode A run (errors elsewhere allowed) and no copied/artifact A run
Cm2=EM&~U.modes_u.fillna('').str.contains('A_FR1Nterm')&~A&U.ctype.isin(U[A].ctype.unique())
d2=U[A|Cm2].copy(); d2["y"]=A[A|Cm2].astype(int)
print(fit(d2,M2,label='M2 control = all EM units without first-occurrence mode A'))
pd.DataFrame(out).to_csv('v04_model_results.csv',index=False)
