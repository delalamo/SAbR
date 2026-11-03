"""Does SEQRES truncation add anything beyond 'the model starts after IMGT 1'?
Restrict to EM case/control units whose SEQRES starts at IMGT 1-8 and whose first built V residue is IMGT 2-8
(so every unit has an N-terminal strand and an unbuilt/absent IMGT 1). Compare truncated SEQRES (2-8) with
full SEQRES (residue 1 present in SEQRES but not built), stratified/adjusted for first-built position."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_integrity'
U=pd.read_pickle(f'{SP}/deeper/cache/verify_SN1/units_verify.pkl')
U['nonhm']=0
sp=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables/chain_novelty.csv.gz',usecols=['pdb','chain','q_species_all'],low_memory=False)
U=U.drop(columns=[c for c in ['q_species_all'] if c in U]).merge(sp,on=['pdb','chain'],how='left')
U['nonhm']=(~U.q_species_all.isin(['human','mouse'])).astype(int)
case=U.any_EM&U.caseA_u; ctrl=U.any_EM&~U.err_any_u&U.ctype.isin(['H','K','L'])
D=U[case|ctrl].copy(); D['y']=case[case|ctrl].astype(int)
log=open(f'{OUT}/v04_conditional.log','w')
def P(*x):
    s=' '.join(str(i) for i in x); print(s); log.write(s+'\n')
S=D[D.seq_first_imgt.between(1,8)&D.built_first_anarci.between(2,8)].copy()
S['fb']=S.built_first_anarci.clip(upper=5).astype(int).astype(str)
P('subset: first built IMGT 2-8, SEQRES start 1-8: cases',S.y.sum(),'controls',(S.y==0).sum())
P(pd.crosstab([S.fb,S.trunc],S.y,margins=True).to_string())
tabs=[]
for k,gg in S.groupby('fb'):
    t=pd.crosstab(gg.trunc,gg.y).reindex(index=[1,0],columns=[1,0],fill_value=0).values.astype(float)
    P(f'  fb={k}: trunc err {int(t[0,0])}/{int(t[0].sum())} ({t[0,0]/max(t[0].sum(),1):.3f})  full-unbuilt err {int(t[1,0])}/{int(t[1].sum())} ({t[1,0]/max(t[1].sum(),1):.3f})')
    if (t.sum(1)>0).all(): tabs.append(t)
st=StratifiedTable(tabs)
P(f'MH OR truncated vs full-but-unbuilt, strata = first built position: {st.oddsratio_pooled:.2f} ({st.oddsratio_pooled_confint()[0]:.2f}-{st.oddsratio_pooled_confint()[1]:.2f}) p={st.test_null_odds().pvalue:.3g}; homogeneity p={st.test_equal_odds().pvalue:.3g}')
a1=S[S.y==1].trunc.sum(); n1=S.y.sum(); a0=S[S.y==0].trunc.sum(); n0=(S.y==0).sum()
P(f'raw: trunc share err {a1}/{n1} ({a1/n1:.3f}) ctrl {a0}/{n0} ({a0/n0:.3f}) Fisher OR {stats.fisher_exact([[a1,n1-a1],[a0,n0-a0]])[0]:.2f} p={stats.fisher_exact([[a1,n1-a1],[a0,n0-a0]])[1]:.3g}')
M2=' + C(yb) + res_med + C(ctype) + vhh_u + nonhm + sars_i'
for f,lab in [('y ~ trunc'+M2,'M2'),('y ~ trunc + C(fb)'+M2,'M2 + first-built position'),('y ~ trunc + built_first_anarci'+M2,'M2 + first-built (linear)')]:
    try:
        fit=smf.logit(f,data=S).fit(disp=0,maxiter=300)
        o=np.exp(fit.params['trunc']); lo,hi=np.exp(fit.conf_int().loc['trunc'])
        P(f'{lab:32s} OR {o:.2f} ({lo:.2f}-{hi:.2f}) p={fit.pvalues["trunc"]:.3g}')
    except Exception as e: P(lab,'failed',e)
# full-sample model with first-built category as a covariate (all EM case/control units)
D['fbc']=D.built_first_anarci.clip(upper=5).fillna(1).astype(int).astype(str)
D['trunc28']=D.seq_first_imgt.between(2,8).astype(int)
for f,lab,t in [('y ~ trunc'+M2,'all units M2 (claim)','trunc'),('y ~ trunc + C(fbc)'+M2,'all units M2 + first-built category','trunc'),
                ('y ~ trunc28 + C(fbc)'+M2,'all units M2 + first-built cat, trunc 2-8','trunc28')]:
    fit=smf.logit(f,data=D).fit(disp=0,maxiter=300)
    o=np.exp(fit.params[t]); lo,hi=np.exp(fit.conf_int().loc[t])
    P(f'{lab:45s} OR {o:.2f} ({lo:.2f}-{hi:.2f}) p={fit.pvalues[t]:.3g}')
# how many truncated cases have their first SEQRES residue built (SEQRES == modelled residues at the N-terminus)?
c=D[(D.y==1)&(D.trunc==1)]
P('truncated cases with first SEQRES residue built:',int((c.built_first_anarci==c.seq_first_imgt).sum()),'/',len(c))
k=D[(D.y==0)&(D.trunc==1)&D.seq_first_imgt.between(2,8)]
P('truncated (2-8) controls with first SEQRES residue built:',int((k.built_first_anarci==k.seq_first_imgt).sum()),'/',len(k))
