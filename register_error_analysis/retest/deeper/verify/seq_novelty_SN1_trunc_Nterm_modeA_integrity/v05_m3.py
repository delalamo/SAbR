"""Reproduce M3 (first structure of the antibody, human/mouse, non-nanobody) and add first-built position."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_integrity'
V=pd.read_pickle(f'{SP}/deeper/cache/verify_SN1/units_verify.pkl').drop(columns=['q_species_all'],errors='ignore')
U=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables/units.csv.gz',low_memory=False,usecols=['unit','n_prior_same_ab','q_species_all'])
V=V.merge(U,on='unit',how='left')
V['nonhm']=(~V.q_species_all.isin(['human','mouse'])).astype(int)
case=V.any_EM&V.caseA_u; ctrl=V.any_EM&~V.err_any_u
D=V[case|ctrl].copy(); D['y']=case[case|ctrl].astype(int)
D3=D[(D.n_prior_same_ab==0)&(D.nonhm==0)&(D.vhh_u==0)].copy()
D3['fbc']=D3.built_first_anarci.clip(upper=5).fillna(1).astype(int).astype(str)
D3['trunc28']=D3.seq_first_imgt.between(2,8).astype(int)
log=open(f'{OUT}/v05_m3.log','w')
def P(*x):
    s=' '.join(str(i) for i in x); print(s); log.write(s+'\n')
P('M3 n',D3.y.sum(),(D3.y==0).sum(),'trunc err',round(D3[D3.y==1].trunc.mean(),3),'ctrl',round(D3[D3.y==0].trunc.mean(),3))
for f,t in [('y ~ trunc + C(yb) + res_med + C(ctype) + sars_i','trunc'),
            ('y ~ trunc + C(fbc) + C(yb) + res_med + C(ctype) + sars_i','trunc'),
            ('y ~ trunc28 + C(fbc) + C(yb) + res_med + C(ctype) + sars_i','trunc28')]:
    fit=smf.logit(f,data=D3).fit(disp=0,maxiter=300)
    P(f'{f:70s} OR {np.exp(fit.params[t]):.2f} {np.round(np.exp(fit.conf_int().loc[t]).values,2).tolist()} p={fit.pvalues[t]:.3g}')
