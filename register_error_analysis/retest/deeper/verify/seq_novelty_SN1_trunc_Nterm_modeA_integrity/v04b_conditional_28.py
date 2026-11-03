"""Incremental effect of SEQRES truncation beyond an unmodelled N-terminus, using the claim's 'starts at IMGT 2-8' definition:
units with SEQRES starting at IMGT 1-8 and first modelled residue at IMGT 2-8; M2 + first-modelled position."""
import pandas as pd, numpy as np, warnings, statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{T}/units.csv.gz',low_memory=False)
R=pd.read_csv(f'{SP}/deeper/cache/verify_SN1/first_modelled.csv.gz')
U=U.merge(R[['pdb','chain','first_mod_anarci']],on=['pdb','chain'],how='left')
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100]).astype(str)
U['nonhm']=(~U.q_species_all.isin(['human','mouse'])).astype(int); U['vhh_i']=U.vhh_u.fillna(0).astype(int); U['sars_i']=U.sars2_u.astype(int)
EM=U.method_u=='EM'; A=EM&U.err_first_u&U.modes_u.fillna('').str.contains('A_FR1Nterm'); Cm=EM&U.clean
d=U[A|Cm].copy(); d['y']=A[A|Cm].astype(int)
d=d[d.seq_first_imgt.between(1,8)&d.first_mod_anarci.between(2,8)].copy()
d['trunc']=(d.seq_first_imgt>1).astype(int); d['fm']=d.first_mod_anarci.clip(upper=5).astype(int).astype(str)
print('n',d.y.sum(),(d.y==0).sum()); print(pd.crosstab([d.fm,d.trunc],d.y))
M2='trunc + C(yb) + res_med + C(ctype) + vhh_i + nonhm + sars_i'
for lab,f in [('M2',M2),('M2 + first-modelled cat',M2+' + C(fm)')]:
    r=smf.logit('y ~ '+f,data=d).fit(disp=0); print(lab,'OR %.2f (%.2f-%.2f) p %.3g'%(np.exp(r.params['trunc']),*np.exp(r.conf_int().loc['trunc']),r.pvalues['trunc']))
st=StratifiedTable([pd.crosstab(g.trunc,g.y).reindex(index=[1,0],columns=[1,0],fill_value=0).values+0.0 for _,g in d.groupby('fm')])
print('CMH OR %.2f p %.3g'%(st.oddsratio_pooled,st.test_null_odds().pvalue))
