"""Finer model-start adjustment (exact first built IMGT position, binned 1,2,3,4,5-6,7-9,>=10) and a Mantel-Haenszel
estimate across model-start strata."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable
warnings.filterwarnings('ignore')
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
D=pd.read_csv(f'{OUT}/units_modeA_analysis.csv.gz',low_memory=False)
D=D[D.x_model_first_anarci.notna()]
D['ms']=pd.cut(D.x_model_first_anarci,[0,1,2,3,4,6,9,200],labels=['1','2','3','4','5-6','7-9','10+']).astype(str)
L=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); L.append(s)
t=D.groupby(['ms','seq_trunc']).y.agg(['sum','count']); t['rate']=(t['sum']/t['count']).round(3); P(t.to_string())
M2=' + C(yb) + res_med + C(ctype) + vhh_i + nonhm + sars_i'
for lab,f in [('M2 + fine model-start bins','y ~ seq_trunc'+M2+' + C(ms)'),('M2 + fine bins, paper-clustered','y ~ seq_trunc'+M2+' + C(ms)')]:
    d=D.copy()
    try:
        if 'cluster' in lab: m=smf.logit(f,data=d).fit(disp=0,maxiter=500,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper.astype(str))[0]})
        else: m=smf.logit(f,data=d).fit(disp=0,maxiter=500)
        P(f'{lab}: OR {np.exp(m.params.seq_trunc):.2f} [{np.exp(m.conf_int().loc["seq_trunc"][0]):.2f}-{np.exp(m.conf_int().loc["seq_trunc"][1]):.2f}] p={m.pvalues.seq_trunc:.2g}')
    except Exception as e: P(lab,'fail',e)
# MH over strata with at least one truncated and one full unit
tabs=[]
for s,d in D[D.x_model_first_anarci>1].groupby('ms'):
    tb=pd.crosstab(d.seq_trunc,d.y).reindex(index=[1,0],columns=[1,0],fill_value=0).values
    if tb.sum(1).min()>0 and tb.sum(0).min()>0: tabs.append(tb.astype(float))
st=StratifiedTable(tabs)
P(f'Mantel-Haenszel OR across model-start strata (start>1): {st.oddsratio_pooled:.2f} [{st.oddsratio_pooled_confint()[0]:.2f}-{st.oddsratio_pooled_confint()[1]:.2f}] p={st.test_null_odds().pvalue:.2g}; homogeneity p={st.test_equal_odds().pvalue:.2g}')
open(f'{OUT}/v07_finer.log','w').write('\n'.join(L)+'\n')
