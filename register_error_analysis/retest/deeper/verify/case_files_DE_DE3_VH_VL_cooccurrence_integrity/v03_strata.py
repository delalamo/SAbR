"""Extra strata for DE3: SARS-CoV-2 vs not, 2020-21 only, entries with one Fab only; exact p at paper level;
investigator's own logit (cluster SE) vs GEE exchangeable for the L_oth outcome."""
import pandas as pd, numpy as np, warnings
from scipy.stats import fisher_exact
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE3_VH_VL_cooccurrence_integrity'
D='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
T=pd.read_csv(f'{OUT}/entry_table.csv')
CH=pd.read_csv(f'{D}/chains.csv.gz')
s2=CH.groupby('pdb').sars2.max(); T['sars2']=T.pdb.map(s2).astype(bool)
T['yb']=pd.cut(T.year,[0,2019,2021,2023,2030]).astype(str)
rows=[]
def f(t,label,a='H_DE',b='L_any'):
    tab=pd.crosstab(t[a],t[b]).reindex(index=[False,True],columns=[False,True],fill_value=0)
    o,p=fisher_exact(tab.values)
    rows.append(dict(test=label,n=len(t),n_given=int(t[a].sum()),events=int((t[a]&t[b]).sum()),rate_given=t[t[a]][b].mean(),rate_not=t[~t[a]][b].mean(),OR=o,p=p,papers_event=t[t[a]&t[b]].paper.nunique()))
f(T[T.sars2],'SARS-CoV-2 entries'); f(T[~T.sars2],'non-SARS-CoV-2 entries')
f(T[T.year.between(2020,2021)],'deposited 2020-21'); f(T[~T.year.between(2020,2021)],'deposited outside 2020-21')
f(T[(T.nH==1)&(T.nL==1)],'single-Fab entries (1 VH, 1 VL)')
P=T.groupby('paper').agg(H_DE=('H_DE','max'),L_any=('L_any','max'),paper=('paper','first'))
f(P,'paper level exact')
O=pd.DataFrame(rows); O.to_csv(f'{OUT}/strata.csv',index=False)
pd.set_option('display.width',220); print(O.round(5).to_string())
# investigator logit (cluster) vs GEE exchangeable, L_oth outcome
z=T.assign(y=T.L_oth.astype(int),x=T.H_DE.astype(int)).sort_values('paper')
m=smf.logit('y ~ x + C(yb) + res',data=z).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(z.paper)[0]})
g=smf.gee('y ~ x + C(yb) + res',groups=pd.factorize(z.paper)[0],data=z,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
print('L_oth: logit-cluster OR %.1f p %.2g | GEE-exch OR %.1f p %.2g'%(np.exp(m.params['x']),m.pvalues['x'],np.exp(g.params['x']),g.pvalues['x']))
