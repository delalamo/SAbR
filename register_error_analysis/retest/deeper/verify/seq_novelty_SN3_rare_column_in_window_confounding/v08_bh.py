"""Collect this verification's tests and apply Benjamini-Hochberg."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
rows=[]
# v01: matched expectation under alternative control sets
R=pd.read_csv(f'{OUT}/v01_unit_modes.csv')
for c in ['clean_EM (investigator)','all_EM_not_first_error','all_EM_units','clean_Xray']:
    e=R['exp_'+c].sum(); o=int(R.obs.sum())
    rows.append(dict(source='v01',test=f'SN3 O/E, controls = {c}',effect=f'O/E {o/e:.1f} ({o}/{len(R)} vs exp {e:.2f})',p=stats.poisson.sf(o-1,e)))
# v03/v06: unit x window models
L=pd.read_csv(f'{OUT}/v06_unit_window_frame.csv.gz')
L['yb']=pd.cut(pd.to_datetime(L.dep).dt.year,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
L['em']=(L.method=='EM').astype(int)
def fit(d,form,lab,term='rc',y='y'):
    d=d.dropna(subset=['yb','res']).copy()
    f=smf.logit(form,data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]},maxiter=200)
    ci=np.exp(f.conf_int().loc[term].values)
    rows.append(dict(source='v06',test=lab,n=len(d),effect=f'OR {np.exp(f.params[term]):.2f} ({ci[0]:.2f}-{ci[1]:.2f})',p=f.pvalues[term]))
# outcome = ANY run in window (incl. copied/artifact), all units: does a rare column predict a run regardless of method?
fit(L[L.window!='C'],'yany ~ rc + C(window) + C(yb) + res + C(ctype)','any run in W ~ rare col, EM+X-ray, excl. FR3 window',y='yany')
for m in ['EM','X-ray']:
    fit(L[(L.method==m)&(L.window!='C')].rename(columns={}),'yany ~ rc + C(window) + C(yb) + res + C(ctype)',f'any run in W ~ rare col, {m} only, excl. FR3 window')
fit(L[L.window!='C'],'yany ~ rc*em + C(window) + C(yb) + res + C(ctype)','any run: rare col x EM interaction (excl. FR3)',term='rc:em')
v6=pd.read_csv(f'{OUT}/v06_models.csv')
for _,r in v6.iterrows():
    rows.append(dict(source='v06',test='first-occurrence run in W: '+r.test,n=r.n,effect=f'OR {r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f}); {int(r.n_y_rc)}/{int(r.n_rc)} rare-col unit-windows are errors',p=r.p))
# v05: high-res X-ray
H=pd.read_csv(f'{OUT}/v05_hires_scan.csv'); H=H[H.note.isna()&~H.runs.fillna('').str.startswith('1-127')]
k=int(H.run_touches_rare.sum()); n=len(H)
rows.append(dict(source='v05',test='high-res (1.1-2.6 A) X-ray chains with a rare column: run touching the rare column vs 4.7% detector background',n=n,
                 effect=f'{k}/{n} = {k/n:.2f}',p=stats.binomtest(k,n,0.047,alternative='greater').pvalue))
fam=H[H.group!='other_rare_ABDE']; k2=int(fam.run_touches_rare.sum())
rows.append(dict(source='v05',test='same column families as the SN3 positives',n=len(fam),effect=f'{k2}/{len(fam)}',p=stats.binomtest(k2,len(fam),0.047,alternative='greater').pvalue))
# v07 recounts
v7=pd.read_csv(f'{OUT}/v07_recount.csv')
for _,r in v7.iterrows():
    rows.append(dict(source='v07',test='SN3 recount: '+r.test.strip(),n=r.n,effect=f'{int(r.obs)}/{int(r.n)} vs exp {r.exp:.2f}, O/E {r.ratio:.1f}',p=r.p))
T=pd.DataFrame(rows); T['q_BH']=multipletests(T.p.clip(lower=1e-300),method='fdr_bh')[1]
T.to_csv(f'{OUT}/all_tests_bh.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',110)
print(T[['test','n','effect','p','q_BH']].to_string())
