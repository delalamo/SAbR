"""Population check inside the 3-5 A scan: given that a chain/unit carries a residue at a rarely occupied IMGT column in
window W, how often does the scan report a disagreement run in W (any run, incl. copied/artifact), and does that depend on
method (EM vs X-ray) or period? If an insertion at a rare column produces a run in X-ray and pre-2019 structures as often as
in 2020-21 cryo-EM, the SN3 association is produced by the detector (insertion-placement convention), not by modelling."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
ch=pd.read_csv(f'{SN}/chain_novelty.csv.gz',low_memory=False)
ra=pd.read_csv(f'{SP}/deeper/runs_all.csv'); rf=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
WIN={'A':(1,13),'F':(24,41),'B':(55,72),'C':(74,92),'D':(105,128),'E':(116,128)}
def overl(r,a,b): return (r.imgt0-2<=b)&(r.imgt1+2>=a)
for w,(a,b) in WIN.items():
    for nm,R in (('any',ra),('first',rf)):
        hit=R[overl(R,a,b)][['pdb','chain']].drop_duplicates(); hit[f'run_{nm}_{w}']=True
        ch=ch.merge(hit,on=['pdb','chain'],how='left'); ch[f'run_{nm}_{w}']=ch[f'run_{nm}_{w}'].fillna(False).astype(bool)
ch['period']=pd.cut(pd.to_datetime(ch.deposit).dt.year,[0,2018,2021,2100],labels=['<=2018','2019-21','>=2022'])
ch=ch[ch.dbrow>=0]
rows=[]
for w in WIN:
    ch[f'rc_{w}']=ch[f'rareins_win_{w}'].fillna(0)>0
    for meth in ['EM','X-ray']:
        for per in ['all','<=2018','2019-21','>=2022']:
            d=ch[(ch.method==meth)&((ch.period==per) if per!='all' else True)]
            for flag in (True,False):
                x=d[d[f'rc_{w}']==flag]
                rows.append(dict(window=w,method=meth,period=per,rare_col=flag,n_chains=len(x),n_units=x.unit.nunique(),
                                 run_any=x[f'run_any_{w}'].mean() if len(x) else np.nan,run_first=x[f'run_first_{w}'].mean() if len(x) else np.nan,
                                 units_run_any=x.groupby('unit')[f'run_any_{w}'].any().mean() if len(x) else np.nan))
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/v03_run_given_rarecol.csv',index=False)
pd.set_option('display.width',200)
print(T[T.period=='all'].round(3).to_string(index=False))
print(T[(T.rare_col)&(T.period!='all')].round(3).to_string(index=False))
# pooled over windows: P(run in W | rare col in W) by method; Fisher EM vs X-ray among rare-col chains (unit-collapsed)
print('\nPooled across windows, unit level (unit x window with rare column):')
L=[]
for w in WIN:
    g=ch.groupby(['unit',ch.method]).agg(rc=(f'rc_{w}','any'),run=(f'run_any_{w}','any'),runf=(f'run_first_{w}','any'),
                                         yr=('year','min'),sars2=('sars2','any'),res=('res','median'),ctype=('ctype','first'),
                                         paper=('paper','first')).reset_index()
    g['window']=w; L.append(g)
L=pd.concat(L,ignore_index=True); L.to_csv(f'{OUT}/v03_unit_window_frame.csv.gz',index=False)
for meth in ['EM','X-ray']:
    for rc in (True,False):
        x=L[(L.method==meth)&(L.rc==rc)]
        print(f'{meth:6s} rare_col={rc!s:5s} unit-windows={len(x):6d} with run(any)={x.run.sum():5d} ({x.run.mean():.3f}) first={x.runf.sum()} ({x.runf.mean():.3f})')
# odds ratio of run given rare column, EM vs X-ray, logistic with window FE, cluster by paper
L['y']=L.run.astype(int); L['rc_i']=L.rc.astype(int); L['em']=(L.method=='EM').astype(int)
L['yb']=pd.cut(L.yr,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
for meth in ['EM','X-ray']:
    d=L[L.method==meth].dropna(subset=['yb']).copy(); d['paper']=d.paper.fillna('na')
    try:
        f=smf.logit('y ~ rc_i + C(window) + C(yb)',data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
        print(meth,'OR(run | rare col) =',round(np.exp(f.params['rc_i']),1),'p=',f.pvalues['rc_i'])
    except Exception as e: print(meth,e)
d=L[L.rc].copy(); d['paper']=d.paper.fillna('na')
ct=pd.crosstab(d.method,d.run); print(ct); print('Fisher EM vs X-ray among rare-col unit-windows',stats.fisher_exact(ct.values))

print('\nExcluding window C (FR3 insertion already treated as convention artifact by the pipeline):')
for meth in ['EM','X-ray']:
    for rc in (True,False):
        x=L[(L.method==meth)&(L.rc==rc)&(L.window!='C')]
        print(f'{meth:6s} rare_col={rc!s:5s} unit-windows={len(x):6d} run(any)={x.run.sum():4d} ({x.run.mean():.3f}) first={x.runf.sum()} ({x.runf.mean():.3f})')
for meth in ['EM','X-ray']:
    d=L[(L.method==meth)&(L.window!='C')].dropna(subset=['yb']).copy(); d['paper']=d.paper.fillna('na')
    f=smf.logit('y ~ rc_i + C(window) + C(yb)',data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
    print(meth,'excl C: OR(run | rare col) =',round(np.exp(f.params['rc_i']),1),'CI',np.exp(f.conf_int().loc['rc_i'].values).round(1),'p=',f.pvalues['rc_i'])
d=L[L.rc&(L.window!='C')]
d2=L[(L.window!='C')].dropna(subset=['yb']).copy(); d2['paper']=d2.paper.fillna('na')
f=smf.logit('y ~ rc_i*em + C(window) + C(yb)',data=d2).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d2.paper)[0]})
print('interaction rc x EM (excl C): ratio of ORs',round(np.exp(f.params['rc_i:em']),2),'p',f.pvalues['rc_i:em'])
