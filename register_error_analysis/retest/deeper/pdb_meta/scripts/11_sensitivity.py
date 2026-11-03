"""Sensitivity of the leading entry-level signals: period strata, paper-weighted (one vote per paper), +SARS-CoV-2 adjustment,
+entry Q-score adjustment, with/without amyloid fibrils."""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
d=pd.read_csv(f'{OUT}/em_entry_analysis_frame.csv.gz')
d['period']=np.where(d.dep_year<=2019,'<=2019',np.where(d.dep_year<=2021,'2020-21','>=2022'))
d['sars2']=d.sars2.astype(float)
top=[('vg_clashscore','cont'),('non_clash','cont'),('ab_clash','cont'),('vg_percent_ramachandran_outliers','cont'),('ab_cis_zero','bin'),('dq_ab_minus_non','cont'),
     ('krios','bin'),('kv300','bin'),('map_pix','cont'),('det_K3','bin'),('kw_membrane','bin'),('fund_none','bin'),('emdb_n_fields','cont'),('has_half_maps','bin'),
     ('log2_tot_mw','cont'),('ab_small_part_lt15pct','bin'),('any_local_or_focused','bin'),('log10_particles','cont'),('vem_Q_score','cont'),('proc_PDBC','bin')]
rows=[]
def wfit(x,feat,kind,cov):
    x=x[['y',feat,'cluster','w']+[c.replace('C(','').replace(')','') for c in cov]].dropna()
    s=1 if kind=='bin' else x[feat].std()
    x=x.assign(_x=x[feat]/s)
    m=smf.glm('y ~ _x + '+' + '.join(cov),data=x,family=sm.families.Binomial(),freq_weights=x.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.cluster)[0]})
    return np.exp(m.params['_x']),m.pvalues['_x']
d['w']=1/d.groupby('cluster').pdb.transform('size')
for f,k in top:
    base=['C(ybin)','res','l2nV']
    r={'feature':f}
    a=fit_feature(d,f,k,base); r['OR_main']=a.get('OR'); r['p_main']=a.get('p')
    for per in ['<=2019','2020-21','>=2022']:
        sub=d[d.period==per]; a=fit_feature(sub,f,k,base)
        r[f'OR_{per}']=a.get('OR'); r[f'p_{per}']=a.get('p'); r[f'n_err_{per}']=a.get('n_err')
    a=fit_feature(d,f,k,base+['sars2']); r['OR_+sars2']=a.get('OR'); r['p_+sars2']=a.get('p')
    if f!='vem_Q_score':
        a=fit_feature(d,f,k,base+['vem_Q_score']); r['OR_+entryQ']=a.get('OR'); r['p_+entryQ']=a.get('p')
    try: r['OR_paperweighted'],r['p_paperweighted']=wfit(d,f,k,base)
    except Exception as ex: r['note']=str(ex)[:60]
    rows.append(r)
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/entry_sensitivity.csv',index=False)
pd.set_option('display.width',260)
print(R.to_string(index=False,float_format=lambda x:f'{x:.3g}'))
