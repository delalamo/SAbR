"""DE2 check: does map-model fit (lq_z, chain level) or resolution explain the framework-RMSD difference?"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy.stats import spearmanr
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_integrity'
A=pd.read_csv(f'{OUT}/analysis_table.csv')
CH=pd.read_csv(f'{SP}/deeper/chains.csv.gz')[['pdb','chain','lq','lq_z']]
A=A.merge(CH,on=['pdb','chain'],how='left')
e=A[(A.group=='error')&(A.confirm!='not confirmed')]; mc=A[A.group=='matched_clean']; wp=A[(A.group=='within_paper_clean')&A.paper.isin(e.paper)]
rows=[]
d=pd.concat([e.assign(err=1),mc.assign(err=0)]).dropna(subset=['rep_rmsd','lq_z'])
d['x']=(d.rep_rmsd-d.rep_rmsd.mean())/d.rep_rmsd.std(); d['lz']=(d.lq_z-d.lq_z.mean())/d.lq_z.std()
for f,lab in [('err ~ x + C(yb) + res + C(ct)','base'),('err ~ x + lz + C(yb) + res + C(ct)','+ lq_z'),('err ~ lz + C(yb) + res + C(ct)','lq_z only')]:
    m=smf.logit(f,data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
    rows.append(dict(test='matched logit '+lab,n_err=int(d.err.sum()),n_ctl=int((d.err==0).sum()),
      OR_rmsd=np.exp(m.params.get('x',np.nan)),p_rmsd=m.pvalues.get('x',np.nan),OR_lqz=np.exp(m.params.get('lz',np.nan)),p_lqz=m.pvalues.get('lz',np.nan)))
r=spearmanr(d.rep_rmsd,d.lq_z); rows.append(dict(test='spearman rmsd vs lq_z (matched sample)',OR_rmsd=r.correlation,p_rmsd=r.pvalue))
# within paper with resolution and lq_z
z=pd.concat([e.assign(err=1),wp.assign(err=0)]).dropna(subset=['rep_rmsd','lq_z'])
ok=z.groupby('paper').err.agg(['min','max']); ok=ok[(ok['min']==0)&(ok['max']==1)].index; z=z[z.paper.isin(ok)].copy()
for c_ in ['rep_rmsd','res','lq_z']: z[c_+'_s']=(z[c_]-z[c_].mean())/z[c_].std()
for cols,lab in [(['rep_rmsd_s'],'base'),(['rep_rmsd_s','res_s'],'+res'),(['rep_rmsd_s','lq_z_s'],'+lq_z'),(['rep_rmsd_s','res_s','lq_z_s'],'+res+lq_z')]:
    m=ConditionalLogit(z.err.values,z[cols].values,groups=pd.factorize(z.paper)[0]).fit(disp=0)
    rows.append(dict(test='within-paper clogit '+lab,n_err=int(z.err.sum()),n_ctl=int((z.err==0).sum()),n_papers=z.paper.nunique(),
      OR_rmsd=np.exp(m.params[0]),p_rmsd=m.pvalues[0],OR_lqz=np.exp(m.params[-1]) if 'lq_z_s' in cols else np.nan,p_lqz=m.pvalues[-1] if 'lq_z_s' in cols else np.nan))
rows.append(dict(test='within-paper resolution err vs ctl (mean)',OR_rmsd=z[z.err==1].res.mean(),OR_lqz=z[z.err==0].res.mean()))
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/verify_mapfit_adjust.csv',index=False)
pd.set_option('display.width',250); print(T.round(4).to_string())
