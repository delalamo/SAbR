"""Reproduce: EM entries whose CLEAN kappa chains (Pro8 modelled) are all trans at Pro8 have more heavy-chain
first-occurrence errors (investigator: 51.7% vs 37.0%, OR 2.1, p=0.016)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_integrity'
D=pd.read_csv(f'{OUT}/frame.csv.gz')
ch=pd.read_csv(SP+'/deeper/chains.csv.gz',usecols=['pdb','chain','ctype','method','err_first','err_any','title','paper','deposit','res'])
ch=ch[(ch.method=='EM')&~ch.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)]
Hk=ch[ch.ctype=='H'].groupby('pdb').agg(h_err=('err_first','max'))
ent=ch.groupby('pdb').agg(e_any=('err_any','max'),paper=('paper','first'),dep=('deposit','first'),res=('res','first'))
K=D[(D.method=='EM')&(~D.amyloid)&(~D.err_any)&(D.pro8==True)]
kk=K.groupby('pdb').agg(k_trans=('cis8',lambda s:(~s.astype(bool)).all()))
X=kk.join(Hk,how='inner').join(ent)
X['cluster']=X.paper.fillna(pd.Series(X.index,index=X.index))
X=X[(X.h_err)|(~X.e_any)].copy(); X['y']=X.h_err.astype(int); X['x']=X.k_trans.astype(int)
X['yr']=pd.to_datetime(X.dep).dt.year; X['ybin']=np.where(X.yr<=2018,'<=2018',np.where(X.yr>=2024,'>=2024',X.yr.astype(str)))
m=smf.glm('y~x+C(ybin)+res',data=X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(X.cluster)[0]})
print(X.groupby('y').x.agg(['sum','size','mean']))
print('Fisher p',stats.fisher_exact(pd.crosstab(X.y,X.x).values)[1],'adj OR',np.exp(m.params['x']),'p',m.pvalues['x'])
