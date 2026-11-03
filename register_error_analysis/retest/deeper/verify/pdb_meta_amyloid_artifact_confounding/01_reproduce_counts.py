"""Reproduce the headline counts of the 'amyloid_artifact' finding from the shared tables."""
import numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_confounding'
e=pd.read_csv(f'{SP}/entries.csv'); c=pd.read_csv(f'{SP}/chains.csv.gz'); r=pd.read_csv(f'{SP}/runs_first_occurrence.csv')
rx=r'amyloid(?!-beta)'
e['amy']=e.title.fillna('').str.contains(rx,case=False)
c['amy']=c.title.fillna('').str.contains(rx,case=False)
AMY=set(e[e.amy].pdb)
print('amyloid entries',len(AMY),sorted(AMY))
em=e[e.method=='EM']
print('EM error entries',em.err_first.sum(),' amyloid among them',em[em.amy].err_first.sum(),' amyloid EM entries total',em.amy.sum())
print('clean EM entries',(~em.err_any).sum(),' amyloid clean',(em.amy&~em.err_any).sum())
print('all error entries',e.err_first.sum(),' X-ray amyloid',(e.amy&(e.method!='EM')).sum())
# units
cu=c[c.err_first]
print('error units',cu.unit.nunique(),' amyloid error units',cu[cu.amy].unit.nunique())
# lambda error units
print('lambda error units',cu[cu.ctype=='L'].unit.nunique(),' amyloid',cu[(cu.ctype=='L')&cu.amy].unit.nunique())
# papers
print('amyloid papers (paper key)',c[c.amy].paper.nunique(),' DOIs',c[c.amy].doi.nunique(), c[c.amy].groupby('doi',dropna=False).pdb.unique().to_dict())
print('amyloid last authors',c[c.amy].last_author.unique())
# runs
r['amy']=r.pdb.isin(AMY)
print('first-occ runs',len(r),' amyloid',r.amy.sum(), f'{r.amy.mean():.3f}')
print(pd.crosstab(r.origin,r.amy))
print(pd.crosstab(r['mode'],r.amy))
# mode-A units
ru=r.merge(c[['pdb','chain','unit']],on=['pdb','chain'],how='left')
for m in sorted(ru['mode'].dropna().unique()):
    s=ru[ru['mode']==m]; print(m,'units',s.unit.nunique(),'amyloid units',s[s.amy].unit.nunique(),'runs',len(s),'amy runs',s.amy.sum())
# share of 'unexplained' origin categories
for o in ['9 FR3 / other (still unexplained)','8 CDR1/CDR2 boundary, typical length','3 chain end or next to an unbuilt loop']:
    s=ru[ru.origin==o]; print(o,'| runs',len(s),'amy',s.amy.sum(),'| units',s.unit.nunique(),'amy units',s[s.amy].unit.nunique())
ac=c[c.amy]
print(ac.groupby('pdb').agg(n=('chain','size'),nerr=('err_first','sum'),ctype=('ctype','first'),v=('v_gene','first'),L1=('L1','first'),modes=('err_modes','first'),res=('res','first'),dep=('deposit','first'),unit=('unit','nunique')).to_string())
