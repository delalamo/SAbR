import numpy as np, pandas as pd
pd.set_option('display.width',250)
K=pd.read_csv('kappa_frame.csv.gz')
K=K[~K.amyloid]
E=K[(K.method=='EM')&(K.pro8)&((K.err_first)|(~K.err_any))].copy()
print('EM kappa pro8 rows',len(E),'err',E.err_first.sum(),'cis8 err',E[E.err_first].cis8.sum(),'ctrl cis',E[~E.err_first].cis8.mean())
er=E[E.err_first]
print('error chains per paper'); print(er.groupby('paper').agg(n=('pdb','size'),ent=('pdb','nunique'),cis=('cis8','sum'),lab=('last_author','first'),yr=('year','first'),modes=('modes',lambda s:';'.join(sorted(set(';'.join(s.fillna('')).split(';')))))).sort_values('n',ascending=False).to_string())
print('n papers',er.paper.nunique(),'labs',er.last_author.nunique(),'units',er.unit.nunique(),'entries',er.pdb.nunique())
# control cis8 rate by year, and by res bin
E['rb']=pd.cut(E.res,[2.99,3.3,3.6,4.0,5.01])
print(E.groupby(['year']).apply(lambda g:pd.Series({'n_ctrl':(~g.err_first).sum(),'ctrl_cis':g[~g.err_first].cis8.mean(),'n_err':g.err_first.sum(),'err_cis':g[g.err_first].cis8.mean()})).to_string())
print(E.groupby(['rb']).apply(lambda g:pd.Series({'n_ctrl':(~g.err_first).sum(),'ctrl_cis':g[~g.err_first].cis8.mean(),'n_err':g.err_first.sum(),'err_cis':g[g.err_first].cis8.mean()})).to_string())
# error papers: cis rate of their clean kappa chains
ep=set(er.paper)
cp=E[(~E.err_first)&(E.paper.isin(ep))]
print('clean kappa chains in error papers',len(cp),'cis8',cp.cis8.mean(), 'papers',cp.paper.nunique())
cp2=E[(~E.err_first)&(~E.paper.isin(ep))]
print('clean kappa chains in non-error papers',len(cp2),'cis8',cp2.cis8.mean())
# chains in error papers: error vs clean
print(E[E.paper.isin(ep)].groupby('err_first').cis8.agg(['mean','size']))
# other v cis & const cis among error chains
print(er[['cis8','v_cis_n','const_cis_n','n_const_modeled','imgt7_built','first_imgt','q_pro8','irm','fit']].to_string())
