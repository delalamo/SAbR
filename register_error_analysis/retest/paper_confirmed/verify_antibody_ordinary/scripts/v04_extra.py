"""Extra checks: year-adjusted has_hq in EM (excluding units with an X-ray chain at exactly <=3.0 A), prior same antibody
crude, method-standardised has_hq control rate, conkit-missed rabbit/bovine count."""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DEEP='/home/user/SAbR/register_error_analysis/retest/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_antibody_ordinary/tables'
CH=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
V=pd.read_csv(f'{OUT}/v01_paper_errors_rederived.csv')
st=set(map(tuple,V[V.status=='confirmed first occurrence'][['pdb','chain']].values))
CH['strict']=[k in st for k in zip(CH.pdb,CH.chain)]
s1=set(V.pdb)
CH['ctrl']=(CH.release.astype(str)<='2022-04-05')&~CH.pdb.isin(s1)&~CH.err_any.astype(bool)
g=CH.groupby('unit')
U=pd.DataFrame({'paper':g.paper.first(),'strict':g.strict.any(),'ctrl':g.ctrl.all(),
  'minres_x':g.apply(lambda d:d.loc[d.method!='EM','res'].min()),'anyEM':g.method.agg(lambda s:(s=='EM').any())})
U['EM']=CH[CH.strict].groupby('unit').method.agg(lambda s:(s=='EM').any()).reindex(U.index).fillna(U.anyEM).astype(bool)
SN=pd.read_csv(f'{DEEP}/seq_novelty/tables/units.csv.gz',low_memory=False).set_index('unit')
LD=pd.read_csv(f'{DEEP}/literature_dean/tables/units_literature.csv.gz',low_memory=False).set_index('unit')
U=U.join(SN[['dep_year','n_prior_same_ab','res_med']]).join(LD[['has_hq']])
d=U[(U.strict|U.ctrl)].copy(); d['y']=d.strict.astype(int); d['hq']=d.has_hq.astype(int)
d['yb']=pd.cut(d.dep_year,[0,2017,2019,2020,2100],labels=['a','b','c','d']).astype(str)
for nm,dd in [('EM',d[d.EM]),('EM excl <=3.0 X-ray chain',d[d.EM&~(d.minres_x<=3.0)]),('all excl <=3.0',d[~(d.minres_x<=3.0)])]:
    f='y ~ hq + C(yb) + res_med' + ('' if nm.startswith('EM') else ' + EM')
    m=smf.gee(f,groups=pd.factorize(dd.paper)[0],data=dd,family=sm.families.Binomial()).fit()
    print(f'has_hq GEE {nm}: OR {np.exp(m.params.hq):.2f} ({np.exp(m.conf_int().loc["hq"][0]):.2f}-{np.exp(m.conf_int().loc["hq"][1]):.2f}) p={m.pvalues.hq:.3g}')
ca=d[d.strict]; co=d[d.ctrl]
std=sum(co[co.EM==e].hq.mean()*(ca.EM==e).sum() for e in [True,False])/len(ca)
print(f'method-standardised control has_hq rate {std:.3f} (crude {co.hq.mean():.3f}); cases {ca.hq.mean():.3f}')
a=int((ca.n_prior_same_ab>0).sum()); c=int((co.n_prior_same_ab>0).sum())
print('prior same antibody crude',a,len(ca),c,len(co),stats.fisher_exact([[a,len(ca)-a],[c,len(co)-c]]))
