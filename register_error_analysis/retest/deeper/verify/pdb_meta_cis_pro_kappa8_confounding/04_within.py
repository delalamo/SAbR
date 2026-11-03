"""Within-paper and within-entry comparisons: are error kappa chains less often cis-Pro8 than clean kappa chains
of the SAME paper / SAME entry? (removes paper/lab/period/map-level confounding)."""
import numpy as np, pandas as pd, warnings
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy import stats
warnings.filterwarnings('ignore'); pd.set_option('display.width',250)
K=pd.read_csv('kappa_frame.csv.gz'); K=K[~K.amyloid]
K['y']=K.err_first.astype(int); K['x']=K.cis8.astype(int); K['cluster']=K.paper.fillna(K.pdb)
EM=K[(K.method=='EM')&(K.pro8)&((K.err_first)|(~K.err_any))].copy()
out=[]
for grp in ['cluster','pdb']:
    g=EM.groupby(grp).filter(lambda d:d.y.nunique()==2)
    tab=g.groupby([grp,'y']).x.agg(['sum','size']).unstack()
    print(f'\n== within {grp}: {g[grp].nunique()} strata with both error and clean kappa Pro8 chains')
    print(g.groupby([grp,'y']).agg(n=('x','size'),cis=('x','sum'),lab=('last_author','first')).unstack('y').to_string())
    e=g[g.y==1]; c=g[g.y==0]
    # stratified MH
    tbls=[]
    for k,d in g.groupby(grp):
        a=((d.y==1)&(d.x==1)).sum(); b=((d.y==1)&(d.x==0)).sum(); cc=((d.y==0)&(d.x==1)).sum(); dd=((d.y==0)&(d.x==0)).sum()
        tbls.append(np.array([[a,b],[cc,dd]])+0.0)
    st=StratifiedTable(tbls)
    try: mh=st.oddsratio_pooled; p=st.test_null_odds().pvalue
    except Exception: mh=np.nan; p=np.nan
    # conditional logit
    try:
        m=ConditionalLogit(g.y.values,g[['x']].values.astype(float),groups=pd.factorize(g[grp])[0]).fit(disp=0)
        cor=np.exp(m.params[0]); cp=m.pvalues[0]; ci=np.exp(m.conf_int()[0])
    except Exception as ex: cor=cp=np.nan; ci=[np.nan,np.nan]
    # informative strata: those where x varies
    inf=g.groupby(grp).filter(lambda d:d.x.nunique()==2)
    out.append({'test':f'within {grp}','n_strata':g[grp].nunique(),'n_informative_strata':inf[grp].nunique(),'err_cis':f'{e.x.mean()*100:.1f}% ({e.x.sum()}/{len(e)})','ctrl_cis':f'{c.x.mean()*100:.1f}% ({c.x.sum()}/{len(c)})',
                'MH_OR':mh,'MH_p':p,'condlogit_OR':cor,'cl_lo':ci[0],'cl_hi':ci[1],'cl_p':cp})
    # informative stratum list
    print('informative strata (x varies):'); print(inf.groupby([grp,'y']).x.agg(['sum','size']).unstack('y').to_string())
R=pd.DataFrame(out); R.to_csv('within_paper_entry.csv',index=False); print(R.to_string(index=False))
