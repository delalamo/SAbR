import numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
def ybin(y):
    return np.where(y<=2018,'<=2018',np.where(y>=2024,'>=2024',y.astype(str)))
def fit_feature(df, feat, kind, covars, cluster='cluster', y='y', scale=None):
    """kind: 'bin' (0/1), 'cont' (scaled per SD unless scale given). Returns dict."""
    d=df[[y,feat,cluster]+[c.replace('C(','').replace(')','') for c in covars]].dropna()
    n1=int(d[y].sum()); n0=int((1-d[y]).sum())
    out={'feature':feat,'kind':kind,'n_err':n1,'n_ctrl':n0}
    if n1<5 or n0<5 or d[feat].nunique()<2: out['note']='too few'; return out
    e=d[d[y]==1][feat]; c=d[d[y]==0][feat]
    if kind=='bin':
        out['err_val']=f'{e.mean()*100:.1f}% ({int(e.sum())}/{len(e)})'; out['ctrl_val']=f'{c.mean()*100:.1f}% ({int(c.sum())}/{len(c)})'
        tab=[[e.sum(),len(e)-e.sum()],[c.sum(),len(c)-c.sum()]]
        out['p_unadj']=stats.fisher_exact(tab)[1]; x=feat; unit='yes vs no'
        if min(e.sum(),c.sum(),len(e)-e.sum(),len(c)-c.sum())==0 or e.sum()+c.sum()<8:
            a,b_,cc,dd_=[v+0.5 for v in (e.sum(),len(e)-e.sum(),c.sum(),len(c)-c.sum())]
            orr=(a*dd_)/(b_*cc); se=np.sqrt(1/a+1/b_+1/cc+1/dd_)
            out.update(OR=orr,lo=orr*np.exp(-1.96*se),hi=orr*np.exp(1.96*se),p=out['p_unadj'],unit='yes vs no',note='rare: unadjusted Fisher, Haldane OR')
            return out
    else:
        out['err_val']=f'median {e.median():.4g} (IQR {e.quantile(.25):.3g}-{e.quantile(.75):.3g})'; out['ctrl_val']=f'median {c.median():.4g} (IQR {c.quantile(.25):.3g}-{c.quantile(.75):.3g})'
        out['p_unadj']=stats.mannwhitneyu(e,c).pvalue
        s=scale or d[feat].std(); d=d.assign(_x=d[feat]/s); x='_x'; unit=f'per {s:.3g}'
    out['unit']=unit
    f=f'{y} ~ {x}'+(' + '+' + '.join(covars) if covars else '')
    try:
        m=smf.glm(f,data=d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[cluster])[0]})
        b=m.params[x]; se=m.bse[x]
        out.update(OR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=m.pvalues[x])
    except Exception as ex:
        out['note']=f'fit failed: {ex}'[:80]
    return out
def bh(df,col='p'):
    p=df[col].values; ok=~np.isnan(p); q=np.full(len(p),np.nan)
    if ok.sum(): q[ok]=multipletests(p[ok],method='fdr_bh')[1]
    df['q']=q; return df
