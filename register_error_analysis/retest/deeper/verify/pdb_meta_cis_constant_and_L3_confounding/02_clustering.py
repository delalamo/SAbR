"""Confounding by paper/lab clustering: leave-one-paper-out, paper-level (one vote per paper), within-paper and
within-entry conditional logistic regression, paper-weighted GLM; plus +SARS-CoV-2, +chain Q, +starting model."""
import sys, warnings, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
warnings.filterwarnings('ignore')
T='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
D=pd.read_csv('frame.csv.gz')
D=D.merge(pd.read_csv(f'{T}/chain_features_rcsb.csv.gz',usecols=['pdb','chain','chain_q']),on=['pdb','chain'],how='left')
D['sars2']=D.sars2.astype(float)
ZHOU='10.1038/s41422-021-00487-9'
sets={}
for ct in ['K','H']:
    s=D[(D.ctype==ct)&D.const_ok].copy(); s['x']=s.const_cis_any; sets[f'{ct}_const']=(s,['C(ybin)','res'])
s=D[D.const_ok&D.ctype.isin(['K','H','L'])].copy(); s['x']=s.const_cis_any; sets['all_const']=(s,['C(ybin)','res','C(ctype)'])
K=D[(D.ctype=='K')&D.pro115]
s=K[~(K.mode_D_inv&(K.y==1))].copy(); s['x']=s.cis115.astype(float); sets['K_Pro115_noCDRmode(inv)']=(s,['C(ybin)','res'])
s=K[~(K.mode_D_true&(K.y==1))].copy(); s['x']=s.cis115.astype(float); sets['K_Pro115_noCDR3FR4mode']=(s,['C(ybin)','res'])
rows=[]
def rec(name,analysis,**kw): r={'set':name,'analysis':analysis}; r.update(kw); rows.append(r); print(name,analysis,{k:(round(v,4) if isinstance(v,float) else v) for k,v in kw.items()})
for name,(s,cov) in sets.items():
    a=fit_feature(s,'x','bin',cov); rec(name,'investigator model (chain, year bin + res, cluster=paper)',n_err=a['n_err'],n_ctrl=a['n_ctrl'],err=a.get('err_val'),ctrl=a.get('ctrl_val'),OR=a.get('OR'),p=a.get('p'),note=a.get('note'))
    e=s[s.y==1]; rec(name,'error clusters',n_err_papers=e.cluster.nunique(),n_err_entries=e.pdb.nunique(),top_paper_share=e.cluster.value_counts().iloc[0]/len(e))
    # drop Zhou paper entirely (error and its clean chains)
    s2=s[s.cluster!=ZHOU]; a=fit_feature(s2,'x','bin',cov)
    rec(name,'drop Zhou Q. 2021 Cell Res paper',n_err=a['n_err'],n_ctrl=a['n_ctrl'],err=a.get('err_val'),ctrl=a.get('ctrl_val'),OR=a.get('OR'),p=a.get('p'),p_fisher=a.get('p_unadj'),note=a.get('note'))
    # leave-one-error-paper-out: max p
    ps=[];ors=[]
    for c in e.cluster.unique():
        a=fit_feature(s[s.cluster!=c],'x','bin',cov); ps.append(a.get('p',np.nan)); ors.append(a.get('OR',np.nan))
    rec(name,'leave-one-error-paper-out (worst case)',OR_max=np.nanmax(ors),p_max=np.nanmax(ps))
    # paper-level: one vote per paper -- paper mean cis among its error chains vs among its clean chains
    pe=e.groupby('cluster').x.mean(); pc=s[s.y==0].groupby('cluster').x.mean()
    rec(name,'paper-level means (one vote per paper), Mann-Whitney',n_err_papers=len(pe),n_ctrl_papers=len(pc),err=f'{pe.mean():.3f}',ctrl=f'{pc.mean():.3f}',
        err_any_cis=f'{(pe>0).sum()}/{len(pe)}',ctrl_any_cis=f'{(pc>0).sum()}/{len(pc)}',p=stats.mannwhitneyu(pe,pc).pvalue,
        p_fisher_anycis=stats.fisher_exact([[(pe>0).sum(),(pe==0).sum()],[(pc>0).sum(),(pc==0).sum()]])[1])
    # paper-weighted GLM
    x=s.dropna(subset=['x','res']).copy(); x['w']=1/x.groupby(['cluster','y']).pdb.transform('size')
    try:
        m=smf.glm('y ~ x + '+' + '.join(cov),data=x,family=sm.families.Binomial(),var_weights=x.w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.cluster)[0]})
        rec(name,'paper-weighted GLM (1 vote per paper x status)',OR=float(np.exp(m.params['x'])),p=float(m.pvalues['x']))
    except Exception as ex: rec(name,'paper-weighted GLM',note=str(ex)[:80])
    # extra covariates
    for extra in (['sars2'],['chain_q'],['C(irm)'],['sars2','chain_q','C(irm)']):
        a=fit_feature(s,'x','bin',cov+extra); rec(name,'+'+'+'.join(extra),n_err=a['n_err'],OR=a.get('OR'),p=a.get('p'),note=a.get('note'))
    # within-paper and within-entry conditional logistic (only strata with both error and clean chains)
    for g in ['cluster','pdb']:
        x=s.dropna(subset=['x']); st=x.groupby(g).y.transform(lambda v:v.nunique()==2); x=x[st]
        inf=x.groupby(g).filter(lambda t:t.x.nunique()==2)
        ne=x.y.sum(); nc=(x.y==0).sum()
        er=x[x.y==1].x.mean() if ne else np.nan; cr=x[x.y==0].x.mean() if nc else np.nan
        try:
            m=ConditionalLogit(x.y.values,x[['x']].values,groups=pd.factorize(x[g])[0]).fit(disp=0)
            orr,p=float(np.exp(m.params[0])),float(m.pvalues[0])
        except Exception as ex: orr,p=np.nan,np.nan
        rec(name,f'within-{"paper" if g=="cluster" else "entry"} conditional logit',n_strata=x[g].nunique(),n_informative_strata=inf[g].nunique(),
            n_err=int(ne),n_ctrl=int(nc),err=f'{er:.3f}',ctrl=f'{cr:.3f}',OR=orr,p=p)
pd.DataFrame(rows).to_csv('02_clustering_results.csv',index=False)
