"""Further confounding checks: unit-level non-FR1, top-paper influence, software/germline adjustment, copied-error
chains, mode breakdown, entry-level heavy-chain marker, and risk-scale numbers by period at unit level."""
import numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats
warnings.filterwarnings('ignore'); pd.set_option('display.width',250)
def ybin(y): return np.where(y<=2018,'<=2018',np.where(y>=2024,'>=2024',y.astype(str)))
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
K=pd.read_csv('kappa_frame.csv.gz'); K=K[~K.amyloid]
K['y']=K.err_first.astype(int); K['x']=K.cis8.astype(int); K['cluster']=K.paper.fillna(K.pdb)
K['dep_year']=pd.to_datetime(K.deposit).dt.year; K['ybin']=ybin(K.dep_year.values)
K['modes']=K.modes.fillna('')
K['modeA']=K.modes.str.contains('A_FR1'); K['modeDE']=K.modes.str.contains('D|E_FR4')
K['vfam2']=K.v_gene.fillna('NA').str.extract(r'^(IGKV\d+)')[0].fillna('other')
for s in ['sw_ISOLDE','sw_Rosetta','sw_Coot','sw_Phenix']: K[s]=K[s].fillna(False).astype(int)
K['irm2']=K.irm.fillna('none recorded'); K['fit2']=K.fit.fillna('not recorded')
EM=K[(K.method=='EM')&(K.pro8)].copy()
A=EM[(EM.err_first)|(~EM.err_any)].copy()
res=[]
def glm(d,form,label,group='cluster'):
    cols=[c for c in ['y','x','res','q_pro8','chain_q'] if c in d.columns and (c in ('y','x','res') or c in form)]
    d=d.dropna(subset=cols)
    e=d[d.y==1].x; c=d[d.y==0].x
    out={'test':label,'n_err':len(e),'n_ctrl':len(c),'n_err_papers':d[d.y==1][group].nunique(),'err_cis':f'{e.mean()*100:.1f}% ({int(e.sum())}/{len(e)})','ctrl_cis':f'{c.mean()*100:.1f}% ({int(c.sum())}/{len(c)})'}
    try:
        m=smf.glm(form,data=d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[group])[0]})
        b=m.params['x']; se=m.bse['x']; out.update(OR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=m.pvalues['x'])
    except Exception as ex: out['note']=str(ex)[:60]
    res.append(out); return out
F0='y ~ x + C(ybin) + res'
def unit(d):
    U=d.groupby('unit').agg(y=('y','max'),x=('x','max'),res=('res','median'),dep_year=('dep_year','min'),cluster=('cluster','first'),
        modeA=('modeA','max'),modeDE=('modeDE','max')).reset_index(); U['ybin']=ybin(U.dep_year.values); return U
U=unit(A)
top2=['10.1038/s41422-021-00487-9','10.7554/eLife.61056']
glm(U[~(U.modeA&(U.y==1))],F0,'M1 unit level, error units without FR1 error')
glm(U[~((U.modeA)&(U.y==1))&~U.cluster.isin(top2)],F0,'M2 unit level, no FR1 error, drop 2 largest error papers')
glm(U[~((U.modeA|U.modeDE)&(U.y==1))],F0,'M3 unit level, errors only outside FR1 and CDR3/FR4')
glm(U[~U.cluster.isin(top2)],F0,'M4 unit level, drop 2 largest error papers')
glm(A,'y ~ x + C(ybin) + res + sw_ISOLDE + sw_Rosetta + sw_Coot + sw_Phenix ','M5 chain, + software flags')
glm(A,'y ~ x + C(ybin) + res + C(vfam2)','M6 chain, + IGKV family')
glm(A,'y ~ x + C(ybin) + res + C(irm2) + C(fit2) + sw_ISOLDE + sw_Rosetta + sw_Coot + sw_Phenix + C(vfam2) + q_pro8 + chain_q','M7 chain, kitchen sink (year,res,start model,fit,software,IGKV fam,local Q,chain Q)')
# mode breakdown at chain level
er=A[A.y==1]
mb=[]
for m in ['A_FR1Nterm','F_CDR1','B_CDR2FR3','C_FR3','Dplus_CDR3short','Dminus_CDR3long','D0','E_FR4bulge','other']:
    s=er[er.modes.str.contains(m)]
    mb.append({'mode':m,'n_chains':len(s),'n_units':s.unit.nunique(),'n_papers':s.cluster.nunique(),'cis8':int(s.x.sum())})
MB=pd.DataFrame(mb); print(MB.to_string(index=False)); MB.to_csv('mode_breakdown.csv',index=False)
# copied / artifact-only chains (excluded from both arms): cis8 rate
cp=EM[EM.err_copied_or_artifact_only==True]
print('copied/artifact-only EM kappa Pro8 chains:',len(cp),'cis8',cp.x.mean().round(3),'papers',cp.cluster.nunique())
# risk scale at unit level by period
U['period']=np.where(U.dep_year<=2019,'<=2019',np.where(U.dep_year<=2021,'2020-21','>=2022'))
print(U.groupby(['period','x']).y.agg(['sum','size','mean']).to_string())
# entry-level marker: among EM entries with >=1 CLEAN kappa Pro8 chain, does lack of cis8 in those clean chains
# predict a first-occurrence HEAVY-chain error in the same entry?
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','ctype','method','err_first','title','paper','deposit','res'])
ch=ch[~ch.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)]
herr=ch[(ch.ctype=='H')&(ch.err_first)].pdb.unique()
cl=EM[(EM.y==0)&(~EM.err_any)]
E=cl.groupby('pdb').agg(xk=('x','mean'),res=('res','first'),dep_year=('dep_year','first'),cluster=('cluster','first')).reset_index()
E['x']=(E.xk==0).astype(int); E['y']=E.pdb.isin(herr).astype(int); E['ybin']=ybin(E.dep_year.values)
d=E.dropna(subset=['res'])
m=smf.glm('y ~ x + C(ybin) + res',data=d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.cluster)[0]})
print('entry marker (all EM entries w/ clean kappa Pro8): H-error rate if clean kappa all-trans',d[d.x==1].y.mean().round(4),f'({d[d.x==1].y.sum()}/{(d.x==1).sum()})',
      'vs cis',d[d.x==0].y.mean().round(4),f'({d[d.x==0].y.sum()}/{(d.x==0).sum()})','OR',np.exp(m.params['x']).round(2),'p',m.pvalues['x'])
res.append({'test':'M8 entry level: clean kappa chains all-trans -> heavy-chain first error in same entry','n_err':int(d.y.sum()),'n_ctrl':int((1-d.y).sum()),
            'err_cis':f'{d[d.y==1].x.mean()*100:.1f}% all-trans','ctrl_cis':f'{d[d.y==0].x.mean()*100:.1f}% all-trans','OR':np.exp(m.params['x']),
            'lo':np.exp(m.conf_int().loc['x',0]),'hi':np.exp(m.conf_int().loc['x',1]),'p':m.pvalues['x']})
# unit-level within paper conditional logit
g=U.groupby('cluster').filter(lambda d:d.y.nunique()==2)
inf=g.groupby('cluster').filter(lambda d:d.x.nunique()==2)
cm=ConditionalLogit(g.y.values,g[['x']].values.astype(float),groups=pd.factorize(g.cluster)[0]).fit(disp=0)
print('unit-level within-paper: strata',g.cluster.nunique(),'informative',inf.cluster.nunique(),'OR',np.exp(cm.params[0]),'CI',np.exp(cm.conf_int()[0]),'p',cm.pvalues[0])
print(inf.groupby(['cluster','y']).x.agg(['sum','size']).unstack('y').to_string())
res.append({'test':'M9 unit level, within paper (conditional logit)','n_err':int(g.y.sum()),'n_ctrl':int((1-g.y).sum()),'n_err_papers':g.cluster.nunique(),
            'err_cis':f"{g[g.y==1].x.mean()*100:.1f}% ({g[g.y==1].x.sum()}/{(g.y==1).sum()})",'ctrl_cis':f"{g[g.y==0].x.mean()*100:.1f}% ({g[g.y==0].x.sum()}/{(g.y==0).sum()})",
            'OR':np.exp(cm.params[0]),'lo':np.exp(cm.conf_int()[0][0]),'hi':np.exp(cm.conf_int()[0][1]),'p':cm.pvalues[0]})
R=pd.DataFrame(res); R.to_csv('models_more.csv',index=False); print(R.to_string(index=False,float_format=lambda v:f'{v:.3g}'))
