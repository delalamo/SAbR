"""Confounding re-analysis of the kappa cis-Pro8 finding (pdb_meta). Chain-level logistic models with cluster-robust SE
by paper (and by lab), unit- and paper-level collapses, within-paper and within-entry conditional analyses, strata."""
import sys, numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',90)
def ybin(y): return np.where(y<=2018,'<=2018',np.where(y>=2024,'>=2024',y.astype(str)))
K=pd.read_csv('kappa_frame.csv.gz'); K=K[~K.amyloid]
K['y']=K.err_first.astype(int); K['x']=K.cis8.astype(int); K['cluster']=K.paper.fillna(K.pdb)
K['lab']=K.last_author.fillna(K.cluster)
K['dep_year']=pd.to_datetime(K.deposit).dt.year; K['ybin']=ybin(K.dep_year.values)
K['period']=np.where(K.dep_year<=2019,'<=2019',np.where(K.dep_year<=2021,'2020-21','>=2022'))
K['sars2i']=K.sars2.fillna(False).astype(int); K['mouse']=(K.species=='mouse').astype(int)
K['modeA']=K.modes.fillna('').str.contains('A_FR1'); K['modeDE']=K.modes.fillna('').str.contains('D|E_FR4')
K['irm2']=K.irm.fillna('none recorded'); K['fit2']=K.fit.fillna('not recorded')
K['rb']=pd.cut(K.res,[2.99,3.3,3.6,4.0,5.01]).astype(str)
K['ab_only_trans_all']=((K.v_cis_n==0)).astype(int)
base=K[(K.pro8)&((K.err_first)|(~K.err_any))].copy()
EM=base[base.method=='EM'].copy()
res=[]
def glm(d,form,label,group='cluster',term='x'):
    d=d.dropna(subset=[c for c in ['y','x','res','q_pro8','q_V','chain_q'] if c in form or c in ('y','x')])
    n1=int(d.y.sum()); n0=int((1-d.y).sum())
    e=d[d.y==1].x; c=d[d.y==0].x
    out={'test':label,'n_err':n1,'n_ctrl':n0,'err_cis':f'{e.mean()*100:.1f}% ({int(e.sum())}/{len(e)})','ctrl_cis':f'{c.mean()*100:.1f}% ({int(c.sum())}/{len(c)})',
         'n_err_clusters':d[d.y==1][group].nunique()}
    try:
        m=smf.glm(form,data=d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[group])[0]})
        b=m.params[term]; se=m.bse[term]
        out.update(OR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=m.pvalues[term])
    except Exception as ex: out['note']=str(ex)[:80]
    res.append(out); return out
F0='y ~ x + C(ybin) + res'
glm(EM,F0,'R0 reproduce: EM, ybin+res, cluster paper')
glm(EM,F0,'R1 same, cluster by last author (lab)',group='lab')
glm(EM,'y ~ x + C(ybin) + res + sars2i + mouse','R2 + SARS-CoV-2 + mouse')
glm(EM,'y ~ x + C(ybin) + C(rb) + sars2i + mouse + C(irm2) + C(fit2)','R3 + res bins, sars2, species, start model, fit protocol')
glm(EM,'y ~ x + C(ybin) + res + sars2i + mouse + C(irm2) + C(fit2) + q_pro8','R4 + local Q at Pro8')
glm(EM,'y ~ x + C(ybin) + res + sars2i + mouse + C(irm2) + C(fit2) + q_V + chain_q','R5 + V-domain mean Q + chain Q')
glm(EM,'y ~ x + C(ybin) + res + v_cis_other','R6 + number of other V-domain cis peptides')
glm(EM[~(EM.modeA&(EM.y==1))],F0,'R7 excluding FR1 (mode A) errors')
glm(EM[~((EM.modeA|EM.modeDE)&(EM.y==1))],F0,'R8 errors only in CDR1/CDR2/FR3/other (no FR1, no CDR3/FR4)')
for per in ['<=2019','2020-21','>=2022']:
    glm(EM[EM.period==per],'y ~ x + res','R9 period '+per)
glm(EM[EM.sars2i==0],F0,'R10 non-SARS-CoV-2 only'); glm(EM[EM.sars2i==1],F0,'R10 SARS-CoV-2 only')
glm(EM[EM.res<3.6],F0,'R11 res < 3.6'); glm(EM[EM.res>=3.6],F0,'R11 res >= 3.6')
top2=['10.1038/s41422-021-00487-9','10.7554/eLife.61056']
glm(EM[~EM.cluster.isin(top2)],F0,'R12 drop the two largest error papers (27 of 70 chains)')
glm(EM[EM.novel_v==True],F0,'R13 novel V only')
# X-ray replication
XR=base[base.method=='X-ray']
glm(XR,F0,'R14 X-ray kappa (replication)')
# GEE exchangeable by paper
d=EM.dropna(subset=['res'])
try:
    g=smf.gee('y ~ x + C(ybin) + res',groups='cluster',data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    res.append({'test':'R15 GEE exchangeable by paper','n_err':int(d.y.sum()),'n_ctrl':int((1-d.y).sum()),'OR':np.exp(g.params['x']),'lo':np.exp(g.conf_int().loc['x',0]),'hi':np.exp(g.conf_int().loc['x',1]),'p':g.pvalues['x']})
except Exception as ex: print('gee',ex)
# ---- unit level (one antibody per paper): y = unit has first error; x = any chain of the unit cis8
U=EM.groupby('unit').agg(y=('y','max'),x=('x','max'),xfrac=('x','mean'),res=('res','median'),dep_year=('dep_year','min'),cluster=('cluster','first'),lab=('lab','first'),sars2i=('sars2i','max')).reset_index()
U['ybin']=ybin(U.dep_year.values)
glm(U,F0,'U1 unit level (paper|V region): any chain cis8')
U['x']=(U.xfrac>=0.5).astype(int); glm(U,F0,'U2 unit level: majority chains cis8')
# ---- paper level: y = paper has >=1 kappa error chain; x = paper majority cis8 (all its kappa Pro8 chains)
P=EM.groupby('cluster').agg(y=('y','max'),xfrac=('x','mean'),res=('res','median'),dep_year=('dep_year','min'),n=('y','size'),lab=('lab','first'),sars2i=('sars2i','max')).reset_index()
P['ybin']=ybin(P.dep_year.values); P['x']=(P.xfrac>=0.5).astype(int); P['logn']=np.log2(P.n)
glm(P,'y ~ x + C(ybin) + res + logn','P1 paper level: majority cis8 (+log n chains)',group='lab')
P['x']=(P.xfrac>0).astype(int); glm(P,'y ~ x + C(ybin) + res + logn','P2 paper level: any cis8 chain',group='lab')
# paper level using ONLY clean chains for x (so x is not built from the error chains)
EMc=EM[EM.y==0].groupby('cluster').x.mean().rename('xclean')
P2=P.merge(EMc,on='cluster',how='inner'); P2['x']=(P2.xclean>=0.5).astype(int)
glm(P2,'y ~ x + C(ybin) + res + logn','P3 paper level: x from the paper\'s CLEAN kappa chains only (papers with >=1 clean chain)',group='lab')
R=pd.DataFrame(res)
R.to_csv('models_main.csv',index=False)
print(R.to_string(index=False,float_format=lambda v:f'{v:.3g}'))
