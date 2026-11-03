"""Pro115 conditional on Pro8 status (is Pro115 independent of the 'all-trans V' chains?); paper-level tests without the
Zhou paper; entry-wide zero-cis restriction; BH across all verifier tests."""
import sys, warnings, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
warnings.filterwarnings('ignore')
ZHOU='10.1038/s41422-021-00487-9'
T='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
X=pd.read_csv('03_canonical_sites.csv.gz')
E=pd.read_csv(f'{T}/entry_features_rcsb.csv.gz',usecols=['pdb','cis_peptide_count'])
X=X.merge(E,on='pdb',how='left')
rows=[]
def add(label,s,feat,cov=['C(ybin)','res']):
    a=fit_feature(s.assign(x=s[feat].astype(float)),'x','bin',cov)
    rows.append({'test':label,'n_err':a['n_err'],'n_err_papers':s[s.y==1].cluster.nunique(),'n_ctrl':a['n_ctrl'],'err':a.get('err_val'),'ctrl':a.get('ctrl_val'),
                 'OR':a.get('OR'),'p':a.get('p'),'note':a.get('note')})
def paper_level(label,s,feat):
    e=s[s.y==1].groupby('cluster')[feat].mean(); c=s[s.y==0].groupby('cluster')[feat].mean()
    t=[[(e>0).sum(),(e==0).sum()],[(c>0).sum(),(c==0).sum()]]
    rows.append({'test':label+' [paper level: papers with any cis among their error / clean chains, Fisher]','n_err':len(e),'n_err_papers':len(e),'n_ctrl':len(c),
                 'err':f'{(e>0).sum()}/{len(e)}','ctrl':f'{(c>0).sum()}/{len(c)}','OR':stats.fisher_exact(t)[0],'p':stats.fisher_exact(t)[1]})
CK=X[(X.ctype=='K')&(X.ck_YP_modelled==1)]
H=X[(X.ctype=='H')&(X.const_ok)]
ALLC=X[X.const_ok]
P=X[(X.ctype=='K')&(X.k_P115_present==1)&~(X.mode_D_true&(X.y==1))]
nz=lambda s:s[s.cluster!=ZHOU]
# reproduced originals
add('C-kappa any cis (investigator set) [reproduce]',X[(X.ctype=='K')&X.const_ok],'const_cis_any')
add('CH1 any cis [reproduce]',H,'const_cis_any')
add('all constant any cis [reproduce]',ALLC,'const_cis_any',['C(ybin)','res','C(ctype)'])
add('kappa Pro115 cis, no CDR-mode error (investigator regex) [reproduce]',X[(X.ctype=='K')&(X.k_P115_present==1)&~(X.mode_D_inv&(X.y==1))],'k_P115_cis')
# drop dominant paper
add('C-kappa any cis | drop Zhou',nz(X[(X.ctype=='K')&X.const_ok]),'const_cis_any')
add('CH1 any cis | drop Zhou',nz(H),'const_cis_any')
add('all constant any cis | drop Zhou',nz(ALLC),'const_cis_any',['C(ybin)','res','C(ctype)'])
add('kappa Pro115 cis (no CDR3/FR4 error) | drop Zhou',nz(P),'k_P115_cis')
paper_level('C-kappa any cis | drop Zhou',nz(X[(X.ctype=='K')&X.const_ok]),'const_cis_any')
paper_level('CH1 any cis | drop Zhou',nz(H),'const_cis_any')
paper_level('all constant any cis | drop Zhou',nz(ALLC),'const_cis_any')
paper_level('kappa Pro115 cis | drop Zhou',nz(P),'k_P115_cis')
# entries with at least one cis peptide anywhere in the entry (removes entry-wide all-trans models)
add('C-kappa any cis | entries with >=1 cis anywhere',X[(X.ctype=='K')&X.const_ok&(X.cis_peptide_count>0)],'const_cis_any')
add('all constant any cis | entries with >=1 cis anywhere',ALLC[ALLC.cis_peptide_count>0],'const_cis_any',['C(ybin)','res','C(ctype)'])
add('kappa Pro115 cis | entries with >=1 cis anywhere',P[P.cis_peptide_count>0],'k_P115_cis')
# Pro115 conditional on Pro8 status (Pro8 present and site in a chain without mode-A error, so Pro8 status is not definitional)
P8=P[P.pro8==True].copy()
add('kappa Pro115 cis | Pro8 modelled TRANS (stratum)',P8[P8.cis8==False],'k_P115_cis')
add('kappa Pro115 cis | Pro8 modelled CIS (stratum)',P8[P8.cis8==True],'k_P115_cis')
add('kappa Pro115 cis, adjusted for Pro8 cis status',P8.assign(c8=P8.cis8.astype(int)),'k_P115_cis',['C(ybin)','res','c8'])
# entry level: whole entry modelled without any cis peptide (error entries vs clean entries, EM, amyloid excluded)
F=pd.read_csv(f'{T}/em_entry_analysis_frame.csv.gz')
F=F.merge(E,on='pdb',how='left',suffixes=('','_e')); F['zero_cis']=(F.cis_peptide_count==0).astype(float)
for lab,ff in [('entry: no cis peptide anywhere in the entry (EM entries)',F),('entry: no cis peptide anywhere | drop Zhou entries',F[F.cluster!=ZHOU])]:
    a=fit_feature(ff,'zero_cis','bin',['C(ybin)','res','l2nV'])
    rows.append({'test':lab,'n_err':a['n_err'],'n_err_papers':ff[ff.y==1].cluster.nunique(),'n_ctrl':a['n_ctrl'],'err':a.get('err_val'),'ctrl':a.get('ctrl_val'),'OR':a.get('OR'),'p':a.get('p'),'note':a.get('note')})
R=pd.DataFrame(rows)
for f,lab in [('02_clustering_results.csv',None),('04_period_novel_localres_tests.csv',None),('03_canonical_sites_tests.csv',None)]:
    pass
# BH across all verifier tests with p-values (this table + 02/03/04/05)
allp=[R[['test','p']].assign(src='06')]
a=pd.read_csv('02_clustering_results.csv'); a=a[a.p.notna()]; allp.append(pd.DataFrame({'test':a.set+': '+a.analysis,'p':a.p,'src':'02'}))
for f in ['03_canonical_sites_tests.csv','04_period_novel_localres_tests.csv']:
    a=pd.read_csv(f); a=a[a.p.notna()]; allp.append(pd.DataFrame({'test':a.test,'p':a.p,'src':f[:2]}))
a=pd.read_csv('05_L3_local_disagreement.csv'); allp.append(pd.DataFrame({'test':'Pro115 cis, chains with full SAbR/ANARCI agreement over IMGT 105-118 | '+a.set,'p':a.p,'src':'05'}))
B=pd.concat(allp,ignore_index=True).dropna(subset=['p']).drop_duplicates('test'); B['q_BH']=multipletests(B.p,method='fdr_bh')[1]
B.to_csv('all_tests_bh.csv',index=False)
R=R.merge(B[['test','q_BH']],on='test',how='left'); R.to_csv('06_summary_tests.csv',index=False)
pd.set_option('display.width',260); pd.set_option('display.max_colwidth',100)
print(R.to_string(index=False,float_format=lambda v:f'{v:.3g}'))
print('n tests in BH:',len(B))
