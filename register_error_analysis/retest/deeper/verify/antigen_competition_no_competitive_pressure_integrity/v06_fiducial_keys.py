# Bug check: fiducial accessions (BRIL, MBP, T4L, GFP, G-protein subunits) inside fused target entities inflate competitor counts
import warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
INV='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
F=pd.read_csv(f'{INV}/tables/focal_entries_antigen_features.csv.gz'); F['sars2']=F.sars2.astype(bool)
EM=F[(F.method=='EM')&F.ab_status2.isin(['bound','fiducial-only'])&F.labs_acc_12m.notna()].copy()
FID={'P0ABE7','P0AEX9','P00720','P42212','P63092','P63096','P62873','P59768','B5Z8H1','P42592','Q13043'}
EM['fidkey']=EM.tgt_keys.fillna('').str.split('|').apply(lambda ks: any(k in FID for k in ks) and len(ks)>1)
print('EM analysis entries whose target keys mix a fiducial accession with a real target:',EM.fidkey.sum(),'errors',EM[EM.fidkey].y.sum())
print(EM[EM.fidkey].groupby('tgt_cls').agg(n=('y','size'),err=('y','sum'),labs12_med=('labs_acc_12m','median')).to_string())
def fit(D,term,extra='C(ycat)+res'):
    mod=smf.glm(f'y ~ {term} + {extra}',D,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(D.paper)[0]})
    ci=mod.conf_int().loc[term]; return 'OR %.3f (%.3f-%.3f) p=%.4f n=%d err=%d'%(np.exp(mod.params[term]),np.exp(ci[0]),np.exp(ci[1]),mod.pvalues[term],len(D),D.y.sum())
print('headline l_labs_acc_12m, excluding these entries:',fit(EM[~EM.fidkey],'l_labs_acc_12m'))
print('non-SARS2 l_labs_acc_12m, excluding these entries:',fit(EM[~EM.fidkey&~EM.sars2],'l_labs_acc_12m'))
