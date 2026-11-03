# Error modes by target class (EM first-occurrence error entries; an entry can carry several modes)
import json, collections, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
m=pd.read_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz')
m['cap']=(m.tgt_cls=='viral capsid').astype(int); m['enz']=m.tgt_cls.isin(['enzyme','viral other']).astype(int)
EM=m[m.method=='EM']
print('\nError modes (first-occurrence entries, EM) by target class:')
x=EM[EM.y==1].copy(); x['mode']=x.modes.fillna('').str.split(';'); x=x.explode('mode').reset_index(drop=True)
x['grp']=np.where(x.cap==1,'capsid',np.where(x.enz==1,'enzyme',np.where(x.sars2,'SARS-CoV-2','other')))
print(pd.crosstab(x['mode'],x.grp).to_string())
