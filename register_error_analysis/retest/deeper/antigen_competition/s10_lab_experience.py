# Is "novel antigen / capsid / enzyme" a proxy for labs new to antibody structures? Lab experience from the PDB-wide antibody set
import re, unicodedata, collections, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
def norm_name(n):
    if not isinstance(n,str) or not n: return None
    n=unicodedata.normalize('NFKD',n).encode('ascii','ignore').decode().lower()
    sur,_,ini=n.partition(','); return re.sub(r'[^a-z]','',sur)+'_'+re.sub(r'[^a-z]','',ini)[:1]
A=pd.read_csv(f'{OUT}/tables/pdbwide_ab_entries_antigen.csv.gz',usecols=['pdb','lab','date'])
A['date']=pd.to_datetime(A.date.str[:10])
# all PDB entries: last author + deposition date
rows=[]
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    q=l.split('\t')
    if len(q)<6: continue
    au=[a.strip() for a in re.split(r'(?<=\.),\s*',q[5]) if a.strip()]
    if not au: continue
    try: d=pd.to_datetime(q[2],format='%m/%d/%y')
    except Exception: continue
    if d.year>2026: d=d.replace(year=d.year-100)
    rows.append((q[0].lower(),norm_name(au[-1]),d,'ELECTRON' in (q[7] if len(q)>7 else '')))
I=pd.DataFrame(rows,columns=['pdb','lab','date','em'])
labA=A.groupby('lab').date.apply(lambda s:np.sort(s.values)).to_dict()
labI=I.groupby('lab').date.apply(lambda s:np.sort(s.values)).to_dict()
labIem=I[I.em].groupby('lab').date.apply(lambda s:np.sort(s.values)).to_dict()
m=pd.read_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz')
m['dep']=pd.to_datetime(m.deposit)
def prior(dct,lab,d):
    a=dct.get(lab)
    if a is None: return 0
    return int(np.searchsorted(a,np.datetime64(d-pd.Timedelta(days=30))))
m['lab_prior_ab']=[prior(labA,l,d) for l,d in zip(m.lab,m.dep)]
m['lab_prior_any']=[prior(labI,l,d) for l,d in zip(m.lab,m.dep)]
m['lab_prior_em']=[prior(labIem,l,d) for l,d in zip(m.lab,m.dep)]
for c in ['lab_prior_ab','lab_prior_any','lab_prior_em']: m['l_'+c]=np.log2(1+m[c])
m['new_ab_lab']=(m.lab_prior_ab==0).astype(int)
m['cap']=(m.tgt_cls=='viral capsid').astype(int); m['enz']=m.tgt_cls.isin(['enzyme','viral other']).astype(int)
m['cls2']=m.tgt_cls.replace({'viral other':'enzyme'})
EM=m[m.method=='EM'].copy()
def fit(d,f,terms):
    mod=smf.glm(f,d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]})
    return {t:(np.exp(mod.params[t]),*np.exp(mod.conf_int().loc[t]).values,mod.pvalues[t]) for t in terms}
T=[]
print('EM: lab experience, errors vs clean (medians):')
print(EM.groupby('y')[['lab_prior_ab','lab_prior_any','lab_prior_em','new_ab_lab']].median().to_string())
print(EM.groupby('y')[['new_ab_lab']].mean().round(3).to_string())
print('\nEM error rate by lab antibody experience (prior antibody entries of the last author):')
EM['expbin']=pd.cut(EM.lab_prior_ab,[-1,0,4,19,1e5],labels=['0','1-4','5-19','>=20'])
print(EM.groupby(['period','expbin']).y.agg(['mean','size','sum']).round(3).to_string())
for t in ['l_lab_prior_ab','new_ab_lab','l_lab_prior_any','l_lab_prior_em']:
    for nm,d in [('EM',EM),('EM non-SARS2',EM[~EM.sars2]),('X-ray',m[m.method=='X-ray'])]:
        f=f'y ~ {t} + C(ycat) + res' if nm!='X-ray' else f'y ~ {t} + C(period) + res'
        r=fit(d,f,[t])[t]; T.append(dict(test=f'lab experience: {t}',stratum=nm,OR=r[0],lo=r[1],hi=r[2],p=r[3],n=len(d),n_err=int(d.y.sum())))
        print(f'  {t:18s} {nm:14s} OR {r[0]:.2f} ({r[1]:.2f}-{r[2]:.2f}) p={r[3]:.2g}')
print('\nDo capsid/enzyme/novel-family effects survive adjustment for lab antibody experience? (EM)')
D=EM[EM.labs_acc_12m.notna()&EM.ab_status2.isin(['bound','fiducial-only'])]
for f,lab in [('y ~ cap + enz + first_fam + C(ycat) + res','without experience'),('y ~ cap + enz + first_fam + l_lab_prior_ab + C(ycat) + res','with log2(1+prior antibody entries)'),
              ('y ~ cap + enz + first_fam + l_lab_prior_ab + l_lab_prior_em + C(ycat) + res','with antibody + EM experience')]:
    terms=['cap','enz','first_fam']+(['l_lab_prior_ab'] if 'l_lab_prior_ab' in f else [])+(['l_lab_prior_em'] if 'l_lab_prior_em' in f else [])
    r=fit(D,f,terms)
    for t in terms:
        print(f'  {lab:40s} {t:16s} OR {r[t][0]:.2f} ({r[t][1]:.2f}-{r[t][2]:.2f}) p={r[t][3]:.2g}')
        T.append(dict(test=f'joint model ({lab}): {t}',stratum='EM',OR=r[t][0],lo=r[t][1],hi=r[t][2],p=r[t][3],n=len(D),n_err=int(D.y.sum())))
print('\nLab antibody experience by target class (EM, median prior antibody entries; share of new antibody labs):')
print(EM.groupby('cls2').agg(n=('y','size'),med_prior_ab=('lab_prior_ab','median'),new_lab=('new_ab_lab','mean'),err=('y','mean')).round(3).sort_values('n',ascending=False).to_string())
pd.DataFrame(T).to_csv(f'{OUT}/tables/lab_experience_tests.csv',index=False)
m[['pdb','lab_prior_ab','lab_prior_any','lab_prior_em','new_ab_lab']].to_csv(f'{OUT}/tables/lab_experience.csv',index=False)
