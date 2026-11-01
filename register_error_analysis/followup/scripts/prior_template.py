import gzip, re, json, collections, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
A='/tmp/analysis'; O=f'{A}/out'
# PDB-wide: entry date + resolution
info={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    f=l.split('\t')
    if len(f)<8: continue
    try: dt=pd.to_datetime(f[2],format='%m/%d/%y')
    except Exception: continue
    try: r=float(f[6].split(',')[0])
    except Exception: r=np.nan
    info[f[0].lower()]=(dt,r)
# V-region key = first 105 aa of chain sequence
key2best=collections.defaultdict(list)  # key -> list of (date,res,pdb)
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    hdr=None
    for line in fh:
        if line.startswith('>'):
            hdr=line[1:].split()[0]; mol=line.split()[1]
        elif mol=='mol:protein':
            s=line.strip()
            if len(s)>=100:
                p=hdr.split('_')[0]
                if p in info: key2best[s[:105]].append((info[p][0],info[p][1],p))
print('keys',len(key2best))
d=pd.read_pickle(f'{A}/domains_full2.pkl')
ch=pd.read_pickle(f'{A}/chains.pkl'); seq=dict(zip(zip(ch.pdb,ch.chain),ch.seq))
ab=d[d.cls.isin(['Antibody V','Antibody C'])].copy()
dep=dict(zip(pd.read_pickle(f'{A}/entries35.pkl').pdb,pd.read_pickle(f'{A}/entries35.pkl').deposit))
def prior(row):
    s=seq.get((row.pdb,row.chain),'')
    if len(s)<105: return pd.Series({'prior_any':np.nan,'prior_hires':np.nan})
    L=[x for x in key2best.get(s[:105],[]) if x[2]!=row.pdb and x[0]<dep[row.pdb]]
    return pd.Series({'prior_any':int(len(L)>0),'prior_hires':int(any((x[1]==x[1]) and x[1]<3.0 for x in L))})
ab[['prior_any','prior_hires']]=ab.apply(prior,axis=1)
ab=ab.dropna(subset=['prior_any'])
print(ab.groupby(['cls','role']).prior_hires.mean().unstack().round(3))
v=ab[ab.cls=='Antibody V']
print(v.groupby(['prior_hires']).agg(domains=('flag','size'),entries=('pdb','nunique'),flagged=('flag','sum'),pct=('flag',lambda x:round(100*x.mean(),2))))
print(v.groupby(['role','prior_hires']).agg(domains=('flag','size'),flagged=('flag','sum'),pct=('flag',lambda x:round(100*x.mean(),2))).to_string())
m=smf.logit("flag ~ prior_hires + C(method)*C(rb) + C(yr) + logL",data=v).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(v.pdb)[0]})
print('Antibody V: OR for identical V-region sequence previously deposited at <3 Å: %.2f [%.2f-%.2f] p=%.3g'%(np.exp(m.params.prior_hires),*np.exp(m.conf_int().loc['prior_hires']),m.pvalues.prior_hires))
vv=v[v.role.isin(['Auxiliary','Antibody of interest'])].copy(); vv['aux']=(vv.role=='Auxiliary').astype(int)
m=smf.logit("flag ~ aux + prior_hires + C(method)*C(rb) + C(yr) + logL",data=vv).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(vv.pdb)[0]})
print('Antibody V, aux adjusted for prior template: OR aux %.2f [%.2f-%.2f]; prior %.2f [%.2f-%.2f]'%(np.exp(m.params.aux),*np.exp(m.conf_int().loc['aux']),np.exp(m.params.prior_hires),*np.exp(m.conf_int().loc['prior_hires'])))
ab.to_pickle(f'{A}/ab_prior.pkl')
t=v.groupby(['role','prior_hires']).agg(domains=('flag','size'),flagged=('flag','sum'),pct=('flag',lambda x:round(100*x.mean(),2))).reset_index(); t.to_csv(f'{O}/X5_prior_template.csv',index=False)
