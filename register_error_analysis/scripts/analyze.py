import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
d=pd.read_pickle('/tmp/analysis/ecod_domains.pkl')
ch=pd.read_pickle('/tmp/analysis/chains.pkl')
seq=dict(zip(zip(ch.pdb,ch.chain),ch.seq))
d['seq']=[seq[k] for k in zip(d.pdb,d.chain)]
IGSF={'V-set','C1-set','I-set','C2-set','C2-set_2','Ig_2','Ig_3','ig','V-set,Mst1_SARAH'}
def grp(r):
    if r.igT and r.ab=='antibody':
        return 'Antibody V' if 'V-set' in str(r.f_name) else ('Antibody C' if 'C1-set' in str(r.f_name) else 'Antibody other Ig')
    if r.igT and r.ab=='TCR':
        return 'TCR V' if 'V-set' in str(r.f_name) else 'TCR C/other'
    if r.igT and r.f_name in IGSF: return 'Other IgSF (V/C1/C2/I-set; MHC, β2m, receptors)'
    if r.igT and r.f_name=='fn3': return 'Fibronectin type III'
    if r.igT: return 'Other Ig/E-set/PapD T-group'
    if r.x_name=='Immunoglobulin-like beta-sandwich': return 'Other Ig-like X-group'
    if r.architecture_name=='beta sandwiches': return 'Other β-sandwiches'
    return r.architecture_name
d['group']=d.apply(grp,axis=1)
d=d[d.group!='Antibody other Ig']
d['yr']=pd.cut(d.year,[0,2014,2017,2019,2023]).astype(str)
d['rb']=d.res_bin.astype(str)
d['logL']=np.log(d.dlen.clip(lower=20))
d['err_frac']=d.err_res/d.dlen

rng=np.random.default_rng(0)
def boot_ci(sub,col='flag',B=400):
    g=sub.groupby('pdb')[col].agg(['sum','count'])
    s,c=g['sum'].values,g['count'].values; n=len(g)
    bs=[]
    for _ in range(B):
        i=rng.integers(0,n,n); bs.append(s[i].sum()/c[i].sum())
    return np.percentile(bs,[2.5,97.5])
rows=[]
for gname,sub in d.groupby('group'):
    if len(sub)<300: continue
    lo,hi=boot_ci(sub)
    # dedup: unique sequence x domain family x start
    u=sub.groupby(['seq','f_id','dstart']).flag.max()
    rows.append(dict(group=gname,domains=len(sub),entries=sub.pdb.nunique(),
        pct_flagged=100*sub.flag.mean(),ci=f'{100*lo:.1f}–{100*hi:.1f}',
        pct_residues_in_errors=100*sub.err_res.sum()/sub.dlen.sum(),
        uniq_seq_domains=len(u),pct_flagged_uniq=100*u.mean(),
        pct_EM=100*(sub.method=='EM').mean(),median_res=sub.res.median()))
tab=pd.DataFrame(rows).sort_values('pct_flagged',ascending=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(tab.round(2).to_string(index=False))
tab.to_csv('/tmp/analysis/table_raw.csv',index=False)

# adjusted logistic
big=d.groupby('group').filter(lambda x: len(x)>=300).copy()
m=smf.logit("flag ~ C(group, Treatment('Antibody V')) + C(method)*C(rb) + C(yr) + logL",data=big).fit(
    disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(big.pdb)[0]})
or_=pd.DataFrame({'OR':np.exp(m.params),'lo':np.exp(m.conf_int()[0]),'hi':np.exp(m.conf_int()[1]),'p':m.pvalues})
or_=or_[or_.index.str.contains('group')]
or_.index=or_.index.str.replace(r"C\(group, Treatment\('Antibody V'\)\)\[T\.",'',regex=True).str.rstrip(']')
print("\nAdjusted OR of being flagged vs Antibody V (method x resolution, year, log domain length; entry-clustered SE)")
print(or_.sort_values('OR').round(3).to_string())
or_.to_csv('/tmp/analysis/table_adjusted_vs_abV.csv')

# within-entry: entries containing antibody V domains AND non-antibody/TCR domains
ab_entries=set(d.pdb[d.group=='Antibody V'])
w=d[d.pdb.isin(ab_entries)].copy()
w['cls']=np.where(w.group.str.startswith('Antibody V'),'Antibody V',np.where(w.group.str.startswith('Antibody C'),'Antibody C',np.where(w.ab=='other','Non-antibody (antigen etc.)','TCR')))
print('\nWithin entries containing antibodies:')
print(w.groupby(['method','cls']).agg(domains=('flag','size'),pct_flagged=('flag',lambda x:100*x.mean()),pct_res=('err_frac',lambda x:100*x.mean())).round(2).to_string())
mw=smf.logit("flag ~ C(cls, Treatment('Antibody V')) + logL",data=w[w.cls!='TCR']).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(w[w.cls!='TCR'].pdb)[0]})
print(np.exp(pd.concat([mw.params,mw.conf_int()],axis=1)).round(3).to_string())
# by method and resolution, Ab V vs comparators
cmp=d[d.group.isin(['Antibody V','Antibody C','Other IgSF (V/C1/C2/I-set; MHC, β2m, receptors)','Fibronectin type III','Other β-sandwiches'])]
print('\n% flagged by method x resolution bin')
print((100*cmp.pivot_table(index=['method','rb'],columns='group',values='flag',aggfunc='mean')).round(1).to_string())
print(cmp.pivot_table(index=['method','rb'],columns='group',values='flag',aggfunc='size').to_string())
d.to_pickle('/tmp/analysis/ecod_grouped.pkl')
