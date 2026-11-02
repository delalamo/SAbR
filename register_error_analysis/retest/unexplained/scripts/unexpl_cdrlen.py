import json, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250)
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
M=pd.read_pickle(f'{SP}/unexpl/master3.pkl'); R=pd.read_pickle(f'{SP}/unexpl/runs.pkl'); U=pd.read_pickle(f'{SP}/unexpl/unexplained_patterns.pkl')
ac=json.load(open(f'{SP}/sabr/anarci_cache.json'))
def lens(s):
    d=ac.get(s) or []
    if len(d)!=1: return (np.nan,)*3
    n=[q[1] for q in d[0]['pos']]
    return (sum(27<=x<=38 for x in n),sum(56<=x<=65 for x in n),sum(105<=x<=117 for x in n))
L=[lens(s) for s in M.seq]; M['L1'],M['L2'],M['L3']=zip(*L)
# typicality: frequency of that length among all chains of the same type (unique sequences)
uq=M.drop_duplicates('seq')
for c in ['L1','L2','L3']:
    freq=uq.groupby(['ctype',c]).size()/uq.groupby('ctype').size()
    M[c+'_freq']=[freq.get((t,v),np.nan) for t,v in zip(M.ctype,M[c])]
    M[c+'_atyp']=(M[c+'_freq']<0.10).astype(int)
# runs touching each CDR (start or end within CDR or within 2 IMGT positions of its boundaries)
bounds={'L1':(25,40),'L2':(54,67),'L3':(103,120)}
for c,(a,b) in bounds.items():
    hit=R[(R.imgt1>=a)&(R.imgt0<=b)]
    M['run_'+c]=[int(k in set(zip(hit.pdb,hit.chain))) for k in zip(M.pdb,M.chain)]
def rate(x): return f'{100*x.mean():.2f}% (n={len(x)})'
for c in ['L1','L2','L3']:
    print(f'== disagreement run touching CDR{c[1]}: typical vs atypical CDR{c[1]} length (<10% of chains of that type)')
    print(M.pivot_table(index=c+'_atyp',columns='ctype',values='run_'+c,aggfunc=rate).to_string())
    m=smf.logit(f"run_{c} ~ {c}_atyp + C(ctype) + C(method)*C(rb) + C(yb) + lq_z",data=M.dropna(subset=[c+'_freq','lq_z'])).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(M.dropna(subset=[c+'_freq','lq_z']).pdb)[0]})
    print('   adjusted OR atypical length: %.2f [%.2f-%.2f] p=%.2g'%(np.exp(m.params[c+'_atyp']),*np.exp(m.conf_int().loc[c+'_atyp']),m.pvalues[c+'_atyp']))
# reclassify unexplained runs
U=U.merge(M[['pdb','chain','L1_atyp','L2_atyp','L3_atyp']],on=['pdb','chain'],how='left')
def near_atyp(r):
    return (r.L3_atyp==1 and r.imgt1>=103) or (r.L2_atyp==1 and r.imgt0<=67 and r.imgt1>=54) or (r.L1_atyp==1 and r.imgt0<=40 and r.imgt1>=25)
U['near_atypical_cdr']=U.apply(near_atyp,axis=1)
print('\n== unexplained runs: share next to an atypical-length CDR, and share repeated in another entry of the same paper')
print(U.groupby('pattern').agg(runs=('pdb','size'),near_atypical_cdr=('near_atypical_cdr',lambda x: round(100*x.mean())),same_paper=('same_paper_other_entry',lambda x: round(100*x.mean()))).sort_values('runs',ascending=False).to_string())
print('all unexplained: near atypical CDR %.0f%%, same paper %.0f%%'%(100*U.near_atypical_cdr.mean(),100*U.same_paper_other_entry.mean()))
# control: share of all V domains with an atypical CDR at those positions
print('baseline: chains with any atypical CDR length: %.0f%%'%(100*M[['L1_atyp','L2_atyp','L3_atyp']].max(axis=1).mean()))
U.to_pickle(f'{SP}/unexpl/unexplained_final.pkl'); M.to_pickle(f'{SP}/unexpl/master4.pkl')
