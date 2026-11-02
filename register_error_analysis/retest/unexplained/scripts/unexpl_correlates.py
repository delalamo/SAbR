import json, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250)
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
M=pd.read_pickle(f'{SP}/retest/master2.pkl'); M=M[M.single&M.res.notna()].copy()
U=pd.read_pickle(f'{SP}/unexpl/unexplained_patterns.pkl')
R=pd.read_pickle(f'{SP}/unexpl/runs.pkl')
G=json.load(open(f'{SP}/unexpl/germline.json'))
M['species']=[(G.get(s) or {}).get('species') for s in M.seq]
M['v_gene']=[(G.get(s) or {}).get('v_gene') for s in M.seq]
M['v_id']=[(G.get(s) or {}).get('v_id') for s in M.seq]
M['vfam']=M.v_gene.str.extract(r'^(IG[HKL]V\d+)')[0]
key=set(zip(U[U.pattern=='CDR3→FR4: conserved W/F118 displaced'].pdb,U[U.pattern=='CDR3→FR4: conserved W/F118 displaced'].chain))
M['cdr3fr4']=[int((p,c) in key) for p,c in zip(M.pdb,M.chain)]
ue=set(zip(U.pdb,U.chain)); M['unexpl']=[int((p,c) in ue) for p,c in zip(M.pdb,M.chain)]
ent_types=M.groupby('pdb').ctype.apply(set)
M['vhh']=((M.ctype=='H')&(M.chain_desc.str.contains('nanobod|vhh|sybod|megabod|single.domain',case=False)|M.pdb.map(lambda p: not ({'K','L'} & ent_types[p])))).astype(int)
def rate(x): return f'{100*x.mean():.2f}% (n={len(x)})'
print('== species (ANARCI germline):'); print(M.groupby('species').agg(n=('flag','size'),any_flag=('flag',rate),cdr3fr4=('cdr3fr4',rate),unexplained=('unexpl',rate)).sort_values('n',ascending=False).head(8).to_string())
print('\n== format'); print(M[M.ctype=='H'].groupby('vhh').agg(any_flag=('flag',rate),cdr3fr4=('cdr3fr4',rate)).to_string())
print('\n== CDR3 length (IMGT 105-117 count), CDR3→FR4 displacement rate'); 
M['cdr3_bin']=pd.cut(M.cdr3_len,[0,9,11,13,15,17,20,25,40]).astype(str)
print(M.pivot_table(index='cdr3_bin',columns='ctype',values='cdr3fr4',aggfunc=rate).to_string())
print('\n== by period x method'); M['yb']=pd.cut(M.year,[0,2017,2019,2021,2023,2026],labels=['≤2017','2018–19','2020–21','2022–23','2024–26']).astype(str)
print(M.pivot_table(index='yb',columns='method',values='cdr3fr4',aggfunc=rate).to_string())
print('\n== starting model type (CDR3→FR4)'); print(M.groupby(['method','irm']).cdr3fr4.agg(rate).unstack().to_string())
print('\n== V-gene family (heavy, ≥150 chains)'); vf=M[M.ctype=='H'].groupby('vfam').agg(n=('cdr3fr4','size'),cdr3fr4=('cdr3fr4','mean'),any_flag=('flag','mean')); print((vf[vf.n>=150].assign(cdr3fr4=lambda d:(100*d.cdr3fr4).round(2),any_flag=lambda d:(100*d.any_flag).round(2))).sort_values('cdr3fr4',ascending=False).to_string())
print('\n== germline identity of V region (v_id), CDR3→FR4 rate by bin'); M['vid_bin']=pd.cut(M.v_id,[0,0.7,0.8,0.9,0.95,1.01]).astype(str); print(M.groupby('vid_bin').cdr3fr4.agg(rate).to_string())
m=smf.logit("cdr3fr4 ~ C(ctype) + cdr3_len + I(cdr3_len**2) + vhh + C(method)*C(rb) + C(yb) + lq_z",data=M.dropna(subset=['cdr3_len','lq_z'])).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(M.dropna(subset=['cdr3_len','lq_z']).pdb)[0]})
p=m.params; c=m.conf_int(); print('\n== adjusted model for CDR3→FR4 displacement'); print(pd.DataFrame({'OR':np.exp(p),'lo':np.exp(c[0]),'hi':np.exp(c[1]),'p':m.pvalues}).round(3).loc[lambda d: ~d.index.str.contains('Intercept|rb')].to_string())
M.to_pickle(f'{SP}/unexpl/master3.pkl')
