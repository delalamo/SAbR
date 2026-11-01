import re, pandas as pd, numpy as np, collections
ch=pd.read_pickle('/tmp/analysis/chains.pkl')
d=pd.read_pickle('/tmp/analysis/ecod_sub.pkl')
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx').drop(columns=['Unnamed: 0'],errors='ignore')
t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
seg_re=re.compile(r'([^:,]+):(-?\d+)[A-Za-z]?-(-?\d+)[A-Za-z]?')
def segs(s): return [(int(a),int(b)) for c,a,b in seg_re.findall(s)]
d['segs']=d.pdb_range.map(segs)
d=d[d.segs.map(len)>0]
keys=set(zip(ch.pdb,ch.chain))
d=d[[k in keys for k in zip(d.pdb,d.chain)]].reset_index(drop=True)
d['dlen']=d.segs.map(lambda s: sum(b-a+1 for a,b in s))
d['dstart']=d.segs.map(lambda s: min(a for a,b in s))
d['n_err']=0; d['err_res']=0
bykey=d.groupby(['pdb','chain']).groups
assigned=unassigned=nochop=0
for r in t.itertuples():
    k=(r.pdb,r.chain)
    if k not in bykey: nochop+=1; continue
    best,bo=None,0
    for i in bykey[k]:
        ov=sum(max(0,min(b,r.re)-max(a,r.rs)+1) for a,b in d.at[i,'segs'])
        if ov>bo: best,bo=i,ov
    if best is None: unassigned+=1; continue
    assigned+=1; d.at[best,'n_err']+=1; d.at[best,'err_res']+=bo
print(f'errors: assigned to ECOD domain {assigned}, on chain but outside domains {unassigned}, chain not in ECOD {nochop}, total {len(t)}')
d['flag']=(d.n_err>0).astype(int)
d=d.merge(ch[['pdb','chain','method','res','res_bin','year','ab','length','desc']],on=['pdb','chain'])
igT=d.t_name=='Immunoglobulin/Fibronectin type III/E set domains/PapD-like'
d['igT']=igT
d['ig_rank']=d[igT].groupby(['pdb','chain']).dstart.rank(method='first')
print(d[igT&(d.ab=='antibody')].groupby('ig_rank').f_name.value_counts().groupby(level=0).head(4).head(16).to_string())
print(d[igT&(d.ab=='other')].f_name.value_counts().head(20).to_string())
d.drop(columns=['segs']).to_pickle('/tmp/analysis/ecod_domains.pkl')
