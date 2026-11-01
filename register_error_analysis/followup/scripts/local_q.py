import json, re, numpy as np, pandas as pd
A='/tmp/analysis'
d=pd.read_pickle(f'{A}/ecod_grouped.pkl')
ecod=pd.read_pickle(f'{A}/ecod_sub.pkl')[['ecod_domain_id','pdb','chain','pdb_range']]
d=d.merge(ecod,on=['ecod_domain_id','pdb','chain'],how='left',suffixes=('','_y'))
seg_re=re.compile(r'([^:,]+):(-?\d+)[A-Za-z]?-(-?\d+)[A-Za-z]?')
loc={}
for l in open('/tmp/supp/meta2/local_35.jsonl'):
    r=json.loads(l); loc[(r['pdb'],r['chain'])]=r
ents=pd.read_pickle(f'{A}/entries35.pkl').set_index('pdb')
out=[]
for row in d[d.pdb.isin(set(k[0] for k in loc))].itertuples():
    r=loc.get((row.pdb,row.chain))
    if r is None: continue
    segs=[(int(a),int(b)) for c,a,b in seg_re.findall(row.pdb_range or '')]
    am=r['auth_map'] or []
    sids=[i+1 for i,lab in enumerate(am) if lab and re.match(r'-?\d+',lab) and any(a<=int(re.match(r'-?\d+',lab).group())<=b for a,b in segs)]
    rec={'ecod_domain_id':row.ecod_domain_id,'pdb':row.pdb,'chain':row.chain}
    for t in ('Q_SCORE','RSCC'):
        v=r['feats'].get(t,{})
        vals=[v[str(s)] for s in sids if str(s) in v]
        rec[t]=np.mean(vals) if vals else np.nan
        rec[t+'_n']=len(vals)
    rec['n_modeled']=len(sids)
    out.append(rec)
q=pd.DataFrame(out); q.to_pickle(f'{A}/domain_localq.pkl')
q=q.merge(ents[['method']],left_on='pdb',right_index=True)
print(q.groupby('method')[['Q_SCORE','RSCC']].agg(lambda x: x.notna().mean()).round(3))
print(q.groupby('method')[['Q_SCORE','RSCC']].mean().round(3))
