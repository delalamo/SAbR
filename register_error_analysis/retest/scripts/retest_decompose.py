import json, sys, gzip, collections, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
sys.path.insert(0,f'{SP}/scripts'); from sabr_eval import segments, region
M=pd.read_pickle(f'{SP}/retest/master2.pkl')
recs={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r=json.loads(l)
        if r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r['residues']
def runs(res):
    out=[];cur=[]
    v=[x for x in res if x[4] is not None]
    for i,x in enumerate(v+[None]):
        if x is not None and (x[4],x[5] or '')!=(x[6],x[7] or ''): cur.append((i,x))
        else:
            if len(cur)>=3:
                d=[y[6]-y[4] for _,y in cur if region(y[4])=='FR' and not (y[5] or y[7])]
                out.append(dict(i0=cur[0][0],i1=cur[-1][0],a0=cur[0][1][4],delta=int(np.median(d)) if d else None))
            cur=[]
    return v,out
# templates
tp={}
for l in open(f'{SP}/retest/templates.jsonl'):
    r=json.loads(l)
    if r.get('status')=='ok': tp[(r['pdb'],r['chain'])]=r['residues']
info={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    f=l.split('\t')
    if len(f)<8: continue
    try: info[f[0].lower()]=pd.to_datetime(f[2],format='%m/%d/%y')
    except Exception: pass
tkey=collections.defaultdict(list)
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    h=None; mol=None
    for line in fh:
        if line.startswith('>'): h=line[1:].split()[0]; mol=line.split()[1]
        elif mol=='mol:protein' and len(line)>105:
            p,c=h.split('_',1)
            if (p,c) in tp: tkey[line[:105]].append((info.get(p),p,c))
tsig={}
for k,lst in tkey.items():
    s=[]
    for d,p,c in lst:
        _,R=runs(tp[(p,c)]); s+=[(d,r['a0'],r['delta']) for r in R]
    tsig[k]=s
# earlier 3-5 A scanned structures of the same V (other papers)
sig=collections.defaultdict(list)
for r in M.itertuples():
    res=recs.get((r.pdb,r.chain))
    if not res: continue
    _,R=runs(res)
    for x in R: sig[r.seq[:105]].append((pd.Timestamp(r.deposit),r.cite_key,x['a0'],x['delta']))
rows=[]
for r in M[(M.flag==1)&M.single].itertuples():
    res=recs.get((r.pdb,r.chain))
    if not res: continue
    v,R=runs(res); si=[x[3] for x in v]
    gaps={i for i in range(len(si)-1) if si[i+1]-si[i]>1 if si[i] is not None and si[i+1] is not None}
    k=r.seq[:105]; dep=pd.Timestamp(r.deposit)
    def same(lst,a0,dl): return any(abs(a-a0)<=2 and d==dl for a,d in lst)
    cats=[]
    for x in R:
        if same([(a,d) for dd,a,d in tsig.get(k,[]) if dd is not None and dd<dep],x['a0'],x['delta']): cats.append('1 same error in an earlier high-res (<3 Å) template')
        elif same([(a,d) for dd,ck,a,d in sig.get(k,[]) if dd<dep and ck!=r.cite_key],x['a0'],x['delta']): cats.append('2 same error in an earlier 3–5 Å structure (other paper)')
        elif x['i0']<=2 or v[0][4]>3 and x['i0']<=4 or any(x['i0']-3<=g<=x['i1']+3 for g in gaps): cats.append('3 at the N-terminus or next to an unmodelled gap')
        elif 78<=x['a0']<=86 and (x['delta'] is None or any(y[5] for y in v if region(y[4])=='FR')): cats.append('4 FR3 insertion-convention region (possible artifact)')
        else: cats.append('5 other (origin not explained)')
    rows.append(dict(pdb=r.pdb,chain=r.chain,era=r.era,cat=sorted(cats)[0]))
D=pd.DataFrame(rows)
t=pd.crosstab(D.cat,D.era,margins=True); p=(100*pd.crosstab(D.cat,D.era,normalize='columns')).round(0)
print(pd.concat([t,p.add_suffix(' %')],axis=1).to_string())
D.to_csv(f'{SP}/retest/flag_origin.csv',index=False)
