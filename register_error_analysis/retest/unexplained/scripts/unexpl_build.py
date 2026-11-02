"""Run-level origin categories and features for every disagreement run in single-domain chains."""
import json, sys, gzip, collections, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
sys.path.insert(0,f'{SP}/scripts'); from sabr_eval import region
M=pd.read_pickle(f'{SP}/retest/master2.pkl'); M=M[M.single]
recs={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r=json.loads(l)
        if r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r['residues']
AA1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def runs(res):
    v=[x for x in res if x[4] is not None]; out=[]; cur=[]
    for i,x in enumerate(v+[None]):
        if x is not None and (x[4],x[5] or '')!=(x[6],x[7] or ''): cur.append(i)
        else:
            if len(cur)>=3:
                d=[v[j][6]-v[j][4] for j in cur if region(v[j][4])=='FR' and not (v[j][5] or v[j][7])]
                out.append(dict(i0=cur[0],i1=cur[-1],a0=v[cur[0]][4],a1=v[cur[-1]][4],delta=int(np.median(d)) if d else None))
            cur=[]
    return v,out
# templates and earlier 3-5 A signatures (as in retest_decompose)
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
tsig={k:[(d,r['a0'],r['delta']) for d,p,c in lst for r in runs(tp[(p,c)])[1]] for k,lst in tkey.items()}
sig=collections.defaultdict(list)
for r in M.itertuples():
    res=recs.get((r.pdb,r.chain))
    if res:
        for x in runs(res)[1]: sig[r.seq[:105]].append((pd.Timestamp(r.deposit),r.cite_key,r.pdb,x['a0'],x['delta']))
rows=[]
for r in M[M.flag==1].itertuples():
    res=recs.get((r.pdb,r.chain))
    if not res: continue
    v,R=runs(res); si=[x[3] for x in v]
    gaps={i for i in range(len(si)-1) if si[i] is not None and si[i+1] is not None and si[i+1]-si[i]>1}
    k=r.seq[:105]; dep=pd.Timestamp(r.deposit)
    def same(lst,a0,dl): return any(abs(a-a0)<=2 and d==dl for a,d in lst)
    for x in R:
        if same([(a,d) for dd,a,d in tsig.get(k,[]) if dd is not None and dd<dep],x['a0'],x['delta']): cat='template'
        elif same([(a,d) for dd,ck,p,a,d in sig.get(k,[]) if dd<dep and ck!=r.cite_key],x['a0'],x['delta']): cat='earlier 3-5A other paper'
        elif x['i0']<=2 or (v[0][4]>3 and x['i0']<=4) or any(x['i0']-3<=g<=x['i1']+3 for g in gaps): cat='N-term/gap'
        elif 78<=x['a0']<=86 and (x['delta'] is None or any(y[5] for y in v if region(y[4])=='FR')): cat='FR3 convention'
        else: cat='unexplained'
        seg=v[x['i0']:x['i1']+1]
        aa=''.join(AA1.get(y[2],'X') for y in v)
        same_paper=any(abs(a-x['a0'])<=2 and d==x['delta'] and p!=r.pdb for dd,ck,p,a,d in sig.get(k,[]) if ck==r.cite_key)
        rows.append(dict(pdb=r.pdb,chain=r.chain,era=r.era,method=r.method,res=r.res,year=r.year,ctype=r.ctype,cat=cat,
            imgt0=x['a0'],imgt1=x['a1'],delta=x['delta'],length=x['i1']-x['i0']+1,i0=x['i0'],i1=x['i1'],seqkey=k,v_aa=aa,
            same_paper_other_entry=same_paper,last_author=r.last_author,paper=r.cite_key,irm=r.irm,fit=r.fit,lq_z=r.lq_z,
            chain_desc=r.chain_desc,role=r.role,sars2=r.sars2,deposit=r.deposit))
R=pd.DataFrame(rows); R.to_pickle(f'{SP}/unexpl/runs.pkl')
print(R.cat.value_counts().to_string()); print('chains',R.groupby(['pdb','chain']).ngroups)
