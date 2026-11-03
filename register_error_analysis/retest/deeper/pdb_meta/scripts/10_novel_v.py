"""For each V chain: was an identical V region already in the PDB (any entry deposited before this entry)?
Keys = 20 residues before the FR4 W/F-G-x-G motif + motif; candidate hits verified by full V-substring match."""
import gzip, re, collections, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','deposit','vseq','ctype','method'])
MOT=re.compile(r'(?=([WF][GAS].G))')
def keys(s):
    return [s[m.start()-20:m.start()+4] for m in MOT.finditer(s) if m.start()>=20]
vkey={}
for v in ch.vseq.dropna().unique():
    k=keys(v)
    if k: vkey[v]=k[-1]
want=set(vkey.values())
dep={}
for l in open('/tmp/supp/pdbwide/entries.idx',encoding='latin-1'):
    p=l.rstrip('\n').split('\t')
    if len(p)>3 and len(p[0])==4:
        try: dep[p[0].lower()]=pd.Timestamp(pd.to_datetime(p[2],format='%m/%d/%y'))
        except Exception: pass
hits=collections.defaultdict(list)   # key -> [(pdb, seq)]
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as f:
    hdr=None
    for line in f:
        if line.startswith('>'): hdr=line; continue
        if 'mol:protein' not in hdr: continue
        s=line.strip()
        for k in set(keys(s)):
            if k in want: hits[k].append((hdr[1:5],s))
ch['deposit']=pd.to_datetime(ch.deposit)
first={}  # vseq -> earliest deposit date of an entry containing it
cache={}
rows=[]
for v,k in vkey.items():
    ent=sorted({(dep.get(p,pd.Timestamp('2100-01-01')),p) for p,s in hits.get(k,[]) if v in s})
    cache[v]=ent
out=[]
for _,c in ch.iterrows():
    ent=cache.get(c.vseq,[])
    earlier=[(d,p) for d,p in ent if d<c.deposit and p!=c.pdb]
    out.append({'pdb':c.pdb,'chain':c.chain,'n_prior_entries_same_v':len(earlier),'first_prior':earlier[0][1] if earlier else None,
                'novel_v':len(earlier)==0,'n_entries_same_v_total':len(ent)})
O=pd.DataFrame(out); O.to_csv(f'{OUT}/chain_novel_v.csv.gz',index=False)
m=ch.merge(O,on=['pdb','chain'])
print(O.novel_v.mean(), m.groupby('method').novel_v.mean())
