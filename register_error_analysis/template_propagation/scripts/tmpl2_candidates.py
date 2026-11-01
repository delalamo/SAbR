import gzip, json, re, collections, pandas as pd, numpy as np
T='/tmp/tmpl'
t=pd.read_pickle(f'{T}/errors_seqmapped.pkl')
# pdb_seqres unique sequences
seq2chains=collections.defaultdict(list)
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    h=None; mol=None
    for line in fh:
        if line.startswith('>'): h=line[1:].split()[0]; mol=line.split()[1]
        elif mol=='mol:protein':
            p,c=h.split('_',1); seq2chains[line.strip()].append((p,c))
useqs=list(seq2chains); big='\x00'.join(useqs); starts=np.cumsum([0]+[len(s)+1 for s in useqs[:-1]])
print('unique protein seqs',len(useqs))
idx={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    f=l.split('\t')
    if len(f)<8: continue
    try: dt=pd.to_datetime(f[2],format='%m/%d/%y')
    except Exception: continue
    try: r=float(f[6].split(',')[0])
    except Exception: r=np.nan
    auth=[a.strip() for a in re.split(r'(?<=\.),\s*',f[5]) if a.strip()]
    idx[f[0].lower()]=(dt,r,f[7] if f[7] else 'X-RAY DIFFRACTION',auth[-1] if auth else None)
cite={}
for l in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    e=json.loads(l); c=e.get('rcsb_primary_citation') or {}
    cite[e['rcsb_id'].lower()]=c.get('pdbx_database_id_DOI') or (('PMID'+str(c['pdbx_database_id_PubMed'])) if c.get('pdbx_database_id_PubMed') else None)
fams=[]; cand=[]
for fid,g in t.groupby('family'):
    s=g.seq.iloc[0]; s0=int(g.s0.min()); s1=int(g.s1.max())
    a=max(0,s0-12); b=min(len(s),s1+13)
    if b-a<25: a=max(0,a-(25-(b-a))//2-1); b=min(len(s),a+25)
    w=s[a:b]
    hits=[]; pos=big.find(w)
    while pos!=-1:
        k=np.searchsorted(starts,pos,side='right')-1; hits.append((k,pos-starts[k]-a)); pos=big.find(w,pos+1)
    ents={}
    for k,off in hits:
        for p,c in seq2chains[useqs[k]]:
            if p not in ents: ents[p]=(c,off)
    first=g.sort_values('YEAR').iloc[0]
    fams.append(dict(family=fid,s0=s0,s1=s1,win=w,shift=int(g.SHIFT.mode().iloc[0]),n_flagged_entries=g.pdb.nunique(),n_pdb_entries=len(ents)))
    flagged=set(g.pdb)
    for p,(c,off) in ents.items():
        dt,r,meth,last=idx.get(p,(pd.NaT,np.nan,None,None))
        cand.append(dict(family=fid,pdb=p,chain=c,offset=off,date=dt,res=r,method=meth,last_author=last,paper=cite.get(p) or p,flagged=p in flagged))
F=pd.DataFrame(fams); C=pd.DataFrame(cand)
pp=C.groupby('family').agg(papers=('paper','nunique'),labs=('last_author','nunique'))
F=F.merge(pp,left_on='family',right_index=True,how='left')
F.to_pickle(f'{T}/families.pkl'); C.to_pickle(f'{T}/candidates.pkl')
print(F[['n_flagged_entries','n_pdb_entries','papers']].describe(percentiles=[.5,.75,.9,.99]).round(1))
print('families whose sequence window appears in >=2 papers:',(F.papers>=2).sum(),' >=3 papers:',(F.papers>=3).sum())
print('total candidate entries:',len(C))
