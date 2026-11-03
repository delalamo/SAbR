"""Pick high-resolution (<=2.6 A) X-ray PDB chains whose V domain occupies the same rare IMGT column families as the
SN3 positives (and, as a broader sample, any rare column in windows A, B, D, E). Map V domain -> chain via pdb_seqres."""
import pickle, gzip, numpy as np, pandas as pd, re
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
db=pickle.load(open(f'{SP}/deeper/cache/seq_novelty/abdb.pkl','rb')); D,M,keys=db['D'],db['M'],db['keys']
kidx={k:i for i,k in enumerate(keys)}; pos=np.array([k[0] for k in keys])
# resolution / method
res={}
for l in open('/tmp/supp/pdbwide/entries.idx',encoding='latin-1'):
    f=l.rstrip('\n').split('\t')
    if len(f)<8 or len(f[0])!=4: continue
    try: r=float(f[6])
    except: r=np.nan
    res[f[0].lower()]=(r,f[7].strip())
occfrac={}
for ct in 'HKL':
    sel=(D.ctype==ct).values; occfrac[ct]=(M[sel]>0).mean(0)
FAM={'H_4A':('H',[(4,'A')]),'H_FR1_9ABC_10':('H',[(9,'A'),(9,'B'),(9,'C'),(10,' ')]),'L_10':('L',[(10,' ')]),
     'H_67ABC':('H',[(67,'A'),(67,'B'),(67,'C')]),'H_70A':('H',[(70,'A')]),'H_73':('H',[(73,' ')]),
     'K_58_63_64':('K',[(58,' '),(63,' '),(64,' ')]),'K_119A':('K',[(119,'A')]),'H_85A-D':('H',[(85,'A'),(85,'B')])}
WIN={'A':(1,13),'B':(55,72),'D':(105,128),'E':(116,128),'F':(24,41)}
cand=[]
for i,r in D.iterrows():
    ct=r.ctype; q=M[i]>0
    rare=q&(occfrac[ct]<0.01)
    if not rare.any(): continue
    fams=[nm for nm,(c,cols) in FAM.items() if c==ct and any(q[kidx[k]] for k in cols if k in kidx)]
    wins=[w for w,(a,b) in WIN.items() if (rare&(pos>=a)&(pos<=b)).any()]
    # skip ultralong CDR3 (bovine already tested by another verifier) unless also other window
    for e in str(r.entries).split():
        rr,mt=res.get(e,(np.nan,''))
        if mt=='X-RAY DIFFRACTION' and rr<=2.6:
            cand.append(dict(dbrow=i,ctype=ct,vdom=r.vdom,entry=e,res=rr,fams=';'.join(fams),wins=';'.join(wins),
                             rare_cols=' '.join(f'{keys[j][0]}{keys[j][1].strip()}' for j in np.where(rare)[0])[:80],species=r.species_all))
C=pd.DataFrame(cand)
print('candidate (vdom, entry) pairs',len(C),'distinct vdoms',C.dbrow.nunique())
# one best-resolution entry per V domain
C=C.sort_values('res').drop_duplicates('dbrow')
# map to chain id via seqres
need=set(C.entry)
seqs={}
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    hdr=None
    for l in fh:
        if l[0]=='>':
            hdr=l[1:].split()[0]; e,c=hdr.split('_',1); hdr=(e,c) if e in need else None
        elif hdr: seqs.setdefault(hdr[0],[]).append((hdr[1],l.strip()))
def chain_of(e,v):
    for c,s in seqs.get(e,[]):
        if v[5:40] in s: return c,s
    return None,None
C[['chain','seqres']]=[chain_of(e,v) for e,v in zip(C.entry,C.vdom)]
C=C[C.chain.notna()]
# selection: per family up to 6 V domains; plus a random sample of other rare-col V domains with windows A/B/D/E (no family), up to 20
rng=np.random.default_rng(1)
sel=[]
for nm in FAM:
    x=C[C.fams.str.contains(nm,regex=False)]
    sel.append(x.head(8).assign(group=nm))
rest=C[(C.fams=='')&(C.wins.str.contains('A|B|D|E'))]
rest=rest[~rest.rare_cols.str.contains('111H')]   # ultralong CDR3 covered elsewhere
sel.append(rest.sample(min(30,len(rest)),random_state=1).assign(group='other_rare_ABDE'))
S=pd.concat(sel).drop_duplicates('dbrow')
print(S.groupby('group').size()); print(C.groupby(C.fams.replace('','none')).size())
S.drop(columns=['seqres']).to_csv(f'{OUT}/v04_hires_selection.csv',index=False)
