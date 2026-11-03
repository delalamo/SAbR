"""Build the PDB antibody V-domain database (all resolutions, all PDB) from ANARCI IMGT numbering.
One row per unique V-domain sequence per chain type, with earliest release date and the set of entries.
Also IMGT-aligned residue matrix (uint8; 0 = gap), germline assignment (vectorised, ANARCI-equivalent identity).
"""
import json, pickle, collections, numpy as np, pandas as pd
from anarci.germlines import all_germlines as G
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/seq_novelty'
num=pickle.load(open(f'{C}/anarci_num.pkl','rb'))
cand=json.load(open(f'{C}/cand_seqs.json'))
# release / deposit dates
rel={}; dep={}
for l in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    d=json.loads(l); a=d.get('rcsb_accession_info') or {}
    pid=d['rcsb_id'].lower()
    if a.get('initial_release_date'): rel[pid]=a['initial_release_date'][:10]
    if a.get('deposit_date'): dep[pid]=a['deposit_date'][:10]
print('release dates', len(rel))
AA='ACDEFGHIKLMNPQRSTVWYX'
code={a:i+1 for i,a in enumerate(AA)}
def keyorder(k):
    p,ins=k
    if ins==' ': return (p,0,0)
    r=ord(ins)-64
    if p in (33,61,112): return (p,-1,-r)
    return (p,1,r)
dom=collections.defaultdict(lambda: {'entries':set(),'chains':[]})
for s,doms in num.items():
    chs=cand[s]
    for (ct,st,en,ev,nb) in doms:
        if ct not in ('H','K','L'): continue
        vs=''.join(a for _,_,a in nb)
        key=(ct,vs)
        d=dom[key]; d['nb']=nb
        for c in chs:
            d['entries'].add(c[:4]); d['chains'].append(c)
        d['multi']=d.get('multi',False) or len(doms)>1
print('unique V domains', len(dom), collections.Counter(k[0] for k in dom))
keys=sorted({(p,i) for d in dom.values() for p,i,_ in d['nb']}, key=keyorder)
print('columns', len(keys))
kidx={k:i for i,k in enumerate(keys)}
rows=[]; M=np.zeros((len(dom),len(keys)),dtype=np.uint8)
for r,((ct,vs),d) in enumerate(dom.items()):
    for p,i,a in d['nb']: M[r,kidx[(p,i)]]=code.get(a,21)
    rels=sorted(rel[e] for e in d['entries'] if e in rel)
    deps=sorted(dep[e] for e in d['entries'] if e in dep)
    rows.append({'ctype':ct,'vdom':vs,'n_entries':len(d['entries']),'first_release':rels[0] if rels else None,
                 'first_deposit':deps[0] if deps else None,'entries':' '.join(sorted(d['entries'])),'multi_domain_chain':d['multi']})
D=pd.DataFrame(rows)
# germline assignment, ANARCI-equivalent: identity over germline non-gap match positions 1..128
match_cols=np.array([kidx.get((p,' '),-1) for p in range(1,129)])
S=np.zeros((len(D),128),dtype=np.uint8)
ok=match_cols>=0
S[:,ok]=M[:,match_cols[ok]]
def gl_mat(seg,ct,species):
    names=[];mats=[]
    for sp in species:
        for g,sq in G[seg].get(ct,{}).get(sp,{}).items():
            names.append((sp,g)); mats.append([code.get(a,0) if a!='-' else 0 for a in sq])
    return names,np.array(mats,dtype=np.uint8)
def assign(rowsel,ct,species,tag):
    names,GM=gl_mat('V',ct,species)
    Sx=S[rowsel]
    gnz=(GM>0)
    best=np.zeros(len(Sx),dtype=int); bid=np.zeros(len(Sx))
    for b in range(0,len(Sx),2000):
        X=Sx[b:b+2000]
        m=(X[:,None,:]==GM[None,:,:])&gnz[None]
        idn=m.sum(2)/gnz.sum(1)[None]
        best[b:b+2000]=idn.argmax(1); bid[b:b+2000]=idn.max(1)
    vsp=[names[i][0] for i in best]; vg=[names[i][1] for i in best]
    # J gene within assigned species
    jg=[None]*len(Sx); jid=np.full(len(Sx),np.nan)
    for sp in set(vsp):
        jn,JM=gl_mat('J',ct,[sp])
        if not len(jn): continue
        sel=np.array([v==sp for v in vsp])
        X=Sx[sel]; jnz=JM>0
        m=(X[:,None,:]==JM[None])&jnz[None]; idn=m.sum(2)/jnz.sum(1)[None]
        bi=idn.argmax(1)
        for k,(ii,v) in enumerate(zip(np.where(sel)[0],bi)): jg[ii]=jn[v][1]; jid[ii]=idn[k,v]
    idx=np.where(rowsel)[0]
    D.loc[idx,f'species_{tag}']=vsp; D.loc[idx,f'v_gene_{tag}']=vg; D.loc[idx,f'v_id_{tag}']=bid
    D.loc[idx,f'j_gene_{tag}']=jg; D.loc[idx,f'j_id_{tag}']=jid
for ct in 'HKL':
    sel=(D.ctype==ct).values
    assign(sel,ct,['human','mouse'],'hm')
    assign(sel,ct,[sp for sp in ['human','mouse','rat','rabbit','rhesus','pig','alpaca','cow'] if sp in G['V'][ct]],'all')
# CDR lengths (IMGT)
reg={'FR1':(1,26),'CDR1':(27,38),'FR2':(39,55),'CDR2':(56,65),'FR3':(66,104),'CDR3':(105,117),'FR4':(118,128)}
pos=np.array([k[0] for k in keys])
for r,(a,b) in reg.items():
    cols=(pos>=a)&(pos<=b)
    D['len_'+r]=(M[:,cols]>0).sum(1)
pickle.dump({'D':D,'M':M,'keys':keys,'code':code,'S':S},open(f'{C}/abdb.pkl','wb'))
print(D.groupby('ctype')[['n_entries']].describe())
print(D.species_all.value_counts())
print(D.head())
