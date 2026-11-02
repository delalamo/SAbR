"""Structure-only check: burial of the conserved FR4 anchor (ANARCI IMGT 118: Trp in heavy, Phe in light)."""
import json, sys, io, urllib.request, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import MMCIFParser
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
U=pd.read_pickle(f'{SP}/unexpl/unexplained_patterns.pkl'); M=pd.read_pickle(f'{SP}/retest/master2.pkl'); M=M[M.single]
recs={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r=json.loads(l)
        if r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r['residues']
tp={}
for l in open(f'{SP}/retest/templates.jsonl'):
    r=json.loads(l)
    if r.get('status')=='ok': tp[(r['pdb'],r['chain'])]=r['residues']
def fetch(p,c):
    for url in (f'https://www.ebi.ac.uk/pdbe/model-server/v1/{p}/atoms?auth_asym_id={c}&encoding=cif',f'https://models.rcsb.org/v1/{p}/atoms?auth_asym_id={c}&encoding=cif'):
        for k in range(3):
            try: return urllib.request.urlopen(url,timeout=60).read().decode()
            except Exception: time.sleep(1+k)
    return None
def burial(job):
    p,c,res,label=job
    try:
        v=[x for x in res if x[4] is not None]
        a118=[x for x in v if x[4]==118 and not x[5]]
        if not a118: return None
        auth=(a118[0][0],a118[0][1]); txt=fetch(p,c)
        if not txt: return None
        ch=next(iter(MMCIFParser(QUIET=True).get_structure(p,io.StringIO(txt))))[c]
        vauth={(x[0],x[1]) for x in v}
        ca=np.array([r['CA'].coord for r in ch if (r.id[1],r.id[2].strip()) in vauth and 'CA' in r])
        res118=[r for r in ch if (r.id[1],r.id[2].strip())==auth]
        if not res118: return None
        sc=[a.coord for a in res118[0] if a.get_id() not in ('N','CA','C','O','CB')]
        if not sc: return dict(pdb=p,chain=c,label=label,aa=res118[0].get_resname(),sidechain='missing')
        cen=np.mean(sc,axis=0)
        return dict(pdb=p,chain=c,label=label,aa=res118[0].get_resname(),dist_centroid=float(np.linalg.norm(cen-ca.mean(0))),
                    ca_within_10A=int((np.linalg.norm(ca-cen,axis=1)<10).sum()))
    except Exception as e:
        return None
jobs=[]
for r in U[U.pattern.isin(['CDR3→FR4: conserved W/F118 displaced','FR4: SAbR skips IMGT 120 after W/F118-G119'])].drop_duplicates(['pdb','chain']).itertuples():
    jobs.append((r.pdb,r.chain,recs[(r.pdb,r.chain)],r.pattern+' | '+r.method))
clean=M[M.flag==0]
for m in ['EM','X-ray']:
    for q in clean[clean.method==m].sample(250,random_state=5).itertuples():
        if (q.pdb,q.chain) in recs: jobs.append((q.pdb,q.chain,recs[(q.pdb,q.chain)],'clean V domain | '+m))
for (p,c),res in list(tp.items())[:300]: jobs.append((p,c,res,'high-res template (<3 Å)'))
with ThreadPoolExecutor(16) as ex: out=[o for o in ex.map(burial,jobs) if o]
B=pd.DataFrame(out); B.to_csv(f'{SP}/unexpl/w118_burial.csv',index=False)
B=B[B.sidechain.isna()] if 'sidechain' in B else B
pd.set_option('display.width',250)
print(B.groupby('label').agg(n=('pdb','size'),dist_to_domain_centroid=('dist_centroid','median'),CA_within_10A=('ca_within_10A','median'),
    exposed_frac=('ca_within_10A',lambda x: round(100*(x<14).mean(),0))).round(1).to_string())
