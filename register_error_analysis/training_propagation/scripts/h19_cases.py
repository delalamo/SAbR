"""Build Fv inputs (paired VH/VL V-region sequences) for erroneous and clean antibody chains.
For a flagged chain, the partner chain is the closest V domain of the other type (CA centroid distance).
Writes cases.json: [{id, pdb, H, L, Hseq, Lseq, flagged_chain, run:{imgt_start,imgt_end,delta,loc}, deposit, method, res}]"""
import sys, os, json, gzip, urllib.request, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from anarci import anarci
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CIF=f'{SP}/sabr/dl'; os.makedirs(CIF,exist_ok=True)
def dl(pdb):
    fn=f'{CIF}/{pdb}.cif'
    if os.path.exists(fn): return fn
    data=gzip.decompress(urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb.upper()}.cif.gz',timeout=120).read())
    open(fn,'wb').write(data); return fn
def entity_seqs(md):
    strand=md['_pdbx_poly_seq_scheme.pdb_strand_id']; ent=md['_pdbx_poly_seq_scheme.entity_id']
    c2e=dict(zip(strand,ent))
    ids=md['_entity_poly.entity_id']; seqs=md['_entity_poly.pdbx_seq_one_letter_code_can']
    if isinstance(ids,str): ids=[ids]; seqs=[seqs]
    e2s={i:s.replace('\n','').replace(';','').strip() for i,s in zip(ids,seqs)}
    return {c:e2s.get(e,'') for c,e in c2e.items()}
def vdom(seq):
    """-> list of (chain_type, start, end) V domains by ANARCI (IMGT)"""
    num,ali,_=anarci([('q',seq)],scheme='imgt',assign_germline=False)
    if not num[0]: return []
    return [(a['chain_type'],a['query_start'],a['query_end']) for a in ali[0]]
CDR=lambda p: 27<=p<=38 or 56<=p<=65 or 105<=p<=117
def fr_insertion(vseq):
    """True if ANARCI places an insertion code in a framework position (e.g. 4A). SAbR and ANARCI place framework
    insertions differently, which mimics a register shift (7CWU heavy chain: EVQLLEQ... vs germline QVQL-VQ...)."""
    num,_,_=anarci([('q',vseq)],scheme='imgt',assign_germline=False)
    return any(i.strip() and not CDR(p) for (p,i),a in num[0][0][0] if a!='-')
def build(pdb, flagged_chain, ctype):
    fn=dl(pdb); md=MMCIF2Dict(fn); st=MMCIFParser(QUIET=True).get_structure(pdb,fn)[0]
    seqs=entity_seqs(md)
    want='KL' if ctype=='H' else 'H'
    me=seqs[flagged_chain]; d=[x for x in vdom(me) if x[0]==ctype or (ctype in 'KL' and x[0] in 'KL')]
    if len(vdom(me))!=1: return None  # scFv / multi-domain: skip in pilot
    if fr_insertion(me[d[0][1]:d[0][2]+1]): return None
    def cen(c):
        try: return np.mean([r['CA'].coord for r in st[c] if 'CA' in r][:110],axis=0)
        except Exception: return None
    c0=cen(flagged_chain); best=None
    for c,s in seqs.items():
        if c==flagged_chain or c not in st: continue
        v=vdom(s) if s else []
        if len(v)!=1 or v[0][0] not in want: continue
        cc=cen(c)
        if cc is None: continue
        dist=np.linalg.norm(cc-c0)
        if best is None or dist<best[0]: best=(dist,c,s,v[0])
    if best is None or best[0]>40: return None
    a=d[0]; b=best[3]
    pair={ctype if ctype=='H' else 'L': (flagged_chain, me[a[1]:a[2]+1]), ('L' if ctype=='H' else 'H'): (best[1], best[2][b[1]:b[2]+1])}
    return dict(H=pair['H'][0],L=pair['L'][0],Hseq=pair['H'][1],Lseq=pair['L'][1],pair_dist=round(float(best[0]),1))
if __name__=='__main__':
    import random
    R='/home/user/SAbR/register_error_analysis/sabr_anarci_scan/tables'
    r=pd.read_csv(f'{R}/disagreement_runs.csv'); m=pd.read_csv(f'{R}/mapfit_detections.csv')
    m=m[m.set=='detected'].rename(columns={'imgt0':'imgt_start','imgt1':'imgt_end'})
    r=r.merge(m[['pdb','chain','imgt_start','imgt_end','kind','pct']],on=['pdb','chain','imgt_start','imgt_end'],how='left')
    n=int(sys.argv[1]) if len(sys.argv)>1 else 12
    # pilot: framework shifts with |delta|<=3, length>=4, spread over deposition strata
    r=r[(r.start_region=='FR')&(r.delta.abs()<=3)&(r.length>=4)&(r.length<=30)&(r.kind.fillna('framework shift')=='framework shift')]
    r['dep']=pd.to_datetime(r.deposit)
    bins=pd.to_datetime(['1990-01-01','2018-04-30','2020-05-01','2021-07-31','2023-06-01','2027-01-01'])
    r['stratum']=pd.cut(r.dep,bins,labels=['s1 <2018-05','s2 2018-05..2020-04','s3 2020-05..2021-07','s4 2021-08..2023-05','s5 >=2023-06']).astype(str)
    r=r.sort_values('pct').drop_duplicates(['pdb'])
    out=[]; rng=random.Random(1)
    for s,g in r.groupby('stratum'):
        g=g.sample(frac=1,random_state=1); k=0
        for x in g.itertuples():
            if k>=n: break
            try: b=build(x.pdb,x.chain,x.chain_type)
            except Exception as e: print('fail',x.pdb,e); continue
            if not b or len(b['Hseq'])>135 or len(b['Lseq'])>135: continue
            out.append(dict(id=f'{x.pdb}_{b["H"]}{b["L"]}',pdb=x.pdb,flagged_chain=x.chain,chain_type=x.chain_type,stratum=s,
                            run=dict(imgt_start=int(x.imgt_start),imgt_end=int(x.imgt_end),delta=int(x.delta),loc=x.loc),
                            deposit=x.deposit,method=x.method,res=x.res,mapfit_pct=None if pd.isna(x.pct) else round(x.pct,1),**b)); k+=1
            print(s,out[-1]['id'],x.loc,x.delta,flush=True)
    json.dump(out,open(f'{SP}/pred/cases/cases_err.json','w'),indent=1)
    print(len(out))
