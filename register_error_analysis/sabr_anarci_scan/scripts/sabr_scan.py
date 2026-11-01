"""For each entry: download mmCIF, run SAbR (structure-based) and ANARCI (sequence-based) IMGT numbering
on antibody V domains, and record per-residue labels."""
import sys, os, gzip, json, time, urllib.request, warnings, traceback
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from sabr import renumber_structure
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/sabr'
def dl(pdb):
    fn=f'{SP}/dl/{pdb}.cif'
    for url in (f'https://files.rcsb.org/download/{pdb.upper()}.cif.gz',f'https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}.cif'):
        for k in range(3):
            try:
                data=urllib.request.urlopen(url,timeout=120).read()
                if url.endswith('.gz'): data=gzip.decompress(data)
                open(fn,'wb').write(data); return fn
            except Exception as e:
                if '404' in str(e): break
                time.sleep(1+2**k)
    return None
def chain_seq_map(md,chain):
    """author (seq_num, icode) -> (entity seq index 0-based); and entity sequence."""
    strand=md['_pdbx_poly_seq_scheme.pdb_strand_id']; seqid=md['_pdbx_poly_seq_scheme.seq_id']
    num=md['_pdbx_poly_seq_scheme.pdb_seq_num']; ins=md['_pdbx_poly_seq_scheme.pdb_ins_code']; ent=md['_pdbx_poly_seq_scheme.entity_id']
    m={}; eid=None
    for s,i,n,c,e in zip(strand,seqid,num,ins,ent):
        if s==chain:
            m[(int(n),'' if c in ('.','?') else c)]=int(i)-1; eid=e
    ids=md['_entity_poly.entity_id']; seqs=md['_entity_poly.pdbx_seq_one_letter_code_can']
    if isinstance(ids,str): ids=[ids]; seqs=[seqs]
    seq=dict(zip(ids,seqs)).get(eid,'').replace('\n','').replace(';','').strip()
    return m,seq
def process(job):
    pdb,chains,anum=job  # anum: chain -> list of domains with pos [[seqidx,num,ins],...]
    fn=dl(pdb)
    if not fn: return [{'pdb':pdb,'chain':c,'status':'download failed'} for c in chains]
    out=[]
    try:
        st=MMCIFParser(QUIET=True).get_structure(pdb,fn); md=MMCIF2Dict(fn)
    except Exception as e:
        os.remove(fn); return [{'pdb':pdb,'chain':c,'status':f'parse failed: {e}'} for c in chains]
    for c in chains:
        rec={'pdb':pdb,'chain':c}
        try:
            doms=anum[c]
            if len(doms)!=1: rec['status']='multi-domain chain skipped'; out.append(rec); continue
            ct=doms[0]['chain_type']; ct={'H':'H','K':'K','L':'L'}.get(ct)
            if ct is None: rec['status']='not antibody'; out.append(rec); continue
            m,seq=chain_seq_map(md,c)
            a_by_idx={p[0]:(p[1],p[2]) for p in doms[0]['pos']}
            t0=time.time(); new=renumber_structure(st,chain=c,mmcif_dict=md); rec['sabr_s']=round(time.time()-t0,2)
            old=list(st[0][c]); nw=list(new[0][c])
            res=[]
            for o,n in zip(old,nw):
                if o.id[0]!=' ': continue
                key=(o.id[1],o.id[2].strip()); si=m.get(key)
                a=a_by_idx.get(si) if si is not None else None
                res.append([o.id[1],o.id[2].strip(),o.get_resname(),si,a[0] if a else None,a[1] if a else None,n.id[1],n.id[2].strip()])
            rec.update(status='ok',chain_type=ct,residues=res)
        except Exception as e:
            rec['status']=f'error: {type(e).__name__}: {str(e)[:150]}'
        out.append(rec)
    os.remove(fn)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    jobs=json.load(open(sys.argv[1])); outfn=sys.argv[2]; nproc=int(sys.argv[3]) if len(sys.argv)>3 else 4
    done=set()
    if os.path.exists(outfn):
        for l in open(outfn): done.add(json.loads(l)['pdb'])
    jobs=[j for j in jobs if j[0] not in done]
    t0=time.time()
    with Pool(nproc) as p, open(outfn,'a') as f:
        for i,recs in enumerate(p.imap_unordered(process,jobs)):
            for r in recs: f.write(json.dumps(r)+'\n')
            f.flush()
            if i%50==0: print(i,len(jobs),'%.0fs'%(time.time()-t0),flush=True)
    print('done',flush=True)
