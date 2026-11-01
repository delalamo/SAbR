import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[_v]="1"
FAST=os.environ.get("SABR_FAST","1")=="1"
"""Light SAbR-vs-ANARCI scan: per-chain coordinates from PDBe ModelServer; unobserved residues and
author->sequence mapping from RCSB GraphQL (auth_to_entity_poly_seq_mapping)."""
import sys, os, io, re, json, time, urllib.request, warnings
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)
from Bio.PDB import MMCIFParser
from sabr import renumber_structure
A3={'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
def fetch(url,timeout=120):
    for k in range(4):
        try: return urllib.request.urlopen(url,timeout=timeout).read().decode()
        except Exception as e:
            if '404' in str(e) or '400' in str(e): return None
            time.sleep(1+2**k)
    return None
def gql_maps(pdb):
    Q='{entry(entry_id:"%s"){polymer_entities{entity_poly{pdbx_seq_one_letter_code_can} polymer_entity_instances{rcsb_polymer_entity_instance_container_identifiers{auth_asym_id auth_to_entity_poly_seq_mapping}}}}}'%pdb.upper()
    for k in range(4):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q}).encode(),{'Content-Type':'application/json'})
            d=json.load(urllib.request.urlopen(r,timeout=120))['data']['entry']; out={}
            for pe in d['polymer_entities']:
                s=(pe['entity_poly']['pdbx_seq_one_letter_code_can'] or '').replace('\n','')
                for inst in pe['polymer_entity_instances']:
                    c=inst['rcsb_polymer_entity_instance_container_identifiers']; out[c['auth_asym_id']]=(c['auth_to_entity_poly_seq_mapping'],s)
            return out
        except Exception: time.sleep(1+2**k)
    return None
def split_label(lab):
    m=re.match(r'^(-?\d+)([A-Za-z]*)$',lab or '')
    return (int(m.group(1)),m.group(2)) if m else None
def process(job):
    pdb,chains,anum=job; out=[]
    maps=gql_maps(pdb)
    if maps is None: return [{'pdb':pdb,'chain':c,'status':'graphql failed'} for c in chains]
    for c in chains:
        rec={'pdb':pdb,'chain':c}
        try:
            doms=anum[c]
            types=[d['chain_type'] for d in doms]
            if not doms or len(doms)!=1 or types[0] not in ('A','B','G','D'): rec['status']='not single-domain TCR'; out.append(rec); continue
            if any(not d['pos'] for d in doms): rec['status']='ANARCI domain without numbered positions'; out.append(rec); continue
            ct=''.join(types)
            if len(doms)==2 and sorted(types) in (['H','K'],['H','L']): kw=dict(scfv=True); mode='scfv'
            elif len(doms)==1: kw=dict(chain_type=ct); mode='single'
            else: kw=dict(chain_type=ct); mode='multi:'+ct
            if c not in maps: rec['status']='chain not in graphql'; out.append(rec); continue
            amap,seq=maps[c]
            txt=fetch(f'https://www.ebi.ac.uk/pdbe/model-server/v1/{pdb}/atoms?auth_asym_id={c}&encoding=cif') or fetch(f'https://models.rcsb.org/v1/{pdb}/atoms?auth_asym_id={c}&encoding=cif')
            if not txt: rec['status']='coords failed'; out.append(rec); continue
            st=MMCIFParser(QUIET=True).get_structure(pdb,io.StringIO(txt))
            model=next(iter(st)); chobj=model[c]
            observed={(r.id[1],r.id[2].strip()) for r in chobj if r.id[0]==' '}
            lab2idx={}; unobs=[]
            for i,lab in enumerate(amap):
                k=split_label(lab)
                if k is None: continue
                lab2idx[k]=i
                if k not in observed: unobs.append((k,seq[i] if i<len(seq) else 'X'))
            md={'_pdbx_unobs_or_zero_occ_residues.auth_asym_id':[c]*len(unobs),
                '_pdbx_unobs_or_zero_occ_residues.auth_seq_id':[str(k[0]) for k,_ in unobs],
                '_pdbx_unobs_or_zero_occ_residues.PDB_ins_code':[k[1] or '?' for k,_ in unobs],
                '_pdbx_unobs_or_zero_occ_residues.auth_comp_id':[A3.get(a,'UNK') for _,a in unobs],
                '_pdbx_unobs_or_zero_occ_residues.PDB_model_num':[str(model.serial_num)]*len(unobs)}
            a_by_idx={}
            for di,d in enumerate(doms):
                for q in d['pos']: a_by_idx[q[0]]=(q[1],q[2],di)
            vobs=sum(1 for k in observed if lab2idx.get(k) in a_by_idx)
            if len(doms)>=2:
                obs_idx={lab2idx.get(k) for k in observed}
                keep=[di for di,d in enumerate(doms) if sum(1 for q in d['pos'] if q[0] in obs_idx)>=20]
                if len(keep)==1 and len(doms)>1:
                    di=keep[0]; doms_keep=doms[di]
                    a_by_idx={q[0]:(q[1],q[2],0) for q in doms_keep['pos']}
                    kw=dict(chain_type=doms_keep['chain_type']); mode='single domain of '+ct; ct=doms_keep['chain_type']
            if vobs<20: rec['status']='V region not modelled'; rec['v_observed']=vobs; out.append(rec); continue
            vidx=sorted(a_by_idx); labs=[split_label(amap[i]) for i in range(vidx[0],vidx[-1]+1) if i<len(amap)]
            nums=[l[0] for l in labs if l]
            if FAST and nums and nums==sorted(nums):
                lo,hi=min(nums)-3,max(nums)+3
                from Bio.PDB.Structure import Structure
                from Bio.PDB.Model import Model
                tst=Structure(pdb); tst.header=dict(getattr(st,'header',{}) or {}); tm=Model(model.id,model.serial_num); tst.add(tm); tc=chobj.copy()
                for r in list(tc):
                    if not (lo<=r.id[1]<=hi): tc.detach_child(r.id)
                tm.add(tc); st=tst; chobj=tc
            t0=time.time(); new=renumber_structure(st,chain=c,mmcif_dict=md,**kw); rec['sabr_s']=round(time.time()-t0,2); rec['mode']=mode
            res=[]
            for o,n in zip(list(chobj),list(next(iter(new))[c])):
                if o.id[0]!=' ': continue
                si=lab2idx.get((o.id[1],o.id[2].strip())); a=a_by_idx.get(si) if si is not None else None
                snum=n.id[1]-1000*a[2] if a else n.id[1]
                res.append([o.id[1],o.id[2].strip(),o.get_resname(),si,a[0] if a else None,a[1] if a else None,snum,n.id[2].strip(),a[2] if a else None])
            rec.update(status='ok',chain_type=ct,residues=res)
        except Exception as e:
            rec['status']=f'error: {type(e).__name__}: {str(e)[:150]}'
        out.append(rec)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    jobs=json.load(open(sys.argv[1])); outfn=sys.argv[2]; nproc=int(sys.argv[3]) if len(sys.argv)>3 else 6
    done=set()
    if os.path.exists(outfn):
        for l in open(outfn): done.add(json.loads(l)['pdb'])
    jobs=[j for j in jobs if j[0] not in done]; t0=time.time()
    with Pool(nproc) as p, open(outfn,'a') as f:
        for i,recs in enumerate(p.imap_unordered(process,jobs)):
            for r in recs: f.write(json.dumps(r)+'\n')
            f.flush()
            if i%100==0: print(i,len(jobs),'%.0fs'%(time.time()-t0),flush=True)
    print('done',flush=True)
