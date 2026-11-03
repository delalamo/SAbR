# Is the SAbR/ANARCI disagreement in the bovine (FMDV, Lou Z./Li K.) capsid entries a detector artifact?
# Run the scan's exact procedure (SAbR on the whole chain with mmcif_dict; ANARCI IMGT on SEQRES, matched by seq index)
# on high-resolution (<=2.6 A) X-ray bovine Fab heavy chains, with ultralong and with conventional CDR-H3.
# A >=3-residue disagreement run around CDR3/FR4 in these well-determined structures = artifact of the numbering tools.
import os, gzip, json, urllib.request, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from anarci import run_anarci
from sabr import renumber_structure
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA=f'{SP}/deeper/cache/verify_capsid_confounding/cif'; os.makedirs(CA,exist_ok=True)
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_confounding'
JOBS=[('4k3d','H'),('4k3e','H'),('6e9h','A'),('6oo0','H'),('8ecq','H'),('8ecv','H'),('5ihu','H'),('5ilt','H'),('5ijv','H'),
      ('8vbj','H'),('8v4i','H'),('6qn7','H'),('6qn8','H'),('9hv2','H'),('9hv9','H'),
      ('8vbj','L'),('6qn8','L'),('5ihu','L')]   # bovine lambda light chains as a control
def dl(p):
    f=f'{CA}/{p}.cif'
    if not os.path.exists(f):
        data=gzip.decompress(urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif.gz',timeout=120).read()); open(f,'wb').write(data)
    return f
def seqmap(md,chain):
    strand=md['_pdbx_poly_seq_scheme.pdb_strand_id']; seqid=md['_pdbx_poly_seq_scheme.seq_id']; num=md['_pdbx_poly_seq_scheme.pdb_seq_num']
    ins=md['_pdbx_poly_seq_scheme.pdb_ins_code']; ent=md['_pdbx_poly_seq_scheme.entity_id']; m={}; eid=None
    for s,i,n,c,e in zip(strand,seqid,num,ins,ent):
        if s==chain: m[(int(n),'' if c in ('.','?') else c)]=int(i)-1; eid=e
    ids=md['_entity_poly.entity_id']; seqs=md['_entity_poly.pdbx_seq_one_letter_code_can']
    if isinstance(ids,str): ids=[ids]; seqs=[seqs]
    return m,dict(zip(ids,seqs)).get(eid,'').replace('\n','').replace(';','').strip()
rows=[]
for p,c in JOBS:
    try:
        f=dl(p); st=MMCIFParser(QUIET=True).get_structure(p,f); md=MMCIF2Dict(f)
        m,seq=seqmap(md,c)
        an=run_anarci([('q',seq)],scheme='imgt',assign_germline=False)
        dom=an[1][0][0]; num=[(pp,ii.strip()) for (pp,ii),aa in dom[0] if aa!='-']; s0=dom[1]
        a_by_idx={s0+k:x for k,x in enumerate(num)}
        L3=sum(1 for (pp,ii) in num if 105<=pp<=117)
        new=renumber_structure(st,chain=c,mmcif_dict=md)
        old=list(st[0][c]); nw=list(new[0][c]); dis=[]
        for o,n in zip(old,nw):
            if o.id[0]!=' ': continue
            si=m.get((o.id[1],o.id[2].strip())); a=a_by_idx.get(si)
            if a is None: continue
            s=(n.id[1],n.id[2].strip())
            dis.append((a,s,a!=s,seq[si]))
        # runs of >=3 consecutive disagreements
        runs=[]; cur=[]
        for a,s,d,aa in dis:
            if d: cur.append((a,s,aa))
            else:
                if len(cur)>=3: runs.append(cur)
                cur=[]
        if len(cur)>=3: runs.append(cur)
        desc='; '.join(f'{r[0][0][0]}{r[0][0][1]}-{r[-1][0][0]}{r[-1][0][1]} (n={len(r)}; SAbR {r[0][1][0]}{r[0][1][1]}..{r[-1][1][0]}{r[-1][1][1]})' for r in runs)
        rows.append(dict(pdb=p,chain=c,cdr3_len_imgt=L3,n_disagree=sum(d for _,_,d,_ in dis),n_runs=len(runs),runs=desc,
                         cdr3_run=any(any(105<=x[0][0]<=128 for x in r) for r in runs)))
    except Exception as ex:
        rows.append(dict(pdb=p,chain=c,note=f'{type(ex).__name__}: {str(ex)[:120]}'))
    print(rows[-1],flush=True)
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/bovine_hires_check.csv',index=False)
print(R.to_string())
