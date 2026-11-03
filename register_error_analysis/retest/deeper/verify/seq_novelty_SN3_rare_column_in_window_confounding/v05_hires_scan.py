"""Run the scan's procedure (SAbR on the whole chain with mmcif_dict; ANARCI IMGT on SEQRES matched by seq index; runs of
>=3 consecutive disagreeing residues) on high-resolution X-ray chains that carry the same rare-column insertions as the
SN3 positives. A run touching the rare column in a 1.1-2.6 A structure means the run is produced by the numbering tools
(insertion placed in different columns), i.e. it would be flagged as an "error" whatever the map quality."""
import os, sys, gzip, json, urllib.request, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from anarci import run_anarci
from sabr import renumber_structure
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA=f'{SP}/deeper/cache/verify_SN3_confounding/cif'; os.makedirs(CA,exist_ok=True)
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
S=pd.read_csv(f'{OUT}/'+(sys.argv[3] if len(sys.argv)>3 else 'v04_hires_selection.csv'))
part=int(sys.argv[1]) if len(sys.argv)>1 else 0; nparts=int(sys.argv[2]) if len(sys.argv)>2 else 1
S=S.iloc[part::nparts]
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
for _,j in S.iterrows():
    p,c=j.entry,j.chain
    try:
        f=dl(p); st=MMCIFParser(QUIET=True).get_structure(p,f); md=MMCIF2Dict(f)
        m,seq=seqmap(md,c)
        an=run_anarci([('q',seq)],scheme='imgt',assign_germline=False)
        doms=an[1][0]
        # domain matching the V domain of interest
        dom=None
        for d in doms:
            vs=''.join(aa for (pp,ii),aa in d[0] if aa!='-')
            if j.vdom[5:40] in vs: dom=d
        if dom is None: dom=doms[0]
        num=[(pp,ii.strip()) for (pp,ii),aa in dom[0] if aa!='-']; s0=dom[1]
        a_by_idx={s0+k:x for k,x in enumerate(num)}
        new=renumber_structure(st,chain=c,mmcif_dict=md)
        old=list(st[0][c]); nw=list(new[0][c]); dis=[]
        for o,n in zip(old,nw):
            if o.id[0]!=' ': continue
            si=m.get((o.id[1],o.id[2].strip())); a=a_by_idx.get(si)
            if a is None: continue
            s=(n.id[1],n.id[2].strip())
            dis.append((a,s,a!=s,seq[si]))
        runs=[]; cur=[]
        for a,s,d,aa in dis:
            if d: cur.append((a,s,aa))
            else:
                if len(cur)>=3: runs.append(cur)
                cur=[]
        if len(cur)>=3: runs.append(cur)
        rare=set(str(j.rare_cols).split())
        def touches(r):
            lo=r[0][0][0]-2; hi=r[-1][0][0]+2
            return any(lo<=int(''.join(ch for ch in x if ch.isdigit()))<=hi for x in rare)
        desc='; '.join(f'{r[0][0][0]}{r[0][0][1]}-{r[-1][0][0]}{r[-1][0][1]} (n={len(r)}; SAbR {r[0][1][0]}{r[0][1][1]}..{r[-1][1][0]}{r[-1][1][1]})' for r in runs)
        rows.append(dict(group=j.group,pdb=p,chain=c,res=j.res,ctype=j.ctype,species=j.species,rare_cols=j.rare_cols,
                         n_disagree=sum(d for _,_,d,_ in dis),n_runs=len(runs),run_touches_rare=any(touches(r) for r in runs),runs=desc))
    except Exception as ex:
        rows.append(dict(group=j.group,pdb=p,chain=c,res=j.res,note=f'{type(ex).__name__}: {str(ex)[:120]}'))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/v05_hires_scan_part{part}'+(sys.argv[4] if len(sys.argv)>4 else '')+'.csv',index=False)
