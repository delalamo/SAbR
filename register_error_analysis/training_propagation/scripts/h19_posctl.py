"""Positive control: write the deposited V-region coordinates of each case as a 'prediction' (chains H, L)."""
import sys, json, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from h19_readout import *
from Bio.PDB import PDBIO, Select
cases=json.load(open(sys.argv[1])); out=sys.argv[2]; os.makedirs(out,exist_ok=True)
for c in cases:
    fn=f'{SP}/sabr/dl/{c["pdb"]}.cif'; st=MMCIFParser(QUIET=True).get_structure('d',fn)
    md=MMCIF2Dict(fn); strand=md['_pdbx_poly_seq_scheme.pdb_strand_id']; mon=md['_pdbx_poly_seq_scheme.mon_id']
    num=md['_pdbx_poly_seq_scheme.pdb_seq_num']; ins=md['_pdbx_poly_seq_scheme.pdb_ins_code']
    from Bio.PDB.Structure import Structure; from Bio.PDB.Model import Model; from Bio.PDB.Chain import Chain
    S0=Structure('x'); M=Model(0); S0.add(M)
    for new,old,vseq in (('H',c['H'],c['Hseq']),('L',c['L'],c['Lseq'])):
        full=''.join(seq1(m) for s,m in zip(strand,mon) if s==old); off=full.find(vseq)
        keys=[(int(n),'' if i in '.?' else i) for s,n,i in zip(strand,num,ins) if s==old][off:off+len(vseq)]
        ch=Chain(new); M.add(ch); ks=set(keys)
        for r in st[0][old]:
            if (r.id[1],r.id[2].strip()) in ks and r.id[0]==' ':
                r2=r.copy(); r2.detach_parent(); ch.add(r2)
    # predictions are full-length; deposited chains have gaps -> readout tolerates (offset by first resolved residue)
    io=PDBIO(); io.set_structure(S0); io.save(f'{out}/{c["id"]}.pdb')
print('ok')
