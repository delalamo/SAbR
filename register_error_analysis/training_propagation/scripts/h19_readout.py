"""Does a predicted Fv reproduce the register error of the deposited chain?
Two independent readouts, both reference-free (no corrected structure needed):
 1. detector: SAbR (structure) vs ANARCI (sequence) IMGT labels on the predicted chain; a run of >=3
    disagreeing residues overlapping the deposited error window = the prediction carries a register shift there.
 2. ladder: superpose predicted chain on the deposited chain (framework CA outside the window and the CDRs),
    then for each deposited residue in the window find the predicted residue whose CA is within 3 A.
    shift 0 = the prediction puts the same residue at the same place as the deposited model (copies the error);
    shift = +delta = canonical register (what SAbR says the residue should be).
usage: h19_readout.py cases.json outdir method [method...]  -> prints table, writes readout_<method>.csv"""
import sys, os, json, numpy as np, pandas as pd, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1
from anarci import anarci
from sabr import renumber_structure
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
def anarci_labels(seq):
    num,_,_=anarci([('q',seq)],scheme='imgt',assign_germline=False)
    lab=[]; 
    for (pos,ins),aa in num[0][0][0]:
        if aa!='-': lab.append((pos,ins.strip()))
    st=num[0][0][1]; return lab, st
def kabsch(P,Q):
    pc,qc=P.mean(0),Q.mean(0); H=(P-pc).T@(Q-qc); U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,d])@U.T; return R,pc,qc
def pred_chain(fn,ctype):
    st=PDBParser(QUIET=True).get_structure('p',fn)[0]
    return st, ('H' if ctype=='H' else 'L')
from Bio import Align
_al=Align.PairwiseAligner(mode='global',match_score=2,mismatch_score=-3,open_gap_score=-5,extend_gap_score=-0.5,end_gap_score=0)
def pred_map(st,chain,seq):
    """index in the V sequence for each residue of the model chain (handles trimmed ends and unmodelled gaps)"""
    res=[r for r in st[chain] if r.id[0]==' ']; s=''.join(seq1(r.get_resname()) for r in res)
    off=seq.find(s)
    if off>=0: return list(range(off,off+len(s)))
    a=_al.align(s,seq)[0]; m=[None]*len(s)
    for (a0,a1),(b0,b1) in zip(*a.aligned):
        for k in range(a1-a0): m[a0+k]=b0+k
    return m
def detector_runs(st,chain,ct,seq):
    """runs of >=3 residues with SAbR label != ANARCI label; returns list of (imgt_start, imgt_end, delta)"""
    lab0,_=anarci_labels(seq); pm=pred_map(st,chain,seq); lab0=lab0+[(999,"")]*(len(seq)-len(lab0))
    new=renumber_structure(st.get_parent() if hasattr(st,'get_parent') else st,chain=chain,chain_type=ct)
    nres=[r for r in new[0][chain] if r.id[0]==' ']
    sab=[(r.id[1],r.id[2].strip()) for r in nres]; lab=[lab0[i] if i is not None else (-99,'') for i in pm]
    diff=[a!=b for a,b in zip(lab,sab)]
    runs=[]; i=0
    while i<len(diff):
        if diff[i]:
            j=i
            while j+1<len(diff) and diff[j+1]: j+=1
            if j-i+1>=3:
                d=[sab[k][0]-lab[k][0] for k in range(i,j+1)]
                runs.append((lab[i][0],lab[j][0],int(np.median(d))))
            i=j+1
        else: i+=1
    return runs
DEP={}
def deposited(pdb,chain,vseq):
    """deposited chain residues -> (V index, CA). Uses pdbx_poly_seq_scheme to map to SEQRES."""
    fn=f'{SP}/sabr/dl/{pdb}.cif'; md=MMCIF2Dict(fn); st=MMCIFParser(QUIET=True).get_structure(pdb,fn)[0]
    strand=md['_pdbx_poly_seq_scheme.pdb_strand_id']; seqid=md['_pdbx_poly_seq_scheme.seq_id']
    num=md['_pdbx_poly_seq_scheme.pdb_seq_num']; ins=md['_pdbx_poly_seq_scheme.pdb_ins_code']; mon=md['_pdbx_poly_seq_scheme.mon_id']
    full=''.join(seq1(m) for s,m in zip(strand,mon) if s==chain)
    off=full.find(vseq); assert off>=0
    m={(int(n),'' if c in '.?' else c):int(i)-1-off for s,i,n,c in zip(strand,seqid,num,ins) if s==chain}
    out={}
    for r in st[chain]:
        k=m.get((r.id[1],r.id[2].strip()))
        if k is not None and 0<=k<len(vseq) and 'CA' in r: out[k]=r['CA'].coord
    return out
def ladder(dep,pred_ca,lab,lo,hi):
    """dep, pred_ca: V index -> CA. lab: ANARCI labels per V index."""
    cdr=lambda p: 27<=p<=38 or 56<=p<=65 or 105<=p<=117
    dep={k:v for k,v in dep.items() if k<len(lab)}
    use=[k for k in dep if k in pred_ca and not cdr(lab[k][0]) and not (lo-2<=lab[k][0]<=hi+2)]
    P=np.array([pred_ca[k] for k in use]); Q=np.array([dep[k] for k in use])
    R,pc,qc=kabsch(P,Q); pt={k:(v-pc)@R.T+qc for k,v in pred_ca.items()}
    rms=float(np.sqrt(((np.array([pt[k] for k in use])-Q)**2).sum(1).mean()))
    keys=list(pt); arr=np.array([pt[k] for k in keys]); shifts=[]
    for k in sorted(dep):
        if not (lo<=lab[k][0]<=hi): continue
        d=np.linalg.norm(arr-dep[k],axis=1); j=d.argmin()
        shifts.append(keys[j]-k if d[j]<=3.0 else None)
    return shifts,rms
if __name__=='__main__':
    cases=json.load(open(sys.argv[1])); outd=sys.argv[2]
    for meth in sys.argv[3:]:
        rows=[]
        for c in cases:
            fn=f"{outd}/{meth}/{c['id']}.pdb"
            if not os.path.exists(fn): continue
            ct=c['chain_type']; vseq=c['Hseq'] if ct=='H' else c['Lseq']; lo,hi,delta=c['run']['imgt_start'],c['run']['imgt_end'],c['run']['delta']
            st=PDBParser(QUIET=True).get_structure('p',fn); pc='H' if ct=='H' else 'L'
            if pc not in st[0]: pc=[ch.id for ch in st[0]][0 if ct=='H' else 1]
            try: runs=detector_runs(st[0],pc,ct,vseq)
            except Exception as e: runs=None; print('detector fail',c['id'],e)
            lab,_=anarci_labels(vseq)
            pm=pred_map(st[0],pc,vseq); pred={pm[i]:r['CA'].coord for i,r in enumerate([r for r in st[0][pc] if r.id[0]==' ']) if 'CA' in r and pm[i] is not None}
            dep=deposited(c['pdb'],c['flagged_chain'],vseq)
            sh,rms=ladder(dep,pred,lab,lo,hi)
            ok=[s for s in sh if s is not None]
            same=sum(s==0 for s in ok); canon=sum(s==delta for s in ok)
            hit=[r for r in (runs or []) if r[0]<=hi and r[1]>=lo]
            moved=sum(s!=0 and np.sign(s)==np.sign(delta) for s in ok); n=len(sh)
            call=('poor superposition' if rms>2.5 else 'copies error' if same>=max(2,0.5*n) else 'canonical' if canon>=max(2,0.5*n) else 'shifted, not copied' if moved>=max(2,0.5*n) else 'unclear')
            rows.append(dict(method=meth,id=c['id'],stratum=c['stratum'],loc=c['run']['loc'],window=f'{lo}-{hi}',delta=delta,
                             det_run_in_window=bool(hit), det_runs=';'.join(f'{a}-{b}:{d:+d}' for a,b,d in (runs or [])),
                             ladder=' '.join('.' if s is None else str(s) for s in sh),fw_rmsd=round(rms,2),ladder_call=call))
        t=pd.DataFrame(rows); t.to_csv(f'{outd}/readout_{meth}.csv',index=False)
        pd.set_option('display.width',250); print(t.to_string(index=False))
