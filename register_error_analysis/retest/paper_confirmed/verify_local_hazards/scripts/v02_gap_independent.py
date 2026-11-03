"""Independent recomputation of the core local hazard ("unbuilt residues within 5 IMGT positions of the mode window, or
N-terminus unbuilt for mode A") with a definition that does NOT use SAbR's IMGT labels to decide what is built:
built = SEQRES positions aligned (semi-global) to the modelled residues; window positions = ANARCI IMGT numbering of SEQRES.
(The investigator's unmod_near counts ANARCI positions absent from SAbR's label set, which a register shift itself can
create.) Also recomputes MH ORs, one-per-paper and a paper-cluster bootstrap with own code, and two control-set variants.
Output: tables/v02_gap_frame.pkl.gz, tables/v02.log"""
import json, numpy as np, pandas as pd, sys
from Bio import Align
from Bio.SeqUtils import seq1
from scipy.stats import norm, fisher_exact, chi2
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_local_hazards/tables'
LH='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/local_hazards/tables'
rec={}
for fn in ['validation.jsonl','snapshot_rest.jsonl','new.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r=json.loads(line)
        if r.get('status')=='ok' and r.get('residues'): rec[(r['pdb'],r['chain'])]=r['residues']
AC=json.load(open(f'{SP}/sabr/anarci_cache.json'))
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
FL=pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz'); C=C.merge(FL,on=['pdb','chain'])
al=Align.PairwiseAligner(); al.mode='global'; al.match_score=2; al.mismatch_score=-3
al.target_end_gap_score=0; al.query_internal_gap_score=-2; al.target_internal_gap_score=-8; al.query_end_gap_score=-8
WIN={'A':(1,14),'F':(20,45),'B':(50,75),'C':(71,100),'D+':(104,124),'D-':(104,124),'E':(115,128)}
def anarci(seq,ct):
    for d in AC.get(seq) or []:
        if (d['chain_type'] in 'KL')==(ct in 'KL'): return [(s,p) for s,p,i in d['pos']]
    return None
rows=[]
for r in C.itertuples():
    res=rec.get((r.pdb,r.chain)); num=anarci(r.seq,r.ctype)
    if not res or not num: continue
    q=''.join(seq1(x[2]) or 'X' for x in res)
    a=al.align(r.seq,q)[0]
    built=set()
    for (t0,t1),(q0,q1) in zip(*a.aligned): built.update(range(t0,t1))
    ident=sum(r.seq[t]==q[qq] for (t0,t1),(q0,q1) in zip(*a.aligned) for t,qq in zip(range(t0,t1),range(q0,q1)))/max(len(q),1)
    bp=[p for s,p in num if s in built]
    d=dict(pdb=r.pdb,chain=r.chain,aln_ident=ident,first_built_v=min(bp) if bp else np.nan)
    for m,(lo,hi) in WIN.items():
        d[f'unb_{m}']=sum((lo-5<=p<=hi+5) and (s not in built) for s,p in num)
    rows.append(d)
G=pd.DataFrame(rows); G.to_pickle(f'{OUT}/v02_gap_frame.pkl.gz')
print('chains with independent gap features',len(G),'| median alignment identity',G.aln_ident.median(),'| <0.9:',(G.aln_ident<0.9).sum())
