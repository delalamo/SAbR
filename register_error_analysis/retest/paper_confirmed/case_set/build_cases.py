"""Paper-confirmed case set: Table S1 (Sanchez Rodriguez et al. 2024, conkit-validate, snapshot Apr 2022) V-region errors
that are overlapped by a SAbR-vs-ANARCI disagreement run in the AlphaFold-independent scan. Annotates each with run
origin (copy / first occurrence), mode and artifact flags; writes cases + controls for re-testing the deeper-dive trends."""
import json, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'; O=f'{SP}/paperconf'
REC={}; SRC={}
for fn in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
    for l in open(f'{SP}/sabr/out/{fn}'):
        r=json.loads(l)
        if r.get('status')=='ok' and r.get('residues'):
            REC[(r['pdb'],r['chain'])]=r['residues']; SRC.setdefault((r['pdb'],r['chain']),[]).append(r['residues'])
RUNREC={}
for fn in ['validation.jsonl','snapshot_rest.jsonl','new.jsonl']:   # same order as unexpl_build.py (later files win)
    for l in open(f'{SP}/sabr/out/{fn}'):
        r=json.loads(l)
        if r.get('status')=='ok': RUNREC[(r['pdb'],r['chain'])]=r['residues']
def run_resnums(r):
    # runs index the list of IMGT-numbered residues (x[4] not None), as in retest/unexplained/scripts/unexpl_build.py
    v=[x for x in RUNREC[(r.pdb,r.chain)] if x[4] is not None]
    assert v[r.i0][4]==r.imgt0 and v[r.i1][4]==r.imgt1
    return {v[j][0] for j in range(r.i0,r.i1+1)}
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx')
t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz'); ck=set(zip(C.pdb,C.chain))
t=t[[k in ck for k in zip(t.pdb,t.chain)]].copy()
def vfrac(r):
    inr=[x for x in REC[(r.pdb,r.chain)] if r.rs<=x[0]<=r.re]
    return np.mean([x[4] is not None for x in inr]) if inr else 0
t['vfrac']=t.apply(vfrac,axis=1)
print('Table S1 errors on scanned single-domain V chains:',len(t),'| >=50% in V region:',int((t.vfrac>=0.5).sum()))
t=t[t.vfrac>=0.5].copy()
R=pd.read_csv(f'{SP}/deeper/runs_all.csv')
L=pd.read_csv('retest/deeper/synthesis/artifact_ledger_runs.csv')[['pdb','chain','imgt0','imgt1','flag']]
R=R.merge(L,on=['pdb','chain','imgt0','imgt1'],how='left')
R['resnums']=[run_resnums(r) for r in R.itertuples()]
rows=[]
for e in t.itertuples():
    fl=set(range(e.rs,e.re+1)); cand=R[(R.pdb==e.pdb)&(R.chain==e.chain)]
    best=None; bo=0
    for r in cand.itertuples():
        o=len(fl & r.resnums)
        if o>bo: bo,best=o,r
    d=dict(pdb=e.pdb,chain=e.chain,s1_range=e.RESIDUE_RANGE,s1_shift=e.SHIFT,s1_size=e.SIZE,method=e.METHOD,res=e.RESOLUTION,year=e.YEAR,confirmed=bo>0,overlap=bo)
    if best is not None:
        d.update(imgt0=best.imgt0,imgt1=best.imgt1,delta=best.delta,mode=best.mode,cls=best.cls,origin=best.origin,copied_or_artifact=best.copied_or_artifact,ledger_flag=best.flag)
    rows.append(d)
S=pd.DataFrame(rows)
def status(r):
    if not r.confirmed: return 'not confirmed by SAbR/ANARCI'
    if str(r.cls) in ('1a','1b','2'): return 'copy of an earlier model'
    if str(r.cls)=='4': return 'known detector artifact (origin 4)'
    f=str(r.ledger_flag)
    if f in ('indel_gap_artifact_confirmed','DE_chain_end_artifact','amyloid_fibril','not_a_register_shift'): return 'artifact (synthesis ledger, strict)'
    if f=='near_copy': return 'near copy (ledger)'
    if f in ('indel_gap_artifact_probable','DE_not_confirmed_other'): return 'confirmed, loose-ledger doubt'
    return 'confirmed first occurrence'
S['status']=S.apply(status,axis=1)
M=C.set_index(['pdb','chain'])
S['paper']=[M.loc[(p,c),'paper'] for p,c in zip(S.pdb,S.chain)]; S['vseq']=[M.loc[(p,c),'vseq'] for p,c in zip(S.pdb,S.chain)]
S['ctype']=[M.loc[(p,c),'ctype'] for p,c in zip(S.pdb,S.chain)]; S['release']=[M.loc[(p,c),'release'] for p,c in zip(S.pdb,S.chain)]
S.to_csv(f'{O}/paper_errors_annotated.csv',index=False)
print(S.status.value_counts().to_string())
print('release max', S.release.max())
# chains: case flags
conf=S[S.status.isin(['confirmed first occurrence','confirmed, loose-ledger doubt','copy of an earlier model','near copy (ledger)'])]
C['paper_case_strict']=[k in set(zip(conf[conf.status=='confirmed first occurrence'].pdb,conf[conf.status=='confirmed first occurrence'].chain)) for k in zip(C.pdb,C.chain)]
C['paper_case_first']=[k in set(zip(conf[conf.status.isin(['confirmed first occurrence','confirmed, loose-ledger doubt'])].pdb,conf[conf.status.isin(['confirmed first occurrence','confirmed, loose-ledger doubt'])].chain)) for k in zip(C.pdb,C.chain)]
C['paper_case_incl_copies']=[k in set(zip(conf.pdb,conf.chain)) for k in zip(C.pdb,C.chain)]
s1=set(S.pdb)
C['in_paper_window']=C.release.astype(str)<='2022-04-05'
C['paper_flagged_entry']=C.pdb.isin(s1)
C['paper_control']=C.in_paper_window & ~C.paper_flagged_entry & ~C.err_any.astype(bool)
C[['pdb','chain','paper_case_strict','paper_case_first','paper_case_incl_copies','in_paper_window','paper_flagged_entry','paper_control']].to_csv(f'{O}/chain_flags.csv.gz',index=False)
for k in ['paper_case_strict','paper_case_first','paper_case_incl_copies','paper_control']:
    d=C[C[k]]; u=d.drop_duplicates(['paper','vseq'])
    print(k,'chains',len(d),'entries',d.pdb.nunique(),'units(ab x paper)',len(u),'papers',d.paper.nunique(),'| EM chains',int((d.method=="EM").sum()),'X-ray',int((d.method!="EM").sum()))
print('window chains',int(C.in_paper_window.sum()),'flagged-entry chains',int(C.paper_flagged_entry.sum()))
# which of the deeper first-occurrence units are in the paper window but NOT in Table S1 (conkit missed)
F=C[C.err_first.astype(bool)&C.in_paper_window]
print('deeper first-occurrence error chains in window:',len(F),'of which in Table S1-flagged entries:',int(F.paper_flagged_entry.sum()))
