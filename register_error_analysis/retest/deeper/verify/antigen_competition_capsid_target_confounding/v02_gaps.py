# Per-chain unbuilt-residue features of the V region from the SAbR per-residue records (read-only inputs).
# gap = jump in seq_index between consecutive modelled V residues; nterm_missing = first modelled seq_index > 0.
# Also: whether any first-occurrence run of the chain touches a gap (within 3 residues), for the "simpler factor" check.
import json, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
rows={}
for f in ['validation.jsonl','snapshot_rest.jsonl','new.jsonl']:
    for l in open(f'{SP}/sabr/out/{f}'):
        d=json.loads(l); k=(d['pdb'],d['chain'])
        if k in rows or d.get('status')!='ok': continue
        R=[r for r in d['residues'] if r[3] is not None and r[3]>=0]
        if not R: continue
        si=np.array([r[3] for r in R]); imgt=np.array([r[4] if r[4] is not None else -1 for r in R])
        jumps=np.diff(si); gi=np.where(jumps>1)[0]
        # gap positions (IMGT number of residue before gap)
        gpos=[int(imgt[i]) for i in gi]
        rows[k]=dict(pdb=d['pdb'],chain=d['chain'],n_res=len(R),n_gaps=len(gi),n_missing=int((jumps[gi]-1).sum()) if len(gi) else 0,
                     nterm_missing=int(si[0]),first_imgt=int(imgt[0]),last_imgt=int(imgt[-1]),gap_cdr1=any(20<=g<=40 for g in gpos),gap_cdr2=any(50<=g<=70 for g in gpos),
                     gap_cdr3=any(100<=g<=120 for g in gpos),gap_fr3=any(70<g<100 for g in gpos),gap_imgt=';'.join(map(str,gpos)))
G=pd.DataFrame(rows.values())
G['nterm_trunc']=G.first_imgt>2
G['any_gap']=(G.n_gaps>0)|G.nterm_trunc
G.to_csv(f'{SP}/deeper/cache/verify_capsid_confounding/chain_gaps.csv.gz',index=False)
print(len(G)); print(G[['n_gaps','first_imgt','nterm_trunc','any_gap']].describe()); print(G.any_gap.mean(), G.nterm_trunc.mean())
