# Per-chain unbuilt-residue features of the V region, from the SAbR per-residue records (read-only inputs).
# gap = jump in SEQRES index between consecutive modelled V residues; nterm missing = first modelled IMGT > 2.
import json, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUTC=f'{SP}/deeper/cache/verify_enzyme_confounding'
rows={}
for f in ['validation.jsonl','snapshot_rest.jsonl','new.jsonl']:
    for l in open(f'{SP}/sabr/out/{f}'):
        d=json.loads(l); k=(d['pdb'],d['chain'])
        if k in rows or d.get('status')!='ok': continue
        R=[r for r in d['residues'] if r[3] is not None and r[3]>=0 and r[4] is not None]
        if len(R)<20: continue
        si=np.array([r[3] for r in R]); im=np.array([r[4] for r in R])
        j=np.diff(si); gi=np.where(j>1)[0]
        gpos=[(int(im[i]),int(im[i+1])) for i in gi]   # IMGT (ANARCI) numbers flanking each internal gap
        def near(lo,hi): return any((a>=lo and a<=hi) or (b>=lo and b<=hi) for a,b in gpos)
        rows[k]=dict(pdb=d['pdb'],chain=d['chain'],n_res=len(R),n_gaps=len(gi),n_missing=int((j[gi]-1).sum()) if len(gi) else 0,
                     first_imgt=int(im[0]),last_imgt=int(im[-1]),
                     gap_fr1=near(1,26),gap_cdr1=near(27,38),gap_cdr2fr3=near(55,75),gap_fr3=near(76,104),gap_cdr3=near(105,117),gap_fr4=near(118,128),
                     gap_list=';'.join(f'{a}-{b}' for a,b in gpos))
G=pd.DataFrame(rows.values())
G['nterm_trunc']=G.first_imgt>2
G['cterm_trunc']=G.last_imgt<124
G['any_gap']=(G.n_gaps>0)|G.nterm_trunc
G.to_csv(f'{OUTC}/chain_gaps.csv.gz',index=False)
print(len(G)); print(G[['n_gaps','nterm_trunc','cterm_trunc','any_gap']].mean())
