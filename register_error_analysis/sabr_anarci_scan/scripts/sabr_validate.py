import json, sys, numpy as np, pandas as pd
sys.path.insert(0,'/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/scripts')
from sabr_eval import segments, region
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
recs=[json.loads(l) for f in ['validation.jsonl','validation_multi.jsonl'] for l in open(f'{SP}/sabr/out/{f}')]
ok={(r['pdb'],r['chain']):r for r in recs if r.get('status')=='ok'}
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx'); t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
t=t[[k in ok for k in zip(t.pdb,t.chain)]]
# keep only flagged errors that touch the V region (residues numbered by ANARCI)
def vfrac(r):
    res=ok[(r.pdb,r.chain)]['residues']; inr=[x for x in res if r.rs<=x[0]<=r.re]
    return np.mean([x[4] is not None for x in inr]) if inr else 0
t['vfrac']=t.apply(vfrac,axis=1); tv=t[t.vfrac>=0.5].copy()
print(f'chains scanned OK: {len(ok)}; Table S1 errors on these chains: {len(t)}; in V region: {len(tv)}')
variants=[('minlen3',dict(minlen=3)),('minlen2',dict(minlen=2)),('minlen4',dict(minlen=4)),('minlen3 FR-only',dict(minlen=3,fr_only=True)),('minlen3 skip CDR3',dict(minlen=3,skip_cdr3=True))]
flagged_chains={(p,c) for p,c in zip(tv.pdb,tv.chain)}
allflag_chains={(p,c) for p,c in zip(t.pdb,t.chain)}
rows=[]
for name,kw in variants:
    segs={k:segments(r['residues'],**kw) for k,r in ok.items()}
    hit=[]; hit50=[]; dirok=[]
    for r in tv.itertuples():
        S=segs[(r.pdb,r.chain)]; flagged=set(range(r.rs,r.re+1))
        ov=[len(flagged & set(range(s['start'],s['end']+1))) for s in S]
        best=max(ov) if ov else 0
        hit.append(best>0); hit50.append(best>=0.5*len(flagged))
        if best>0:
            s=S[int(np.argmax(ov))]
            if s['delta'] is not None: dirok.append(abs(s['delta'])==abs(r.SHIFT))
    neg=[k for k in ok if k not in allflag_chains]
    fp=np.mean([len(segs[k])>0 for k in neg])
    pos_chain=np.mean([len(segs[k])>0 for k in flagged_chains])
    rows.append(dict(variant=name,errors=len(tv),err_sensitivity_any=round(100*np.mean(hit),1),err_sensitivity_50pct=round(100*np.mean(hit50),1),
        shift_size_agrees=round(100*np.mean(dirok),1) if dirok else None,flagged_chains=len(flagged_chains),chain_sensitivity=round(100*pos_chain,1),
        unflagged_chains=len(neg),unflagged_with_segment=round(100*fp,1)))
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.to_string(index=False))
R.to_csv(f'{SP}/sabr/out/validation_metrics.csv',index=False)
# details for default variant
segs={k:segments(r['residues'],minlen=3) for k,r in ok.items()}
print('\nPer-error detail (minlen3):')
for r in tv.sort_values(['pdb','chain']).itertuples():
    S=segs[(r.pdb,r.chain)]; flagged=set(range(r.rs,r.re+1))
    best=max(S,key=lambda s: len(flagged & set(range(s['start'],s['end']+1))),default=None)
    ov=len(flagged & set(range(best['start'],best['end']+1))) if best else 0
    print(f"{r.pdb}:{r.chain} {r.METHOD} {r.RESOLUTION} flagged {r.rs}-{r.re} shift {r.SHIFT:+d} | SAbR/ANARCI segment: {(str(best['start'])+'-'+str(best['end'])+' Δ'+str(best['delta'])+' '+'/'.join(best['regions'])) if best and ov else 'none'}")
