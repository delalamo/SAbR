"""For each SN3-positive unit-mode, print the per-residue ANARCI vs SAbR numbering around the run (first-occurrence
runs of that unit and mode), mark the rare columns, unbuilt residues (gaps in seq_index), and say whether the run
touches the rare column (is the disagreement just a different placement of the same insertion?)."""
import json, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
R=pd.read_csv(f'{OUT}/v01_unit_modes.csv'); P=R[R.obs].copy()
ch=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables/chain_novelty.csv.gz',usecols=['pdb','chain','unit'],low_memory=False)
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv').merge(ch,on=['pdb','chain'])
need=set()
sel=runs.merge(P[['unit','mode']],on=['unit','mode'])
for p,c in sel[['pdb','chain']].drop_duplicates().values: need.add((p,c))
REC={}
for f in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
    for l in open(f'{SP}/sabr/out/{f}'):
        d=json.loads(l)
        if (d['pdb'],d['chain']) in need and (d['pdb'],d['chain']) not in REC: REC[(d['pdb'],d['chain'])]=d['residues']
out=[]
for (u,m),g in sel.groupby(['unit','mode']):
    rr=P[(P.unit==u)&(P['mode']==m)].iloc[0]; rare=set(rr.cols.split())
    for _,r in g.drop_duplicates(['pdb','chain']).head(2).iterrows():
        res=REC.get((r.pdb,r.chain))
        if res is None: print('missing',r.pdb,r.chain); continue
        lo,hi=r.imgt0-6,r.imgt1+6
        lines=[]; prev=None; touch=False
        for a,ic,rn,si,ai,ains,sa,sins in res:
            if ai is None and sa is None: continue
            p=ai if ai is not None else sa
            if p is None or p<lo or p>hi: prev=si; continue
            gap='' if prev is None or si==prev+1 else f'  <-- {si-prev-1} unbuilt'
            A=f'{ai}{ains or ""}' if ai is not None else '-'; S=f'{sa}{sins or ""}' if sa is not None else '-'
            dis=A!=S; rc=A in rare or S in rare
            lines.append(f'   {rn} si={si:4d} ANARCI {A:6s} SAbR {S:6s} {"DIS" if dis else "   "} {"RARECOL" if A in rare else ""}{gap}')
            prev=si
        print(f'== {m} {r.pdb}_{r.chain} run IMGT {r.imgt0}-{r.imgt1} delta {r.delta} origin "{r.origin}" next_to_gap {r.next_to_gap_origin}; rare cols {rr.cols[:60]}')
        print('\n'.join(lines))
        out.append(dict(unit=u,mode=m,pdb=r.pdb,chain=r.chain,imgt0=r.imgt0,imgt1=r.imgt1,delta=r.delta,origin=r.origin,rare_cols=rr.cols))
pd.DataFrame(out).to_csv(f'{OUT}/v02_case_runs.csv',index=False)
