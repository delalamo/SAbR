"""Inspect the 19 truncated mode-A error units: SEQRES, per-residue ANARCI vs SAbR numbering at the N-terminus, run details."""
import json, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{SN}/units.csv.gz',low_memory=False)
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
EM=U.method_u=='EM'
A=EM&U.err_first_u&U.modes_u.str.contains('A_FR1Nterm')
need=set()
cu=ch.merge(U.loc[A,['unit','seq_first_imgt']],on='unit')
cu=cu[cu.err_first]
for r in cu.itertuples(): need.add((r.pdb,r.chain))
REC={}
for fn in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r=json.loads(line)
        if (r['pdb'],r['chain']) in need and r.get('residues'): REC[(r['pdb'],r['chain'])]=r['residues']
for u in U.loc[A].sort_values('seq_first_imgt').itertuples():
    cc=cu[cu.unit==u.unit]
    print('='*100); print(u.pdb,u.chain,u.ctype,u.species,u.v_gene,u.year,u.res,u.last_author,'seq_first_imgt',u.seq_first_imgt)
    for c in cc.itertuples():
        rr=runs[(runs.pdb==c.pdb)&(runs.chain==c.chain)]
        print('  chain',c.pdb,c.chain,'SEQRES[:30]',c.seq[:30],'| runs',[(a,b,d,m) for a,b,d,m in zip(rr.imgt0,rr.imgt1,rr.delta,rr['mode'])])
        res=REC.get((c.pdb,c.chain))
        if res:
            print('   aa   ',' '.join(f'{x[2]:>3s}' for x in res[:16]))
            print('   anarci',' '.join(f'{str(x[4])+(x[5] or ""):>3s}' for x in res[:16]))
            print('   sabr  ',' '.join(f'{str(x[6])+(x[7] or ""):>3s}' for x in res[:16]))
            print('   seqidx',' '.join(f'{str(x[3]):>3s}' for x in res[:16]))
        break
