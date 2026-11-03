"""Check that SAbR seq_index == 0-based SEQRES index (needed for mapping RCSB cis-peptide seq_ids)."""
import json, pandas as pd
from Bio.PDB.Polypeptide import three_to_index, index_to_one
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','seq','ctype','err_first','err_any','method'])
seq={(r.pdb,r.chain):r.seq for r in ch.itertuples()}
tot=bad=0; badch=[]
seen=set()
def one(x):
    try: return index_to_one(three_to_index(x))
    except Exception: return 'X'
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k not in seq or k in seen: continue
        seen.add(k); s=seq[k]
        if not isinstance(s,str): continue
        mm=0;n=0
        for x in r.get('residues') or []:
            if x[3] is None: continue
            n+=1
            if x[3]>=len(s) or s[x[3]]!=one(x[2]): mm+=1
        tot+=1
        if mm>0.05*max(n,1): bad+=1; badch.append((k,mm,n))
print('chains',tot,'with >5% residue-name mismatch at seq_index',bad)
print(badch[:20])
