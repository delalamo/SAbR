"""For every unit representative chain: first modelled residue's ANARCI IMGT number (detector arm), from the SAbR scan records.
Writes cache table used by later scripts."""
import json, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{T}/units.csv.gz',low_memory=False,usecols=['unit','pdb','chain'])
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','unit'])
want=set(zip(ch.pdb,ch.chain))
rows=[]
for f in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        d=json.loads(l)
        if (d['pdb'],d['chain']) not in want or d.get('status')!='ok': continue
        res=[x for x in d['residues'] if x[4] is not None]
        if not res: continue
        x=res[0]
        # first SEQRES index modelled overall, and ANARCI/SAbR number of first numbered modelled residue
        rows.append(dict(pdb=d['pdb'],chain=d['chain'],src=f,first_mod_anarci=x[4],first_mod_sabr=x[6],first_mod_seqidx=x[3],
                         n_res=len(res)))
R=pd.DataFrame(rows).drop_duplicates(['pdb','chain'])
R.to_csv(f'{SP}/deeper/cache/verify_SN1/first_modelled.csv.gz',index=False)
print(len(R), R.first_mod_anarci.value_counts().sort_index().head(15).to_dict())
