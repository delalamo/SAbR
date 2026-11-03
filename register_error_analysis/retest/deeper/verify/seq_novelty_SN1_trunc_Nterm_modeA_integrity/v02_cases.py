"""Print per-residue ANARCI vs SAbR labels at the N-terminus for every mode A case chain with truncated SEQRES,
plus the same for non-truncated mode A cases (summary)."""
import json, pandas as pd, numpy as np
from Bio.SeqUtils import seq1
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T=pd.read_pickle(f'{SP}/deeper/cache/verify_SN1/chain_nterm.pkl')
r=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
a=r[r['mode']=='A_FR1Nterm']
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','unit','seq','species'])
a=a.merge(ch,on=['pdb','chain'],how='left').merge(T,on=['pdb','chain'],how='left')
a=a[a.method=='EM']
want={(p,c) for p,c in zip(a.pdb,a.chain)}
recs={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        d=json.loads(l)
        if (d['pdb'],d['chain']) in want and d.get('status')=='ok': recs[(d['pdb'],d['chain'])]=d
out=[]
for unit,g in a.groupby('unit'):
    row=g.iloc[0]; key=(row.pdb,row.chain); rec=recs.get(key)
    res=[x for x in rec['residues'] if x[4] is not None][:14]
    s=lambda x: f"{x[4]}{x[5] or ''}" ; t=lambda x: f"{x[6]}{x[7] or ''}"
    out.append(dict(pdb=row.pdb,chain=row.chain,ctype=row.ctype,species=row.species,lab=row.last_author,year=row.year,
        seq_first_imgt=row.seq_first_imgt,seq_start_idx=row.seq_start_idx,built_first_anarci=row.built_first_anarci,
        seqres_head=row.seq[:14],
        built=''.join(seq1(x[2]) for x in res),
        anarci=' '.join(s(x) for x in res), sabr=' '.join(t(x) for x in res),
        run=f"{row.imgt0}-{row.imgt1} d{row.delta:+.0f}"))
O=pd.DataFrame(out)
O.to_csv('modeA_case_nterm_labels.csv',index=False)
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',80)
tr=O[O.seq_first_imgt>1]
for _,x in tr.iterrows():
    print(f"{x.pdb} {x.chain} {x.ctype} {x.species} {x.lab} {x.year} seqres={x.seqres_head} start_idx={x.seq_start_idx} run={x.run}")
    print('   built :',x.built); print('   ANARCI:',x.anarci); print('   SAbR  :',x.sabr)
print('\nnon-truncated cases: built_first_anarci distribution'); print(O[O.seq_first_imgt==1].built_first_anarci.value_counts())
print('truncated cases: built_first_anarci == seq_first_imgt ?', (tr.built_first_anarci==tr.seq_first_imgt).sum(), len(tr))
