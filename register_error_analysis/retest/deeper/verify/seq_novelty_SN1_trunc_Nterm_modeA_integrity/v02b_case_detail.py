"""Per-case view of the 59 mode-A EM units: SEQRES N-terminus, detector ANARCI vs SAbR numbering of the first modelled residues, run windows."""
import json, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{T}/units.csv.gz',low_memory=False)
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
ra=runs[runs['mode']=='A_FR1Nterm']
EM=U.method_u=='EM'
A=U[EM&U.err_first_u&U.modes_u.fillna('').str.contains('A_FR1Nterm')].copy()
want=set(zip(A.pdb,A.chain))|set(zip(ra.pdb,ra.chain))
rec={}
for f in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        d=json.loads(l)
        if (d['pdb'],d['chain']) in want and d.get('status')=='ok': rec[(d['pdb'],d['chain'])]=d['residues']
seqd=dict(zip(zip(ch.pdb,ch.chain),ch.seq))
rows=[]
for _,u in A.iterrows():
    # use the chain in the unit that carries a mode A run (rep chain if it has one)
    key=(u.pdb,u.chain)
    if not ((ra.pdb==u.pdb)&(ra.chain==u.chain)).any():
        unit_chains=ch[ch.unit==u.unit]
        cand=[(p,c) for p,c in zip(unit_chains.pdb,unit_chains.chain) if ((ra.pdb==p)&(ra.chain==c)).any()]
        key=cand[0]
    r=rec.get(key)
    rr=ra[(ra.pdb==key[0])&(ra.chain==key[1])].sort_values('imgt0').iloc[0]
    s=seqd.get(key,'')
    first=[x for x in r if x[4] is not None][:8] if r else []
    rows.append(dict(unit=u.unit,pdb=key[0],chain=key[1],ctype=u.ctype,trunc=u.seq_nt_trunc,seq_first_imgt=u.seq_first_imgt,
        seq_nterm=s[:14],first_modelled_seqidx=first[0][3] if first else None,
        anarci_first=[x[4] for x in first], sabr_first=[x[6] for x in first], aa_first=''.join(x[2][:1] for x in first),
        run_imgt0=rr.imgt0,run_imgt1=rr.imgt1,delta=rr.delta,run_i0=rr.i0,last_author=u.last_author,paper=u.paper,year=u.dep_year))
D=pd.DataFrame(rows)
D.to_csv('v02_modeA_case_detail.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(D[D.trunc].drop(columns=['unit']).to_string())
print(D[~D.trunc].drop(columns=['unit','paper']).head(40).to_string())
print(pd.crosstab(D.trunc,D.delta))
print(pd.crosstab(D.trunc,D.run_imgt0))
