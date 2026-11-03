"""Per-chain N-terminal features: first modelled residue (ANARCI and SAbR numbering), SEQRES first IMGT,
residues in SEQRES before the V domain; plus unit-level table merged with the investigator's units."""
import json, pickle, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
ch=pd.read_csv(f'{SN}/chain_novelty.csv.gz',low_memory=False)
base=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','seq','vseq'])
ch=ch.drop(columns=[c for c in ['seq'] if c in ch]).merge(base[['pdb','chain','seq']],on=['pdb','chain'],how='left')
db=pickle.load(open(f'{SP}/deeper/cache/seq_novelty/abdb.pkl','rb')); M,keys=db['M'],db['keys']
pos=np.array([k[0] for k in keys])
def first_last(r):
    if not np.isfinite(r) or r<0 or r>=len(M): return np.nan,np.nan
    occ=np.where(M[int(r)]>0)[0]; return pos[occ].min(),pos[occ].max()
fl=[first_last(r) for r in ch.dbrow]
ch['seq_first_imgt']=[a for a,b in fl]
ch['v_offset']=[s.find(v) if isinstance(s,str) and isinstance(v,str) else -1 for s,v in zip(ch.seq,ch.vseq)]
REC={}
for fn in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r=json.loads(line)
        if r.get('status')=='ok' and r.get('residues'): REC[(r['pdb'],r['chain'])]=r['residues']
mf=[];ms=[];m1si=[];nmod10=[]
for p,c in zip(ch.pdb,ch.chain):
    res=REC.get((p,c))
    if not res: mf.append(np.nan); ms.append(np.nan); m1si.append(np.nan); nmod10.append(np.nan); continue
    a=[x for x in res if x[4] is not None]
    mf.append(min(x[4] for x in a) if a else np.nan)
    first=min(a,key=lambda x:x[4]) if a else None
    ms.append(first[6] if first else np.nan); m1si.append(first[3] if first else np.nan)
    nmod10.append(sum(1 for x in a if x[4]<=10))
ch['model_first_anarci']=mf; ch['model_first_sabr']=ms; ch['model_first_seqidx']=m1si; ch['n_mod_1_10']=nmod10
ch[['pdb','chain','unit','ctype','method','err_first','err_any','seq_first_imgt','v_offset','model_first_anarci','model_first_sabr','model_first_seqidx','n_mod_1_10']].to_csv(f'{OUT}/chain_nterm.csv.gz',index=False)
print(ch[['seq_first_imgt','model_first_anarci']].describe())
print('chains with model_first < seq_first (should be 0):',(ch.model_first_anarci<ch.seq_first_imgt).sum())
