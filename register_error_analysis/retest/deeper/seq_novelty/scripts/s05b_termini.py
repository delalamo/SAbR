"""Add sequence-level (SEQRES) V-domain termini to the unit table: first and last IMGT position present in the deposited sequence."""
import pickle, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
db=pickle.load(open(f'{SP}/deeper/cache/seq_novelty/abdb.pkl','rb')); M,keys=db['M'],db['keys']
pos=np.array([k[0] for k in keys])
U=pd.read_csv(f'{OUT}/units.csv.gz',low_memory=False)
fp=[];lp=[]
for r in U.dbrow:
    if r<0 or r>=len(M): fp.append(np.nan); lp.append(np.nan); continue
    occ=np.where(M[r]>0)[0]; fp.append(pos[occ].min()); lp.append(pos[occ].max())
U['seq_first_imgt']=fp; U['seq_last_imgt']=lp
U['seq_nt_trunc']=U.seq_first_imgt>1
U['seq_fr4_trunc']=U.seq_last_imgt<125
U.to_csv(f'{OUT}/units.csv.gz',index=False)
E=U[(U.method_u=='EM')&(U.err_first_u|U.clean)]
print(pd.crosstab(E.seq_first_imgt.clip(0,5),E.err_first_u,normalize='columns').round(3))
print(pd.crosstab(E.seq_last_imgt.clip(118,128),E.err_first_u,normalize='columns').round(3))
