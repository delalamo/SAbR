"""Detector behaviour at truncated N-termini in the 1,039 high-resolution template chains (independent of the
3-5 A case set): does SAbR disagree with ANARCI at the N-terminus simply because SEQRES is truncated?"""
import pandas as pd, numpy as np, json
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_integrity'
TT=pd.read_pickle(f'{SP}/deeper/cache/verify_SN1/templ_nterm.pkl')
TT=TT[TT.seq_first_imgt.notna()&(TT.seq_start_idx<300)]   # drop two non-antibody Ig domains inside large fusions (start idx 776/798)
TT['trunc28']=TT.seq_first_imgt.between(2,8); TT['bgt1']=TT.built_first_anarci>1
TT['cat']=np.where(TT.trunc28,'SEQRES starts IMGT 2-8',np.where(TT.seq_first_imgt>8,'SEQRES starts >8',np.where(TT.bgt1,'full SEQRES, IMGT1 unbuilt','full SEQRES, built from 1')))
idx={}
for l in open('/tmp/supp/pdbwide/entries.idx',errors='ignore'):
    f=l.rstrip('\n').split('\t')
    if len(f)>6: idx[f[0].lower()]=(f[2],f[6][:60] if len(f)>6 else '')
with open(f'{OUT}/v06_templates.log','w') as log:
    s=TT.groupby('cat').agg(n=('pdb','size'),nt_runs=('nt_run','sum'),rate=('nt_run','mean')).to_string()
    print(s); log.write(s+'\n')
    h=TT[TT.nt_run][['pdb','chain','seq_first_imgt','built_first_anarci','nt_run_imgt0','nt_run_imgt1','nt_run_delta']]
    h['deposit']=h.pdb.map(lambda p: idx.get(p,('',''))[0])
    s=h.to_string(); print(s); log.write(s+'\n')
    a=TT[TT.trunc28]; b=TT[~TT.trunc28&(TT.seq_first_imgt==1)]
    t=[[a.nt_run.sum(),len(a)-a.nt_run.sum()],[b.nt_run.sum(),len(b)-b.nt_run.sum()]]
    s=f'templates: N-terminal run in truncated SEQRES {t[0][0]}/{len(a)} vs full SEQRES {t[1][0]}/{len(b)}; Fisher p={stats.fisher_exact(t)[1]:.3g}'
    print(s); log.write(s+'\n')
