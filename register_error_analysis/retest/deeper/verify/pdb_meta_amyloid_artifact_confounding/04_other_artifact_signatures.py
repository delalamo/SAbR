"""Side check: are there other first-occurrence 'errors' with a detector-failure signature like the amyloid ones?
Signatures: (a) whole-domain runs (>=60 residues) or |delta|>=5; (b) SAbR assigns IMGT positions >128 to residues that
ANARCI places inside the V domain (impossible under IMGT); (c) framework RMSD under SAbR's own numbering much worse
than under ANARCI's numbering (from 02_fold_test)."""
import json, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_confounding'
c=pd.read_csv(f'{SP}/deeper/chains.csv.gz'); r=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
c['amy']=c.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)
ef=c[c.err_first]; want=set(zip(ef.pdb,ef.chain))
beyond={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        d=json.loads(l)
        if (d['pdb'],d['chain']) not in want: continue
        n=sum(1 for x in d['residues'] if x[4] is not None and x[6] is not None and x[6]>128)
        beyond[(d['pdb'],d['chain'])]=n
ef=ef.assign(sabr_gt128=[beyond.get((p,ch),np.nan) for p,ch in zip(ef.pdb,ef.chain)])
F=pd.read_csv(f'{OUT}/fold_test_chains.csv')
ef=ef.merge(F[['pdb','chain','fw_rmsd_med','fw_rmsd_sabr_med']],on=['pdb','chain'],how='left')
rr=r.groupby(['pdb','chain']).agg(maxlen=('length','max'),maxabsd=('delta',lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan)).reset_index()
ef=ef.merge(rr,on=['pdb','chain'],how='left')
ef['sig_whole_domain']=(ef.maxlen>=60)
ef['sig_gt128']=(ef.sabr_gt128>0)
ef['sig_sabr_worse']=(ef.fw_rmsd_sabr_med>ef.fw_rmsd_med+3)
ef['sig_nonIg']=(ef.fw_rmsd_med>10)&(ef.fw_rmsd_sabr_med>10)
cols=['pdb','chain','ctype','amy','unit','err_modes','maxlen','maxabsd','sabr_gt128','fw_rmsd_med','fw_rmsd_sabr_med','title']
S=ef[ef.sig_whole_domain|ef.sig_gt128|ef.sig_sabr_worse|ef.sig_nonIg]
S[cols+['sig_whole_domain','sig_gt128','sig_sabr_worse','sig_nonIg']].to_csv(f'{OUT}/artifact_signature_chains.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
for s in ['sig_whole_domain','sig_gt128','sig_sabr_worse','sig_nonIg']:
    x=ef[ef[s]]; print(s,'chains',len(x),'units',x.unit.nunique(),'amyloid units',x[x.amy].unit.nunique(),'non-amyloid units',x[~x.amy].unit.nunique())
print(S[~S.amy].drop_duplicates('unit')[cols].to_string())
