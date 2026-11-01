"""Chain-level register calls with the flagged chain as reference; aggregate to entry level."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/scripts'); sys.path.insert(0,'/tmp/scripts')
from ca2 import get_ca2
from tmpl4_register import call
T='/tmp/tmpl'; T2='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/tmpl2'; O2=f'{T2}/out'
F=pd.read_pickle(f'{T}/families.pkl').set_index('family'); X=pd.read_pickle(f'{T2}/selected_chains.pkl')
def load(p,c):
    ca=get_ca2(p,c)
    return None if ca is None else {sid-1:np.array(xyz) for sid,*xyz in ca}
out=[]
for fid,g in X.groupby('family'):
    if not g.is_ref.any(): continue
    f=F.loc[fid]; k=max(1,abs(int(f['shift'])))
    r0=g[g.is_ref].iloc[0]; ref=load(r0.pdb,r0.chain)
    if not ref: continue
    ref={i-int(r0.offset):v for i,v in ref.items()}
    for r in g.itertuples():
        cand=load(r.pdb,r.chain)
        lab=('no coords',np.nan,np.nan,np.nan) if not cand else call(ref,cand,int(r.offset),int(f.s0),int(f.s1),k)
        d=r._asdict(); d.pop('Index'); d.update(call=lab[0],core_rmsd=lab[1],d_same=lab[2],d_shift=lab[3]); out.append(d)
Rc=pd.DataFrame(out); Rc.to_pickle(f'{O2}/register_calls_chain.pkl')
Rc['S']=Rc.call=='same as flagged'; Rc['O']=Rc.call=='other register'
agg=Rc.groupby(['family','pdb']).agg(is_ref=('is_ref','any'),flagged=('flagged','first'),date=('date','first'),res=('res','first'),method=('method','first'),
    last_author=('last_author','first'),paper=('paper','first'),nS=('S','sum'),nO=('O','sum'),n_chains=('chain','size'),
    call=('call',lambda x: 'same as flagged' if (x=='same as flagged').any() else ('other register' if (x=='other register').any() else x.iloc[0]))).reset_index()
agg['chain']=''
agg.to_pickle(f'{O2}/register_calls.pkl')
print(Rc.call.value_counts().to_string()); print('entries',len(agg)); print(agg.call.value_counts().to_string())
print('entries with mixed copies (some same, some other):',((agg.nS>0)&(agg.nO>0)).sum())
