import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0,'/tmp/scripts'); from ca_fetch import CACHE
T='/tmp/tmpl'
def load(p,c):
    fn=f'{CACHE}/{p}_{c}.json'
    if not os.path.exists(fn): return None
    return {sid-1:np.array(xyz) for sid,*xyz in json.load(open(fn))}
def kabsch(P,Q):
    pc,qc=P.mean(0),Q.mean(0); H=(Q-qc).T@(P-pc); U,S,Vt=np.linalg.svd(H); dd=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,dd]); R=Vt.T@D@U.T; return R,pc,qc
def call(ref,cand,off,s0,s1,k):
    core=[i for i in range(s0-40,s1+41) if (i<s0-4 or i>s1+4) and i in ref and (i+off) in cand]
    if len(core)<12: return 'no core',np.nan,np.nan,np.nan
    P=np.array([ref[i] for i in core]); Q=np.array([cand[i+off] for i in core])
    R,pc,qc=kabsch(P,Q); Qt=(Q-qc)@R.T+pc; rmsd=float(np.sqrt(((Qt-P)**2).sum(1).mean()))
    tr=lambda x:(x-qc)@R.T+pc
    seg=[i for i in range(s0,s1+1) if i in ref]
    def med(dk):
        ds=[np.linalg.norm(ref[i]-tr(cand[i+off+dk])) for i in seg if (i+off+dk) in cand]
        return np.median(ds) if len(ds)>=3 else np.nan
    d0=med(0); dk=np.nanmin([med(k),med(-k)]) if not (np.isnan(med(k)) and np.isnan(med(-k))) else np.nan
    if rmsd>3.0: return 'core mismatch',rmsd,d0,dk
    if np.isnan(d0) and np.isnan(dk): return 'segment missing',rmsd,d0,dk
    if not np.isnan(d0) and d0<2.0 and (np.isnan(dk) or d0<dk-1.0): return 'same as flagged',rmsd,d0,dk
    if not np.isnan(dk) and dk<2.0 and (np.isnan(d0) or dk<d0-1.0): return 'other register',rmsd,d0,dk
    return 'ambiguous',rmsd,d0,dk
def run(families=None):
    F=pd.read_pickle(f'{T}/families.pkl').set_index('family'); S=pd.read_pickle(f'{T}/selected.pkl')
    if families is not None: S=S[S.family.isin(families)]
    out=[]
    for fid,g in S.groupby('family'):
        f=F.loc[fid]; k=max(1,abs(int(f['shift'])))
        r0=g[g.is_ref].iloc[0]; ref=load(r0.pdb,r0.chain)
        if not ref: continue
        ref={i-r0.offset:v for i,v in ref.items()}   # into family seq coords
        for r in g.itertuples():
            cand=load(r.pdb,r.chain)
            if cand is None: lab=('not fetched',np.nan,np.nan,np.nan)
            elif not cand: lab=('no coords',np.nan,np.nan,np.nan)
            else: lab=call(ref,cand,int(r.offset),int(f.s0),int(f.s1),k)
            out.append(dict(family=fid,pdb=r.pdb,chain=r.chain,is_ref=r.is_ref,flagged=r.flagged,date=r.date,res=r.res,method=r.method,
                            last_author=r.last_author,paper=r.paper,call=lab[0],core_rmsd=lab[1],d_same=lab[2],d_shift=lab[3]))
    return pd.DataFrame(out)
if __name__=='__main__':
    R=run(); R.to_pickle(f'{T}/register_calls.pkl'); print(R.call.value_counts())
