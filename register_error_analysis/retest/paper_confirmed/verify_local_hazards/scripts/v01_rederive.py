"""Independent re-derivation of the paper-confirmed case set (verify_local_hazards).
Table S1 -> scanned chains -> V-fraction >= 0.5 -> overlap with SAbR/ANARCI runs (runs_all.csv) -> status."""
import json, re, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_local_hazards'
LED='/home/user/SAbR/register_error_analysis/retest/deeper/synthesis/artifact_ledger_runs.csv'
rec={}
for fn in ['validation.jsonl','snapshot_rest.jsonl','new.jsonl']:
    with open(f'{SP}/sabr/out/{fn}') as fh:
        for line in fh:
            r=json.loads(line)
            if r.get('status')=='ok' and r.get('residues'): rec[(r['pdb'],r['chain'])]=r['residues']
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
keys=set(zip(C.pdb,C.chain))
T=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx')
T['pdb']=[s.split(':')[0].lower() for s in T.PDB_CHAIN_ID]; T['chain']=[s.split(':')[1] for s in T.PDB_CHAIN_ID]
m=T.RESIDUE_RANGE.str.extract(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*$').astype(int); T['lo']=m[0]; T['hi']=m[1]
T['scanned']=[k in keys for k in zip(T.pdb,T.chain)]
# case-insensitive chain match check (are any lost to chain-case mismatches?)
kl={(p,c.upper()) for p,c in keys}
print('Table S1 rows',len(T),'| on scanned chains',T.scanned.sum(),'| would match only case-insensitively',
      sum((not s) and ((p,c.upper()) in kl) for p,c,s in zip(T.pdb,T.chain,T.scanned)))
T=T[T.scanned].copy()
def resin(r):
    return [x for x in rec[(r.pdb,r.chain)] if r.lo<=x[0]<=r.hi]
T['n_in']=[len(resin(r)) for r in T.itertuples()]
T['n_v']=[sum(x[4] is not None for x in resin(r)) for r in T.itertuples()]
T['vfrac']=np.where(T.n_in>0,T.n_v/T.n_in.clip(lower=1),0)
print('vfrac>=0.5:',(T.vfrac>=0.5).sum())
T=T[T.vfrac>=0.5].copy()
R=pd.read_csv(f'{SP}/deeper/runs_all.csv')
L=pd.read_csv(LED)[['pdb','chain','imgt0','imgt1','flag']]
R=R.merge(L,on=['pdb','chain','imgt0','imgt1'],how='left')
def run_keys(r):
    v=[x for x in rec[(r.pdb,r.chain)] if x[4] is not None]
    assert v[r.i0][4]==r.imgt0 and v[r.i1][4]==r.imgt1, (r.pdb,r.chain)
    return [(x[0],x[1]) for x in v[r.i0:r.i1+1]]
R['rk']=[run_keys(r) for r in R.itertuples()]
rows=[]
for e in T.itertuples():
    cand=R[(R.pdb==e.pdb)&(R.chain==e.chain)]
    ov=[(sum(e.lo<=k[0]<=e.hi for k in r.rk),i) for i,r in cand.iterrows()]
    ov=[o for o in ov if o[0]>0]
    d=dict(pdb=e.pdb,chain=e.chain,s1_range=e.RESIDUE_RANGE,n_overlap_runs=len(ov))
    if ov:
        ov.sort(reverse=True); r=cand.loc[ov[0][1]]
        d.update(overlap=ov[0][0],imgt0=r.imgt0,imgt1=r.imgt1,mode=r['mode'],cls=str(r.cls),flag=r.flag,
                 all_cls='|'.join(sorted(str(cand.loc[i].cls) for _,i in ov)),all_flags='|'.join(sorted(str(cand.loc[i].flag) for _,i in ov)))
    rows.append(d)
S=pd.DataFrame(rows)
def st(r):
    if r.n_overlap_runs==0: return 'not confirmed by SAbR/ANARCI'
    if r.cls in ('1a','1b','2'): return 'copy of an earlier model'
    if r.cls=='4': return 'known detector artifact (origin 4)'
    if r.flag in ('indel_gap_artifact_confirmed','DE_chain_end_artifact','amyloid_fibril','not_a_register_shift'): return 'artifact (synthesis ledger, strict)'
    if r.flag=='near_copy': return 'near copy (ledger)'
    if r.flag in ('indel_gap_artifact_probable','DE_not_confirmed_other'): return 'confirmed, loose-ledger doubt'
    return 'confirmed first occurrence'
S['status_v']=S.apply(st,axis=1)
print(S.status_v.value_counts().to_string())
print('errors overlapping >1 run:',(S.n_overlap_runs>1).sum())
print(S[S.n_overlap_runs>1][['pdb','chain','s1_range','all_cls','all_flags','status_v']].to_string())
A=pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv')
A['k']=A.pdb+A.chain+A.s1_range; S['k']=S.pdb+S.chain+S.s1_range
print('rows: annotated',len(A),'mine',len(S),'| key sets equal:',set(A.k)==set(S.k))
M=A.merge(S[['k','status_v','imgt0','imgt1','mode']],on='k',suffixes=('','_v'))
dis=M[(M.status!=M.status_v)|((M.imgt0!=M.imgt0_v)&M.imgt0.notna())]
print('status/run discrepancies:',len(dis)); print(dis[['pdb','chain','s1_range','status','status_v','imgt0','imgt0_v']].to_string())
# chain-level strict set
C=C.set_index(['pdb','chain'])
strict=S[S.status_v=='confirmed first occurrence'][['pdb','chain']].drop_duplicates()
incl=S[S.status_v.isin(['confirmed first occurrence','confirmed, loose-ledger doubt','copy of an earlier model','near copy (ledger)'])][['pdb','chain']].drop_duplicates()
for nm,d in [('strict',strict),('incl_copies',incl)]:
    cc=C.loc[list(zip(d.pdb,d.chain))].reset_index()
    print(nm,'chains',len(cc),'entries',cc.pdb.nunique(),'units',len(cc.drop_duplicates(['paper','vseq'])),'papers',cc.paper.nunique(),
          'EM',int((cc.method=='EM').sum()),'Xray',int((cc.method!='EM').sum()),'max release',cc.release.max())
F=pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz')
print('strict chain set equal to chain_flags:',set(zip(strict.pdb,strict.chain))==set(zip(F[F.paper_case_strict].pdb,F[F.paper_case_strict].chain)))
print('incl chain set equal:',set(zip(incl.pdb,incl.chain))==set(zip(F[F.paper_case_incl_copies].pdb,F[F.paper_case_incl_copies].chain)))
# chains carrying both a strict error and a non-strict status
S.to_csv(f'{OUT}/tables/v01_cases_rederived.csv',index=False)
# strict chains whose run is also flagged artifact on another S1 error
mix=S.groupby(['pdb','chain']).status_v.agg(lambda s:'|'.join(sorted(set(s))))
print('chains with mixed statuses:'); print(mix[mix.str.contains(r'\|')].to_string())
