"""Independent re-derivation of the paper-confirmed case set from Table S1 + runs_all.csv (verifier's own code)."""
import json, collections, pandas as pd, numpy as np, re
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_model_fingerprint/tables'
recs=collections.defaultdict(list)   # all records per chain, in file order validation, snapshot_rest, new
for fn in ['validation','snapshot_rest','new']:
    with open(f'{SP}/sabr/out/{fn}.jsonl') as fh:
        for l in fh:
            r=json.loads(l)
            if r.get('status')=='ok' and r.get('residues'):
                recs[(r['pdb'],r['chain'])].append((fn,r['residues']))
multi={k:v for k,v in recs.items() if len(v)>1}
ndiff=sum(1 for k,v in multi.items() if any(json.dumps(v[0][1])!=json.dumps(x[1]) for x in v[1:]))
print('chains with >1 ok record:',len(multi),'with differing residue lists:',ndiff)
last={k:v[-1][1] for k,v in recs.items()}      # later file wins (new > snapshot_rest > validation)
first={k:v[0][1] for k,v in recs.items()}
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',dtype={'chain':str},usecols=['pdb','chain','paper','method','release','err_first','err_any','title'])
ck=set(zip(C.pdb,C.chain))
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx')
t['pdb']=t.PDB_CHAIN_ID.str.split(':').str[0].str.lower(); t['chain']=t.PDB_CHAIN_ID.str.split(':').str[1]
m=t.RESIDUE_RANGE.astype(str).str.extract(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*$')
print('Table S1 rows',len(t),'unparsable ranges',int(m[0].isna().sum()))
t['rs']=m[0].astype(float); t['re']=m[1].astype(float)
t=t[[k in ck for k in zip(t.pdb,t.chain)]].copy(); print('on scanned chains',len(t))
def vf(r,src):
    res=src.get((r.pdb,r.chain))
    if res is None: return np.nan
    inr=[x for x in res if r.rs<=x[0]<=r.re]
    return np.mean([x[4] is not None for x in inr]) if inr else 0.0
t['vfrac_last']=t.apply(lambda r:vf(r,last),axis=1); t['vfrac_first']=t.apply(lambda r:vf(r,first),axis=1)
print('vfrac>=0.5 (last-wins rec):',int((t.vfrac_last>=.5).sum()),' (first-wins rec):',int((t.vfrac_first>=.5).sum()))
t=t[t.vfrac_last>=.5].copy()
R=pd.read_csv(f'{SP}/deeper/runs_all.csv',dtype={'chain':str})
L=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/synthesis/artifact_ledger_runs.csv',dtype={'chain':str})
R=R.merge(L[['pdb','chain','imgt0','imgt1','flag']].drop_duplicates(['pdb','chain','imgt0','imgt1']),on=['pdb','chain','imgt0','imgt1'],how='left')
bad=0; rn=[]
for r in R.itertuples():
    v=[x for x in last[(r.pdb,r.chain)] if x[4] is not None]
    if not(v[r.i0][4]==r.imgt0 and v[r.i1][4]==r.imgt1): bad+=1
    rn.append({x[0] for x in v[int(r.i0):int(r.i1)+1]})
R['rn']=rn; print('runs whose i0/i1 do not hit imgt0/imgt1:',bad)
Rg=dict(list(R.groupby(['pdb','chain'])))
out=[]
for e in t.itertuples():
    fl=set(range(int(e.rs),int(e.re)+1)); best=None; bo=0; nover=0
    g=Rg.get((e.pdb,e.chain))
    if g is not None:
        for r in g.itertuples():
            o=len(fl&r.rn)
            if o>0: nover+=1
            if o>bo: bo,best=o,r
    d=dict(pdb=e.pdb,chain=e.chain,s1_range=e.RESIDUE_RANGE,v_overlap=bo,v_nruns_overlap=nover)
    if best is not None: d.update(v_imgt0=best.imgt0,v_imgt1=best.imgt1,v_mode=best.mode,v_cls=str(best.cls),v_flag=best.flag)
    out.append(d)
V=pd.DataFrame(out)
def st(r):
    if r.v_overlap==0: return 'not confirmed by SAbR/ANARCI'
    if r.v_cls in ('1a','1b','2'): return 'copy of an earlier model'
    if r.v_cls=='4': return 'known detector artifact (origin 4)'
    f=str(r.v_flag)
    if f in ('indel_gap_artifact_confirmed','DE_chain_end_artifact','amyloid_fibril','not_a_register_shift'): return 'artifact (synthesis ledger, strict)'
    if f=='near_copy': return 'near copy (ledger)'
    if f in ('indel_gap_artifact_probable','DE_not_confirmed_other'): return 'confirmed, loose-ledger doubt'
    return 'confirmed first occurrence'
V['v_status']=V.apply(st,axis=1)
print(V.v_status.value_counts().to_string())
A=pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv',dtype={'chain':str})
print('investigator rows',len(A),'verifier rows',len(V))
A['k']=A.pdb+':'+A.chain+':'+A.s1_range.astype(str); V['k']=V.pdb+':'+V.chain+':'+V.s1_range.astype(str)
J=A.merge(V,on='k',how='outer',indicator=True,suffixes=('','_v'))
print('merge', J._merge.value_counts().to_dict())
both=J[J._merge=='both']
print('status mismatches:',int((both.status!=both.v_status).sum()))
print('overlap mismatches:',int((both.overlap!=both.v_overlap).sum()))
print('imgt0 mismatches:',int(((both.imgt0!=both.v_imgt0)&both.imgt0.notna()).sum()))
print('rows where >1 run overlaps:',int((V.v_nruns_overlap>1).sum()))
print(V[V.v_nruns_overlap>1][['k','v_status','v_nruns_overlap']].to_string())
V.to_csv(f'{OUT}/v01_caseset.csv',index=False)
# chain-level
C=C.merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz',dtype={'chain':str}),on=['pdb','chain'])
sk=set(zip(V[V.v_status=='confirmed first occurrence'].pdb,V[V.v_status=='confirmed first occurrence'].chain))
C['v_strict']=[k in sk for k in zip(C.pdb,C.chain)]
print('strict chains verifier',int(C.v_strict.sum()),'investigator',int(C.paper_case_strict.sum()),'disagree',int((C.v_strict!=C.paper_case_strict).sum()))
d=C[C.v_strict]; print('entries',d.pdb.nunique(),'papers',d.paper.nunique(),'EM',int((d.method=='EM').sum()))
# chains carrying BOTH a strict error and another (copy/artifact/unconfirmed) Table S1 error
mix=V.groupby(['pdb','chain']).v_status.apply(lambda s:';'.join(sorted(set(s))))
print('strict chains with mixed statuses:'); print(mix[[k in sk for k in mix.index]][mix[[k in sk for k in mix.index]].str.contains(';')].to_string())
print('Zhou 2021 strict chains',int((d.paper=='10.1038/s41422-021-00487-9').sum()))
print(d.paper.value_counts().head(6).to_string())
# control: check amyloid and err_any
ctl=C[C.paper_control]; print('controls',len(ctl),'err_any in controls',int(ctl.err_any.astype(bool).sum()),'release>window',int((ctl.release.astype(str)>'2022-04-05').sum()))
