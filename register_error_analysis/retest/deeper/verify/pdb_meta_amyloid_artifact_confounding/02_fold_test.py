"""Detector-independent fold test: is the V chain an Ig fold at all?
For each chain, CA atoms at ANARCI (sequence-based) IMGT framework positions are superposed (Kabsch) onto
reference V domains of the same chain type taken from clean cryo-EM/X-ray entries; median RMSD over 5 references.
An Ig V domain gives ~1-2 A; a non-Ig (amyloid) fold gives >>5 A. Also records SAbR's own score (sabr_s),
run length and |delta| so the signature of the amyloid runs can be compared with other first-occurrence runs."""
import json, os, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_confounding'
DL=f'{SP}/sabr/dl'
c=pd.read_csv(f'{SP}/deeper/chains.csv.gz')
c['amy']=c.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)
cached={f[:-4] for f in os.listdir(DL) if f.endswith('.cif')}
c=c[c.pdb.isin(cached)].copy()
c['grp']=np.where(c.amy,'amyloid',np.where(c.err_first,'error (non-amyloid)',np.where(~c.err_any,'clean','copied/artifact only')))
want=set(zip(c.pdb,c.chain))|{('6hud','A'),('8cpe','A')}
recs={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r=json.loads(l)
        if (r['pdb'],r['chain']) in want and r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r
FW=list(range(1,27))+list(range(39,56))+list(range(66,105))+list(range(118,129))
def read_ca(path,chains):
    cols=[];out={}
    with open(path) as fh:
        inloop=False
        for line in fh:
            if line.startswith('_atom_site.'): cols.append(line.strip().split('.',1)[1]); inloop=True; continue
            if inloop and (line.startswith('ATOM') or line.startswith('HETATM')):
                if not cols: continue
                t=line.split()
                if len(t)!=len(cols): continue
                d=dict(zip(cols,t))
                if d.get('label_atom_id')!='CA' or d.get('pdbx_PDB_model_num','1')!='1': continue
                if d.get('label_alt_id','.') not in ('.','A','?'): continue
                ch=d.get('auth_asym_id')
                if ch not in chains: continue
                ins=d.get('pdbx_PDB_ins_code','?'); ins='' if ins in ('?','.') else ins
                key=(ch,int(d['auth_seq_id']),ins)
                if key not in out: out[key]=np.array([float(d['Cartn_x']),float(d['Cartn_y']),float(d['Cartn_z'])])
            elif inloop and cols and line.startswith('#'):
                if out or 'Cartn_x' in cols: break
    return out
def fw_coords(pdb,chain,ca,col=4):
    r=recs.get((pdb,chain))
    if r is None: return None
    m={}
    for x in r['residues']:
        an,ins=x[col],x[col+1]
        if an is None or ins or an not in FW: continue
        k=(chain,int(x[0]),x[1] or '')
        if k in ca: m[an]=ca[k]
    return m
def kabsch_rmsd(A,B):
    A=A-A.mean(0); B=B-B.mean(0); H=A.T@B; U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,d]); R=Vt.T@D@U.T; return float(np.sqrt(((A@R.T-B)**2).sum(1).mean()))
coords={}; scoords={}
rows=[]
pdbs=sorted({p for p,_ in want})
for i,p in enumerate(pdbs):
    chs={ch for q,ch in want if q==p}
    path=f'{DL}/{p}.cif'
    if not os.path.exists(path): continue
    ca=read_ca(path,chs)
    for ch in chs:
        m=fw_coords(p,ch,ca)
        if m: coords[(p,ch)]=m
        m2=fw_coords(p,ch,ca,col=6)
        if m2: scoords[(p,ch)]=m2
print('chains with coords',len(coords))
# references: clean chains of each type with >=95 framework positions, one per entry, best resolution
cc=c[(c.grp=='clean')].copy(); cc['nfw']=[len(coords.get((p,ch),{})) for p,ch in zip(cc.pdb,cc.chain)]
refs={}
for ct in ['H','K','L']:
    s=cc[(cc.ctype==ct)&(cc.nfw>=88)&(cc.vhh!=True)].sort_values('res').drop_duplicates('pdb').head(5)
    refs[ct]=list(zip(s.pdb,s.chain)); print(ct,'refs',refs[ct])
out=[]
allc=c.copy()
extra=pd.DataFrame([dict(pdb='6hud',chain='A',ctype='L',grp='amyloid (excluded by <80 res filter)',amy=True,err_first=np.nan),dict(pdb='8cpe',chain='A',ctype='L',grp='amyloid (excluded by <80 res filter)',amy=True,err_first=np.nan)])
allc=pd.concat([allc,extra],ignore_index=True)
for _,r in allc.iterrows():
    m=coords.get((r.pdb,r.chain))
    if not m: continue
    rm=[]
    for rp in refs[r.ctype]:
        if rp==(r.pdb,r.chain): continue
        R=coords[rp]; common=sorted(set(m)&set(R))
        if len(common)<30: continue
        rm.append(kabsch_rmsd(np.array([m[k] for k in common]),np.array([R[k] for k in common])))
    rs=[]; m2=scoords.get((r.pdb,r.chain),{})
    for rp in refs[r.ctype]:
        if rp==(r.pdb,r.chain): continue
        R=coords[rp]; common=sorted(set(m2)&set(R))
        if len(common)<30: continue
        rs.append(kabsch_rmsd(np.array([m2[k] for k in common]),np.array([R[k] for k in common])))
    rec=recs.get((r.pdb,r.chain))
    out.append(dict(pdb=r.pdb,chain=r.chain,ctype=r.ctype,grp=r.grp,method=r.get('method'),res=r.get('res'),n_fw=len(m),
                    fw_rmsd_med=np.median(rm) if rm else np.nan,fw_rmsd_sabr_med=np.median(rs) if rs else np.nan,sabr_s=rec.get('sabr_s') if rec else np.nan))
O=pd.DataFrame(out)
O.to_csv(f'{OUT}/fold_test_chains.csv',index=False)
pd.set_option('display.width',200)
print(O.groupby('grp').fw_rmsd_med.describe())
print(O.groupby('grp').fw_rmsd_sabr_med.describe())
print(O.groupby('grp').sabr_s.describe())
print('non-amyloid chains with fw RMSD > 4 A:'); print(O[(~O.grp.str.startswith('amyloid'))&((O.fw_rmsd_med>4)|(O.fw_rmsd_sabr_med>4))].to_string())
print(O[O.grp.str.startswith('amyloid')].groupby('pdb')[['fw_rmsd_med','fw_rmsd_sabr_med','sabr_s','n_fw']].median())
print('max over clean chains: ANARCI-corr',O[O.grp=='clean'].fw_rmsd_med.max(),' SAbR-corr',O[O.grp=='clean'].fw_rmsd_sabr_med.max())
print('non-amyloid error chains with SAbR-corr RMSD > 4:',(O[(O.grp=='error (non-amyloid)')].fw_rmsd_sabr_med>4).sum(),'of',(O.grp=='error (non-amyloid)').sum())
print('amyloid chains with SAbR-corr RMSD > 10:',(O[O.grp.str.startswith('amyloid')].fw_rmsd_sabr_med>10).sum(),'of',O.grp.str.startswith('amyloid').sum())
