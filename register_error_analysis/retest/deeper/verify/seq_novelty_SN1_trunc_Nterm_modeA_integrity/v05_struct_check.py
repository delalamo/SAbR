"""Structure-only sanity check of the mode-A case labels (is the N-terminal strand really shifted, or is it a numbering artifact
at a truncated terminus?). For each mode-A case chain: superpose the ImmuneBuilder prediction (built from sequence, IMGT-numbered)
onto the deposited model using framework Calphas outside FR1 (iterative outlier rejection), then for residues in the N-terminal run
compare the model Calpha to the predicted Calpha at the sequence (ANARCI) position vs at SAbR's position."""
import json, os, gzip, urllib.request, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser, PDBParser
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_SN1/cif'; os.makedirs(CACHE,exist_ok=True)
PRED=f'{SP}/pred/full'
D=pd.read_csv('v02_modeA_case_detail.csv')
units=pd.DataFrame(json.load(open(f'{PRED}/units.json')))
fvs={f['fv_id']:f for f in json.load(open(f'{PRED}/fvs.json'))}
rec={}
want=set(zip(D.pdb,D.chain))
for f in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        d=json.loads(l)
        if (d['pdb'],d['chain']) in want and d.get('status')=='ok': rec[(d['pdb'],d['chain'])]=d['residues']
def getcif(pdb):
    for fn in (f'{SP}/sabr/dl/{pdb}.cif',f'{CACHE}/{pdb}.cif'):
        if os.path.exists(fn): return fn
    fn=f'{CACHE}/{pdb}.cif'
    data=gzip.decompress(urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb.upper()}.cif.gz',timeout=120).read())
    open(fn,'wb').write(data); return fn
def kabsch(P,Q):
    pc,qc=P.mean(0),Q.mean(0); H=(Q-qc).T@(P-pc); U_,S,Vt=np.linalg.svd(H); dd=np.sign(np.linalg.det(Vt.T@U_.T))
    R=Vt.T@np.diag([1,1,dd])@U_.T; return R,pc,qc
rows=[]
for _,c in D.iterrows():
    key=(c.pdb,c.chain); r=rec.get(key)
    u=units[(units.pdb==c.pdb)&(units.chain==c.chain)]
    if r is None or u.empty: rows.append(dict(pdb=c.pdb,chain=c.chain,status='no record/pred')); continue
    u=u.iloc[0]; pf=f'{PRED}/immunebuilder/{u.fv_id}.pdb'
    if not os.path.exists(pf): rows.append(dict(pdb=c.pdb,chain=c.chain,status='no pred file')); continue
    pch='H' if c.ctype=='H' else 'L'
    ps=PDBParser(QUIET=True).get_structure('p',pf)
    if pch not in ps[0]: rows.append(dict(pdb=c.pdb,chain=c.chain,status='pred chain missing')); continue
    pred={(res.id[1],res.id[2].strip()):res['CA'].coord for res in ps[0][pch] if 'CA' in res}
    try:
        st=MMCIFParser(QUIET=True).get_structure(c.pdb,getcif(c.pdb))
    except Exception as e:
        rows.append(dict(pdb=c.pdb,chain=c.chain,status=f'cif fail {e}')); continue
    chn=None
    for ch_ in st[0]:
        if ch_.id==c.chain: chn=ch_
    if chn is None: rows.append(dict(pdb=c.pdb,chain=c.chain,status='model chain missing')); continue
    mod={(res.id[1],res.id[2].strip()):res['CA'].coord for res in chn if 'CA' in res and res.id[0]==' '}
    # residue list: (model CA, anarci IMGT, sabr IMGT)
    L=[(mod.get((x[0],x[1])),(x[4],x[5] or ''),(x[6],x[7] or '')) for x in r if x[4] is not None]
    L=[t for t in L if t[0] is not None]
    core=[(m,a) for m,a,s in L if a==s and (39<=a[0]<=55 or 66<=a[0]<=104 or 118<=a[0]<=128) and a in pred]
    if len(core)<20: rows.append(dict(pdb=c.pdb,chain=c.chain,status='too few core')); continue
    P=np.array([m for m,a in core]); Q=np.array([pred[a] for m,a in core]); keep=np.ones(len(P),bool)
    for it in range(5):
        R,pc,qc=kabsch(P[keep],Q[keep]); Qt=(Q-qc)@R.T+pc; dev=np.linalg.norm(Qt-P,axis=1)
        newkeep=dev<max(2.0,np.percentile(dev,75)*1.5)
        if (newkeep==keep).all(): break
        keep=newkeep
    rmsd=np.sqrt((dev[keep]**2).mean())
    tr=lambda q:(q-qc)@R.T+pc
    run=[(m,a,s) for m,a,s in L if a!=s and a[0]<=26]
    dseq=[];dsab=[]
    for m,a,s in run:
        if a in pred and s in pred:
            dseq.append(np.linalg.norm(m-tr(pred[a]))); dsab.append(np.linalg.norm(m-tr(pred[s])))
    # also in-register control: FR1 residues 1-26 where they agree
    agree=[np.linalg.norm(m-tr(pred[a])) for m,a,s in L if a==s and a[0]<=26 and a in pred]
    rows.append(dict(pdb=c.pdb,chain=c.chain,ctype=c.ctype,trunc=c.trunc,delta=c.delta,status='ok',core_n=int(keep.sum()),core_rmsd=round(rmsd,2),
        n_run_cmp=len(dseq),med_d_seqpos=np.median(dseq) if dseq else np.nan,med_d_sabrpos=np.median(dsab) if dsab else np.nan,
        frac_closer_sabr=np.mean(np.array(dsab)<np.array(dseq)) if dseq else np.nan,med_d_agreeFR1=np.median(agree) if agree else np.nan))
O=pd.DataFrame(rows); O.to_csv('v05_struct_check.csv',index=False)
pd.set_option('display.width',200)
print(O.round(2).to_string())
ok=O[(O.status=='ok')&(O.n_run_cmp>=2)].copy()
ok['sabr_supported']=ok.med_d_sabrpos<ok.med_d_seqpos
print(ok.groupby('trunc').agg(n=('pdb','size'),sabr_supported=('sabr_supported','mean'),med_seq=('med_d_seqpos','median'),med_sabr=('med_d_sabrpos','median'),core_rmsd=('core_rmsd','median')))
