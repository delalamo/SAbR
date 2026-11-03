"""Geometric arbiter for mode-A error chains: after superposing on the framework core (IMGT 39-55, 66-104) onto
high-resolution X-ray references (1N8Z, 12E8, 1AQK), do the residues in the N-terminal disagreement run sit at the
reference CA of their ANARCI (sequence) position, or of their SAbR (structure) position?  Closer to the SAbR position
= the built strand really is out of register (a genuine modelling error, not a detector artifact)."""
import os, gzip, json, urllib.request, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser
from anarci import run_anarci
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_SN1'; os.makedirs(f'{CACHE}/cif',exist_ok=True)
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
AA3=dict(ALA='A',ARG='R',ASN='N',ASP='D',CYS='C',GLN='Q',GLU='E',GLY='G',HIS='H',ILE='I',LEU='L',LYS='K',MET='M',PHE='F',PRO='P',SER='S',THR='T',TRP='W',TYR='Y',VAL='V')
def dl(pdb):
    for d in (f'{SP}/sabr/dl/{pdb}.cif',f'{CACHE}/cif/{pdb}.cif'):
        if os.path.exists(d): return d
    fn=f'{CACHE}/cif/{pdb}.cif'
    for k in range(3):
        try:
            open(fn,'wb').write(gzip.decompress(urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb.upper()}.cif.gz',timeout=120).read())); return fn
        except Exception: time.sleep(1+2**k)
    return None
P=MMCIFParser(QUIET=True)
def ref_coords(pdb,chain,ct):
    st=P.get_structure(pdb,dl(pdb)); ch=st[0][chain]
    res=[r for r in ch if r.id[0]==' ' and r.get_resname() in AA3 and 'CA' in r]
    seq=''.join(AA3[r.get_resname()] for r in res)
    _,num,det,_=run_anarci([('r',seq)],scheme='imgt',allow={'H','K','L'})
    for (nb,s,e),d in zip(num[0],det[0]):
        if (d['chain_type'] in 'KL')==(ct in 'KL'):
            out={};i=s
            for (p,ins),a in nb:
                if a=='-': continue
                out[(p,ins.strip())]=res[i]['CA'].coord; i+=1
            return out
REFS={'H':[('1n8z','B'),('12e8','H'),('1aqk','H')],'K':[('1n8z','A'),('12e8','L')],'L':[('1aqk','L')]}
RC={ct:[ref_coords(p,c,ct) for p,c in v] for ct,v in REFS.items()}
for ct in RC: print(ct,[sorted(k[0] for k in r)[:3] for r in RC[ct]])
CORE=[(p,'') for p in list(range(39,56))+list(range(66,105))]
def kabsch(A,B):
    ca,cb=A.mean(0),B.mean(0); H=(A-ca).T@(B-cb); U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,d])@U.T; return lambda X:(X-ca)@R.T+cb
REC={}
D=pd.read_csv(f'{OUT}/units_modeA_analysis.csv.gz',low_memory=False)
cases=D[D.y==1][['unit','rA_pdb','rA_chain','ctype','seq_trunc','x_model_first_anarci']]
need=set(zip(cases.rA_pdb,cases.rA_chain))
# clean truncated controls as a sanity sample: ANARCI and SAbR agree there, so both distances are equal by construction;
# instead use them to estimate typical N-terminal residual distance
for fn in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r=json.loads(line)
        if (r['pdb'],r['chain']) in need and r.get('residues'): REC[(r['pdb'],r['chain'])]=r['residues']
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); runs=runs[runs['mode']=='A_FR1Nterm']
rows=[]
for c in cases.itertuples():
    key=(c.rA_pdb,c.rA_chain); res=REC.get(key); fn=dl(c.rA_pdb)
    if not res or not fn: rows.append(dict(pdb=key[0],chain=key[1],status='missing')); continue
    st=P.get_structure(key[0],fn); ch=st[0][key[1]]
    ca={}
    for r in ch:
        if r.id[0]==' ' and 'CA' in r: ca[(r.id[1],r.id[2].strip())]=r['CA'].coord
    rr=runs[(runs.pdb==key[0])&(runs.chain==key[1])].iloc[0]
    A={};S={};win=[]
    for x in res:
        k=(x[0],x[1] or '')
        if k not in ca or x[4] is None: continue
        A[(x[4],x[5] or '')]=ca[k]
        if x[6] is not None: S[(x[6],x[7] or '')]=ca[k]
        if rr.imgt0<=x[4]<=rr.imgt1 and x[4]<=40 and ((x[4],x[5] or '')!=(x[6],x[7] or '')): win.append(((x[4],x[5] or ''),(x[6],x[7] or ''),ca[k]))
    out=dict(pdb=key[0],chain=key[1],ctype=c.ctype,seq_trunc=c.seq_trunc,model_first=c.x_model_first_anarci,run=f'{rr.imgt0}-{rr.imgt1}',delta=rr.delta,n_win=len(win))
    dA=[];dS=[];rmsA=[];rmsS=[]
    for ref in RC[c.ctype]:
        for arm,M,dl_,rl in (('A',A,dA,rmsA),('S',S,dS,rmsS)):
            com=[p for p in CORE if p in M and p in ref]
            if len(com)<30: continue
            f=kabsch(np.array([M[p] for p in com]),np.array([ref[p] for p in com]))
            rl.append(np.sqrt(np.mean(np.sum((f(np.array([M[p] for p in com]))-np.array([ref[p] for p in com]))**2,1))))
            ds=[np.linalg.norm(f(cc[None])[0]-ref[(pa if arm=='A' else ps)]) for pa,ps,cc in win if (pa if arm=='A' else ps) in ref and pa in ref and ps in ref]
            if ds: dl_.append(np.mean(ds))
    out.update(core_rmsd_A=np.median(rmsA) if rmsA else np.nan,core_rmsd_S=np.median(rmsS) if rmsS else np.nan,
               win_dist_anarci=np.median(dA) if dA else np.nan,win_dist_sabr=np.median(dS) if dS else np.nan,status='ok')
    rows.append(out)
T=pd.DataFrame(rows)
T['verdict']=np.where(T.win_dist_sabr+0.5<T.win_dist_anarci,'model shifted (SAbR fits)',np.where(T.win_dist_anarci+0.5<T.win_dist_sabr,'ANARCI fits (detector artifact?)','ambiguous'))
T.to_csv(f'{OUT}/v08_geometry_cases.csv',index=False)
pd.set_option('display.width',250)
print(T.round(2).to_string())
print(pd.crosstab(T.seq_trunc,T.verdict))
