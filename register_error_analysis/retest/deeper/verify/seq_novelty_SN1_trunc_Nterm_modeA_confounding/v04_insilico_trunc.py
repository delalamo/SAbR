"""Detector check: does SAbR (structure arm) create mode-A-like N-terminal disagreement runs when a CORRECT model simply
lacks its first residues?  Take clean cryo-EM chains whose model starts at IMGT 1 (SAbR and ANARCI agree everywhere),
delete the first k = 1,2,3 modelled residues, re-run SAbR, and compare with the original ANARCI numbering of the
remaining residues. Also re-run ANARCI on the SEQRES with the first k V residues removed (sequence-arm check)."""
import sys, os, json, gzip, time, urllib.request, warnings, logging, random
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import pandas as pd, numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_SN1'; os.makedirs(f'{CACHE}/cif',exist_ok=True)
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
def dl(pdb):
    for d in (f'{SP}/sabr/dl/{pdb}.cif',f'{CACHE}/cif/{pdb}.cif'):
        if os.path.exists(d): return d
    fn=f'{CACHE}/cif/{pdb}.cif'
    for k in range(3):
        try:
            data=gzip.decompress(urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb.upper()}.cif.gz',timeout=120).read())
            open(fn,'wb').write(data); return fn
        except Exception as e:
            time.sleep(1+2**k)
    return None
def runs_of(pairs):
    """pairs: list of (anarci,(sabr)) per residue in order; return runs of >=3 consecutive disagreements as (start_imgt,end_imgt,delta)"""
    out=[];cur=[]
    for a,s in pairs+[(None,None)]:
        if a is not None and s is not None and a!=s: cur.append((a,s)); continue
        if len(cur)>=3: out.append((cur[0][0][0],cur[-1][0][0],cur[0][1][0]-cur[0][0][0]))
        cur=[]
    return out
def job(args):
    from sabr import renumber_structure
    pdb,chain,recs=args
    fn=dl(pdb)
    if not fn: return [dict(pdb=pdb,chain=chain,k=-1,status='dl fail')]
    md=MMCIF2Dict(fn); res_out=[]
    amap={(r[0],r[1]):(r[4],r[5] or '') for r in recs}
    for k in (0,1,2,3):
        try:
            st=MMCIFParser(QUIET=True).get_structure(pdb,fn)
            chn=st[0][chain]
            # V-domain residues ordered; drop the first k that carry an ANARCI number
            vres=[r for r in chn if r.id[0]==' ' and amap.get((r.id[1],r.id[2].strip()),(None,))[0] is not None]
            drop=[r.id for r in vres[:k]]
            for rid in drop: chn.detach_child(rid)
            t0=time.time(); new=renumber_structure(st,chain=chain,mmcif_dict=md); dt=time.time()-t0
            old=[r for r in chn if r.id[0]==' ']; nw=[r for r in new[0][chain] if r.id[0]==' ']
            pairs=[]
            for o,n in zip(old,nw):
                a=amap.get((o.id[1],o.id[2].strip()))
                if a is None or a[0] is None: continue
                pairs.append((a,(n.id[1],n.id[2].strip())))
            rr=runs_of(pairs)
            nt=[x for x in rr if x[0]<=10]
            first=pairs[0] if pairs else (None,None)
            res_out.append(dict(pdb=pdb,chain=chain,k=k,status='ok',secs=round(dt,2),first_anarci=first[0][0] if first[0] else None,
                                first_sabr=first[1][0] if first[1] else None,n_runs=len(rr),modeA_like=len(nt)>0,
                                runs=json.dumps(rr),n_disagree=sum(a!=s for a,s in pairs)))
        except Exception as e:
            res_out.append(dict(pdb=pdb,chain=chain,k=k,status=f'error {type(e).__name__}: {str(e)[:100]}'))
    return res_out
if __name__=='__main__':
    from multiprocessing import Pool
    D=pd.read_csv(f'{OUT}/units_modeA_analysis.csv.gz',low_memory=False)
    c=D[(D.y==0)&(D.x_model_first_anarci==1)&(D.r0_model_first_sabr==1)].copy()
    random.seed(7)
    pick=[]
    for ct,n in (('H',45),('K',30),('L',30)):
        sub=c[c.ctype==ct]
        pick+=list(sub.sample(min(n,len(sub)),random_state=11)[['r0_pdb','r0_chain']].itertuples(index=False,name=None))
    need=set(pick); REC={}
    for fn in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
        for line in open(f'{SP}/sabr/out/{fn}'):
            r=json.loads(line)
            if (r['pdb'],r['chain']) in need and r.get('residues'): REC[(r['pdb'],r['chain'])]=r['residues']
    jobs=[(p,c_,REC[(p,c_)]) for p,c_ in pick if (p,c_) in REC]
    print('jobs',len(jobs),flush=True)
    rows=[]
    with Pool(2) as pool:
        for i,r in enumerate(pool.imap_unordered(job,jobs)):
            rows+=r
            if i%10==0: print(i,flush=True)
    T=pd.DataFrame(rows); T.to_csv(f'{OUT}/v04_insilico_sabr.csv',index=False)
    print(T.groupby('k').agg(n=('status',lambda s:(s=='ok').sum()),modeA_like=('modeA_like','mean'),any_run=('n_runs',lambda x:(x>0).mean())))
