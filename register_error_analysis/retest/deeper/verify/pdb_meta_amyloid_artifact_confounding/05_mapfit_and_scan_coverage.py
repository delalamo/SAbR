"""(1) Local map fit (lq_z: per-chain mean Q-score over V residues, z-scored within method x resolution bin) of amyloid
chains vs clean and non-amyloid error chains, one value per entry (median over chains); real register errors fit worse.
(2) AL-fibril entries at 3-5 A in the whole PDB vs the analysis set (scan coverage)."""
import json, numpy as np, pandas as pd
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_confounding'
c=pd.read_csv(f'{SP}/deeper/chains.csv.gz'); c['amy']=c.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)
em=c[c.method=='EM'].copy()
em['grp']=np.where(em.amy,'amyloid',np.where(em.err_first,'error (non-amyloid)',np.where(~em.err_any,'clean','copied/artifact only')))
pe=em.groupby(['pdb','grp']).agg(lq_z=('lq_z','median'),res=('res','first')).reset_index()
rows=[]
for g in ['amyloid','error (non-amyloid)']:
    a=pe[pe.grp==g].lq_z.dropna(); b=pe[pe.grp=='clean'].lq_z.dropna()
    rows.append(dict(test=f'entry median lq_z: {g} vs clean (EM)',err_val=round(a.median(),3),ctrl_val=round(b.median(),3),n_err=len(a),n_ctrl=len(b),p=stats.mannwhitneyu(a,b).pvalue))
# restrict clean to same resolution range 3.0-3.9
b=pe[(pe.grp=='clean')&(pe.res<=3.9)].lq_z.dropna(); a=pe[pe.grp=='amyloid'].lq_z.dropna()
rows.append(dict(test='entry median lq_z: amyloid vs clean, clean restricted to 3.0-3.9 A',err_val=round(a.median(),3),ctrl_val=round(b.median(),3),n_err=len(a),n_ctrl=len(b),p=stats.mannwhitneyu(a,b).pvalue))
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/mapfit_tests.csv',index=False); print(R.to_string())
# scan coverage: AL fibrils in PDB at 3-5 A (EM)
idx=pd.read_csv('/tmp/supp/pdbwide/entries.idx',sep='\t',skiprows=2,header=None,usecols=[0,2,3,6,7],names=['pdb','dep','title','res','exp'],dtype=str,on_bad_lines='skip',quoting=3)
idx['res']=pd.to_numeric(idx.res,errors='coerce')
t=idx.title.fillna('')
al=idx[t.str.contains('amyloid|fibril',case=False)&t.str.contains(r'light chain|lambda|kappa|\bAL\b|AL[0-9]|immunoglobulin|amyloidosis patient|myeloma patient',case=False)
       &idx.exp.fillna('').str.contains('ELECTRON')&idx.res.between(3,5)].copy()
al['pdb']=al.pdb.str.lower()
al=al[~al.title.str.contains('ATTR|transthyretin|apolipoprotein|ALECT2|serum amyloid|tau|synuclein|beta-2|wild-type amyloidosis',case=False)]
inset=set(c.pdb)
scanned={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        d=json.loads(l)
        if d['pdb'] in set(al.pdb):
            v=[x for x in d['residues'] if x[4] is not None]
            dis=sum(1 for x in v if (x[4],x[5])!=(x[6],x[7]))
            scanned.setdefault(d['pdb'],[]).append((len(v),dis))
al['in_analysis_set']=al.pdb.isin(inset)
al['sabr_scanned']=al.pdb.isin(scanned)
al['n_V_res_modelled']=[max(x[0] for x in scanned[p]) if p in scanned else np.nan for p in al.pdb]
al['frac_V_res_SAbR_ne_ANARCI']=[round(np.mean([x[1]/x[0] for x in scanned[p]]),2) if p in scanned else np.nan for p in al.pdb]
al.to_csv(f'{OUT}/al_fibril_scan_coverage.csv',index=False)
pd.set_option('display.width',250); print(al.to_string())
# disagreement fraction for comparison: non-amyloid error chains and clean chains
fr=[]
grp=dict(zip(zip(c.pdb,c.chain),np.where(c.amy,'amyloid',np.where(c.err_first,'error (non-amyloid)',np.where(~c.err_any,'clean','copied/artifact only')))))
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        d=json.loads(l); k=(d['pdb'],d['chain'])
        if k not in grp or d.get('status')!='ok' or 'residues' not in d: continue
        v=[x for x in d['residues'] if x[4] is not None]
        if not v: continue
        fr.append(dict(grp=grp[k],frac=sum(1 for x in v if (x[4],x[5])!=(x[6],x[7]))/len(v)))
FR=pd.DataFrame(fr).groupby('grp').frac.describe(percentiles=[.5,.9,.99]); print(FR); FR.to_csv(f'{OUT}/disagreement_fraction_by_group.csv')
