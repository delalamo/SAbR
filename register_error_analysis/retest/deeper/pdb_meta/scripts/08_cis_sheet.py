"""Map RCSB cis-peptide and PROMOTIF sheet annotations of every V chain onto ANARCI IMGT positions (sequence-based,
i.e. the correct register); compare EM error chains with clean chains:
 - cis peptides at canonical positions (by IMGT position and residue type) and in the constant domain;
 - sheet fraction in the V domain away from the error run (>5 residues) and in the constant domain."""
import json, sys, collections, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature, bh, ybin
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz')
ch=ch[~ch.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)]
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
key=set(zip(ch.pdb,ch.chain))
imgt={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l)
        if (r['pdb'],r['chain']) not in key or (r['pdb'],r['chain']) in imgt: continue
        imgt[(r['pdb'],r['chain'])]={x[3]:(x[4],x[5],x[2]) for x in r.get('residues') or [] if x[3] is not None}
feat={}
for l in open(f'{SP}/deeper/cache/pdb_meta/cis_sheet.jsonl'):
    x=json.loads(l); pdb=x['rcsb_id'].split('.')[0].lower(); feat[(pdb,x['auth'])]=x['feats']
# error-run seq-index spans per chain (seq index i0..i1 as in runs file, assumed 0-based seq index)
rs=collections.defaultdict(list)
for _,r in runs.iterrows(): rs[(r.pdb,r.chain)].append((r.i0,r.i1))
rows=[]
for _,c in ch.iterrows():
    k=(c.pdb,c.chain); m=imgt.get(k); fs=feat.get(k)
    if m is None or fs is None: continue
    seq=c.seq if isinstance(c.seq,str) else ''
    vidx=[i for i,(a,ins,rn) in m.items() if a is not None]
    if not vidx: continue
    vmax=max(vidx)
    cis=[]; sheet=set(); unobs=set()
    for f in fs:
        for p in f['feature_positions']:
            b,e=p['beg_seq_id'],p.get('end_seq_id') or p['beg_seq_id']
            if f['type']=='CIS-PEPTIDE': cis.append(e-1)
            elif f['type']=='SHEET': sheet.update(range(b-1,e))
            else: unobs.update(range(b-1,e))
    modeled=set(range(len(seq)))-unobs  # residues present in the model (SEQRES minus RCSB unobserved)
    near=set()
    for i0,i1 in rs.get(k,[]):
        near.update(range(int(i0)-5,int(i1)+6))
    vres=[i for i in vidx if i not in near]
    cres=[i for i in modeled if i>vmax+3]
    r={'pdb':c.pdb,'chain':c.chain,'ctype':c.ctype,'method':c.method,'res':c.res,'deposit':c.deposit,'paper':c.paper,
       'err_first':c.err_first,'err_any':c.err_any,'modes':c.err_modes,
       'v_sheet_away':np.mean([i in sheet for i in vres]) if len(vres)>40 else np.nan,
       'const_sheet':np.mean([i in sheet for i in cres]) if len(cres)>40 else np.nan,
       'n_const_modeled':len(cres),
       'v_cis_n':sum(1 for i in cis if i in m and m[i][0] is not None),
       'const_cis_n':sum(1 for i in cis if i>vmax+3)}
    # canonical cis positions: record (imgt, residue) of each cis in V
    r['v_cis_pos']=';'.join(f"{m[i][0]}{m[i][1] or ''}{(m[i][2] or '')[:3]}" for i in cis if i in m and m[i][0] is not None)
    # prolines present in V by IMGT position (to compute per-position cis rate)
    r['v_pro_pos']=';'.join(f"{a}{ins or ''}" for i,(a,ins,rn) in m.items() if a is not None and rn=='PRO')
    r['const_pro_n']=sum(1 for i in cres if i<len(seq) and seq[i]=='P')
    r['const_cis_pos']=';'.join(str(i-vmax) for i in cis if i>vmax+3)
    rows.append(r)
D=pd.DataFrame(rows); D.to_csv(f'{OUT}/chain_cis_sheet.csv.gz',index=False)
print(len(D),'chains mapped')
# canonical positions: among clean X-ray + EM chains, IMGT positions where Pro is cis most often
for ct in ['H','K','L']:
    x=D[(D.ctype==ct)&(~D.err_any)]
    pro=collections.Counter(p for s in x.v_pro_pos.dropna() for p in s.split(';') if p)
    cis=collections.Counter(p[:-3] for s in x.v_cis_pos.dropna() for p in s.split(';') if p.endswith('PRO'))
    top=[(p,cis[p],pro[p],cis[p]/pro[p]) for p in cis if pro[p]>=50]
    print(ct,'top cis-Pro IMGT positions (clean chains):',sorted(top,key=lambda t:-t[1])[:5])
