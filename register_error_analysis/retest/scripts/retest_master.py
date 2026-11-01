import json, re, collections, numpy as np, pandas as pd, sys
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CH=pd.read_pickle(f'{SP}/sabr/out/chains_flags.pkl')
recs={}
for f in ['validation','snapshot_rest','new','validation_multi','snapshot_rest_multi','new_multi']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r=json.loads(l)
        if r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r['residues']
# ---------- local map quality over V residues
loc={}
for l in open(f'{SP}/retest/local_all.jsonl'):
    r=json.loads(l); f=r['feats']; key='RSCC' if f.get('RSCC') else ('Q_SCORE' if f.get('Q_SCORE') else None)
    if key: loc[(r['pdb'],r['chain'])]={int(k):v for k,v in f[key].items()}
def vq(p,c):
    v=loc.get((p,c)); res=recs.get((p,c))
    if not v or not res: return np.nan
    vals=[v[x[3]+1] for x in res if x[4] is not None and x[3] is not None and (x[3]+1) in v]
    return np.mean(vals) if len(vals)>=30 else np.nan
CH['lq']=[vq(p,c) for p,c in zip(CH.pdb,CH.chain)]
CH['lq_z']=CH.groupby(['method','rb']).lq.transform(lambda x:(x-x.mean())/x.std())
# ---------- modelling metadata
mod={}
for l in open(f'{SP}/retest/modelling.jsonl'):
    e=json.loads(l); mod[e['rcsb_id'].lower()]=e
AUX=re.compile(r'nanobody[- ]?35|\bnb[- ]?35\b|scfv[- ]?16|\bfv[- ]?16|svfv16|\bbril\b|bag2|nabfab|anti[- ]fab|fab[- ]specific|megabod|legobod|sybod|\bsab\b|synthetic|anti[- ]?gfp|alfa',re.I)
def irm_type(e):
    m=e.get('pdbx_initial_refinement_model') or []
    if not m: return 'none recorded'
    t={(x.get('type') or '').lower() for x in m}; s={(x.get('source_name') or '').lower() for x in m}
    if any('alphafold' in x for x in s): return 'AlphaFold model'
    if any('in silico' in x for x in t): return 'other in-silico model'
    return 'experimental (PDB) model'
def fitting(e):
    p={(x.get('ref_protocol') or '').upper() for x in (e.get('em_3d_fitting') or [])}
    if not p: return 'not recorded'
    if 'AB INITIO MODEL' in p: return 'ab initio'
    if 'FLEXIBLE FIT' in p: return 'flexible fit'
    if 'RIGID BODY FIT' in p: return 'rigid body only'
    return 'other'
def soft(e):
    names=[(x.get('name') or '') for x in (e.get('software') or [])]+[(x.get('name') or '') for x in (e.get('em_software') or []) if (x.get('category') or '') in ('MODEL FITTING','MODEL REFINEMENT','OTHER')]
    t=' '.join(names).lower()
    return {k:int(bool(re.search(p,t))) for k,p in [('ModelAngelo','modelangelo'),('ISOLDE','isolde'),('Rosetta','rosetta'),('Coot','coot'),('Phenix','phenix'),('Namdinator','namdinator'),('REFMAC','refmac'),('BUSTER','buster')]}
def revised(e):
    return int(any(g['group']=='Atomic model' and g['revision_ordinal']>1 for g in (e.get('pdbx_audit_revision_group') or [])))
def chain_desc(e):
    d={}
    for pe in e.get('polymer_entities') or []:
        for c in pe['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []: d[c]=pe['rcsb_polymer_entity']['pdbx_description'] or ''
    return d
rows=[]
for p,c in zip(CH.pdb,CH.chain):
    e=mod.get(p,{}); s=soft(e) if e else {}
    rows.append(dict(irm=irm_type(e) if e else None,fit=fitting(e) if e else None,atom_revised=revised(e) if e else None,
        chain_desc=chain_desc(e).get(c,'') if e else '',**{('sw_'+k):v for k,v in s.items()}))
CH=pd.concat([CH.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
CH['aux_reagent']=CH.chain_desc.str.contains(AUX)
CH['hold_days']=(pd.to_datetime(CH.release)-pd.to_datetime(CH.deposit)).dt.days
# ---------- entries per paper (PDB-wide)
cnt=collections.Counter()
for l in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    e=json.loads(l); c=e.get('rcsb_primary_citation') or {}
    k=c.get('pdbx_database_id_DOI') or (('PMID'+str(c['pdbx_database_id_PubMed'])) if c.get('pdbx_database_id_PubMed') else None)
    if k: cnt[k]+=1
CH['entries_per_paper']=[cnt.get(k,1) if isinstance(k,str) and not re.fullmatch(r'[0-9a-z]{4}',k) else 1 for k in CH.cite_key]
# ---------- target novelty / competition (antigen = rarest non-antibody UniProt)
idx={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    f=l.split('\t')
    if len(f)<6: continue
    try: d=pd.to_datetime(f[2],format='%m/%d/%y')
    except Exception: continue
    auth=[a.strip() for a in re.split(r'(?<=\.),\s*',f[5]) if a.strip()]; idx[f[0].lower()]=(d,auth[-1] if auth else None)
up=pd.read_csv('/tmp/supp/pdbwide/pdb_chain_uniprot.tsv.gz',sep='\t',comment='#',usecols=['PDB','SP_PRIMARY']).drop_duplicates()
up['date']=up.PDB.map(lambda p: idx.get(p,(pd.NaT,None))[0]); up['last']=up.PDB.map(lambda p: idx.get(p,(None,None))[1]); up=up.dropna(subset=['date'])
ug=up.groupby('SP_PRIMARY'); u_first=ug.date.min(); u_n=ug.size(); u_entries={k:g[['PDB','date','last']].values for k,g in ug}
abch=set(zip(CH.pdb,CH.chain))
ent_target={}
for p,e in mod.items():
    unis=[]
    for pe in e.get('polymer_entities') or []:
        cs=pe['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []
        if any((p,c) in abch for c in cs): continue
        unis+= [u for u in (pe['rcsb_polymer_entity_container_identifiers']['uniprot_ids'] or []) if u in u_n.index]
    if unis: ent_target[p]=sorted(set(unis),key=lambda u:u_n[u])[0]
def comp(p,date):
    u=ent_target.get(p)
    if u is None: return (np.nan,np.nan)
    arr=u_entries[u]; last=idx.get(p,(None,None))[1]
    dt=np.array([(x-date).days for x in arr[:,1]]); m=(np.abs(dt)<=183)&(arr[:,0]!=p)
    others={a for a,mm in zip(arr[:,2],m) if mm and a and a!=last}
    return (len(others),int((date-u_first[u]).days<=365))
cc=[comp(p,pd.Timestamp(d)) for p,d in zip(CH.pdb,CH.deposit)]
CH['competitors']=[x[0] for x in cc]; CH['novel_target']=[x[1] for x in cc]
# ---------- CDR lengths (single-domain chains)
ac=json.load(open(f'{SP}/sabr/anarci_cache.json'))
def cdr_len(s):
    d=ac.get(s) or []
    if len(d)!=1: return (np.nan,np.nan)
    nums=[q[1] for q in d[0]['pos']]
    return (sum(27<=n<=38 for n in nums),sum(105<=n<=117 for n in nums))
cl=[cdr_len(s) for s in CH.seq]; CH['cdr1_len']=[x[0] for x in cl]; CH['cdr3_len']=[x[1] for x in cl]
CH.drop(columns=['segs']).to_pickle(f'{SP}/retest/master.pkl'); CH[['pdb','chain','segs']].to_pickle(f'{SP}/retest/segs.pkl')
print(len(CH),'V domains/chains; lq available',CH.lq.notna().mean().round(3)); print(CH.irm.value_counts(dropna=False).to_string()); print(CH.fit.value_counts(dropna=False).to_string())
print(CH[[c for c in CH.columns if c.startswith('sw_')]].mean().round(3).to_string()); print('aux reagent chains',CH.aux_reagent.sum(),'| target found',CH.competitors.notna().mean().round(3))
