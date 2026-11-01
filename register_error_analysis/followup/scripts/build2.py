import json, re, sys, collections, numpy as np, pandas as pd
sys.path.insert(0,'/tmp/scripts'); from abrole import classify
A='/tmp/analysis'
ch=pd.read_pickle(f'{A}/chains.pkl')
abchain=dict(zip(zip(ch.pdb,ch.chain),ch.ab))
# ---------- PDB-wide: entries per paper
cite={}
for l in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    e=json.loads(l); c=e.get('rcsb_primary_citation') or {}
    key=c.get('pdbx_database_id_DOI') or (('PMID'+str(c['pdbx_database_id_PubMed'])) if c.get('pdbx_database_id_PubMed') else None)
    cite[e['rcsb_id'].lower()]=key
cnt=collections.Counter(v for v in cite.values() if v)
# ---------- PDB-wide dates & authors
idx={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    f=l.split('\t')
    if len(f)<6: continue
    try: d=pd.to_datetime(f[2],format='%m/%d/%y')
    except Exception: continue
    auth=[a.strip() for a in re.split(r'(?<=\.),\s*',f[5]) if a.strip()]
    idx[f[0].lower()]=(d,auth[-1] if auth else None)
up=pd.read_csv('/tmp/supp/pdbwide/pdb_chain_uniprot.tsv.gz',sep='\t',comment='#',usecols=['PDB','SP_PRIMARY']).drop_duplicates()
up['date']=up.PDB.map(lambda p: idx.get(p,(pd.NaT,None))[0]); up['last']=up.PDB.map(lambda p: idx.get(p,(None,None))[1])
up=up.dropna(subset=['date'])
ug=up.groupby('SP_PRIMARY')
u_first=ug.date.min(); u_n=ug.size()
u_entries={k:g[['PDB','date','last']].values for k,g in ug}
def competition(uni,pdb,date,last,window=183):
    arr=u_entries.get(uni)
    if arr is None: return 0,0
    dt=np.array([(x-date).days for x in arr[:,1]])
    m=(np.abs(dt)<=window)&(arr[:,0]!=pdb)
    others=set(a for a,mm in zip(arr[:,2],m) if mm and a and a!=last)
    prior=(dt< 0)&(dt>=-window)&(arr[:,2]!=last)
    return len(others), int(prior.sum())
# ---------- journal tiers
CNS={'Nature','Science','Cell'}
def tier(j):
    if not j or j.lower().startswith('to be published'): return 'Unpublished'
    if j in CNS: return 'CNS'
    if re.match(r'^(Nat |Nature )|^(Sci Adv|Sci Immunol|Sci Transl Med|Sci Signal)$|^(Mol Cell|Immunity|Neuron|Cell Rep|Cell Host Microbe|Cell Res|Cell Discov|Cell Chem Biol|Dev Cell|Cancer Cell|Cell Metab|Cell Stem Cell|Curr Biol|Structure|Med|Chem)$',j): return 'High-impact family'
    if j in ('Proc Natl Acad Sci U S A','Elife','Embo J','EMBO J','J Am Chem Soc','Plos Biol','PLoS Biol','Angew Chem Int Ed Engl'): return 'High-impact family'
    return 'Other journal'
# ---------- 3-5 A entry table
rows=[]; inst_rows=[]
for l in open('/tmp/supp/meta2/meta_35.jsonl'):
    e=json.loads(l); pdb=e['rcsb_id'].lower()
    meth=[m['method'] for m in e['exptl']]; method='EM' if 'ELECTRON MICROSCOPY' in meth else 'X-ray'
    ents=[]; sars=False; viral=False; target_unis=[]
    for p in e['polymer_entities'] or []:
        if p['entity_poly']['type']!='polypeptide(L)': continue
        chains=p['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []
        ab=any(abchain.get((pdb,c))=='antibody' for c in chains); tcr=any(abchain.get((pdb,c))=='TCR' for c in chains)
        orgs=' '.join(o['scientific_name'] or '' for o in (p['rcsb_entity_source_organism'] or []))
        tax={o['ncbi_taxonomy_id'] for o in (p['rcsb_entity_source_organism'] or [])}
        L=p['entity_poly']['rcsb_sample_sequence_length']
        unis=p['rcsb_polymer_entity_container_identifiers']['uniprot_ids'] or []
        if not ab and not tcr:
            if 2697049 in tax: sars=True
            if re.search(r'virus|viridae',orgs,re.I): viral=True
            if L>=50: target_unis+=unis
        ents.append(dict(desc=p['rcsb_polymer_entity']['pdbx_description'],ab=ab,tcr=tcr,length=L,orgs=orgs,entity=p['rcsb_id']))
        for inst in p['polymer_entity_instances'] or []:
            c=inst['rcsb_polymer_entity_instance_container_identifiers']['auth_asym_id']
            s={x['type']:(x['count'],x['coverage']) for x in inst['rcsb_polymer_instance_feature_summary'] or []}
            inst_rows.append(dict(pdb=pdb,chain=c,entity=p['rcsb_id'],ab=ab,tcr=tcr,L=L,
                clash_frac=s.get('CLASHES',(0,0))[1] or 0, angle_frac=s.get('ANGLE_OUTLIERS',(0,0))[1] or 0,
                bond_frac=s.get('BOND_OUTLIERS',(0,0))[1] or 0, unobs_frac=s.get('UNOBSERVED_RESIDUE_XYZ',(0,0))[1] or 0,
                has_q='Q_SCORE' in s, has_rscc='RSCC' in s))
    kw=(e['struct_keywords'] or {}).get('pdbx_keywords') or ''
    role=classify(e['struct']['title'],kw,ents)
    c=e['rcsb_primary_citation'] or {}
    dep=pd.to_datetime(e['rcsb_accession_info']['deposit_date'][:10]); rel=pd.to_datetime((e['rcsb_accession_info']['initial_release_date'] or '')[:10] or None)
    last=(sorted(e['audit_author'] or [],key=lambda a:a['pdbx_ordinal'])[-1]['name'] if e['audit_author'] else None)
    last_idx=idx.get(pdb,(None,None))[1]
    # target uniprot = rarest in PDB
    tu=sorted(set(target_unis),key=lambda u:u_n.get(u,0))
    tu=[u for u in tu if u in u_n.index]
    comp=prior=np.nan; novel=np.nan; first_days=np.nan; target=None
    if tu:
        target=tu[0]; comp,prior=competition(target,pdb,dep,last_idx)
        first_days=(dep-u_first[target]).days; novel=int(first_days<=365)
    # coordinate revisions after release
    hist={h['ordinal']:h for h in (e['pdbx_audit_revision_history'] or [])}
    atom_rev=any(g['group']=='Atomic model' and g['revision_ordinal']>1 for g in (e['pdbx_audit_revision_group'] or []))
    key=c.get('pdbx_database_id_DOI') or (('PMID'+str(c['pdbx_database_id_PubMed'])) if c.get('pdbx_database_id_PubMed') else None)
    rows.append(dict(pdb=pdb,method=method,res=(e['rcsb_entry_info']['resolution_combined'] or [np.nan])[0],
        deposit=dep,release=rel,hold_days=(rel-dep).days if rel is not pd.NaT else np.nan,year=dep.year,
        title=e['struct']['title'],kw=kw,role=role,journal=c.get('journal_abbrev'),tier=tier(c.get('journal_abbrev')),
        cite_key=key,entries_per_paper=cnt.get(key,1) if key else 1,last_author=last,
        sars2=sars,viral=viral,target_uniprot=target,competitors_6mo=comp,prior_other_groups_6mo=prior,
        days_since_first_target=first_days,novel_target=novel,atom_model_revised=atom_rev,
        mw=e['rcsb_entry_info']['molecular_weight'],n_entities=len(ents),
        n_ab_entities=sum(x['ab'] for x in ents)))
ent=pd.DataFrame(rows); inst=pd.DataFrame(inst_rows)
ent.to_pickle(f'{A}/entries35.pkl'); inst.to_pickle(f'{A}/instances35.pkl')
print(ent.role.value_counts(dropna=False)); print(ent.tier.value_counts()); print(ent[['competitors_6mo','entries_per_paper','hold_days','novel_target']].describe().round(2))
