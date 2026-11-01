import json, sys, pandas as pd
sys.path.insert(0,'/tmp/scripts'); from abrole import classify
an=json.load(open('/tmp/supp/meta2/anarci_all.json'))
def abtype(s):
    t={d['chain_type'] for d in an.get(s,[])}
    return ('H' in t or 'K' in t or 'L' in t), bool(t & {'A','B','G','D'})
# --- EM all years
rows=[]
for l in open('/tmp/supp/meta2/em_meta.jsonl'):
    e=json.loads(l); ents=[]; nab=0; formats=set()
    for p in e['polymer_entities'] or []:
        if p['entity_poly']['type']!='polypeptide(L)': continue
        s=(p['entity_poly']['pdbx_seq_one_letter_code_can'] or '').replace('\n','')
        ab,tcr=abtype(s)
        orgs=' '.join(o['scientific_name'] or '' for o in (p['rcsb_entity_source_organism'] or []))
        ents.append(dict(desc=p['rcsb_polymer_entity']['pdbx_description'],ab=ab,tcr=tcr,length=len(s),orgs=orgs,
                         uniprot=(p['rcsb_polymer_entity_container_identifiers']['uniprot_ids'] or []),
                         nchains=len(p['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or [])))
        if ab:
            nab+=1; nd=len(an[s])
            ct={d['chain_type'] for d in an[s]}
            formats.add('scFv' if nd>=2 and ('H' in ct and (ct&{'K','L'})) else ('VHH/nanobody' if ('H' in ct and len(s)<=160) else ('Fab/IgG chain' )))
    kw=(e['struct_keywords'] or {}).get('pdbx_keywords') or ''
    role=classify(e['struct']['title'],kw,ents)
    cit=e['rcsb_primary_citation'] or {}
    res=(e['rcsb_entry_info']['resolution_combined'] or [None])[0]
    rows.append(dict(pdb=e['rcsb_id'].lower(),deposit=e['rcsb_accession_info']['deposit_date'][:10],
        release=(e['rcsb_accession_info']['initial_release_date'] or '')[:10],res=res,title=e['struct']['title'],kw=kw,
        role=role,n_ab_entities=nab,formats='+'.join(sorted(formats)),
        journal=cit.get('journal_abbrev'),doi=cit.get('pdbx_database_id_DOI'),
        other_desc='; '.join(x['desc'] or '' for x in ents if not x['ab'])[:200],
        ab_desc='; '.join(x['desc'] or '' for x in ents if x['ab'])[:200],
        mw=e['rcsb_entry_info']['molecular_weight'],
        antigen_kda=sum(0.11*x['length']*x['nchains'] for x in ents if not x['ab'] and not x['tcr']),
        n_other_entities=sum(1 for x in ents if not x['ab'] and not x['tcr'] and x['length']>=30),
        last_author=(sorted(e['audit_author'] or [],key=lambda a:a['pdbx_ordinal'])[-1]['name'] if e['audit_author'] else None)))
em=pd.DataFrame(rows); em.to_pickle('/tmp/analysis/em_all_roles.pkl')
print(em.role.value_counts(dropna=False))
