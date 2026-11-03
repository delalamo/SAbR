# Extract, for the focal entries, (a) antibody-entity source organisms (by matching V chains to entities),
# (b) audit and citation author lists, from the antigen_competition lens's GraphQL cache (read-only).
import json, gzip, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C=f'{SP}/deeper/cache/antigen_competition'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_confounding'
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain'])
want=ch.groupby('pdb').chain.apply(set).to_dict()
rows=[]; auth=[]
seen=set()
for l in open(f'{C}/gql_entries.jsonl'):
    e=json.loads(l); p=e['rcsb_id'].lower()
    if p not in want or p in seen: continue
    seen.add(p)
    cit=e.get('rcsb_primary_citation') or {}
    aa=[a['name'] for a in sorted(e.get('audit_author') or [],key=lambda x:x.get('pdbx_ordinal') or 0)]
    auth.append(dict(pdb=p,audit='|'.join(aa),cit_authors='|'.join(cit.get('rcsb_authors') or [])))
    for pe in e.get('polymer_entities') or []:
        ci=pe['rcsb_polymer_entity_container_identifiers']
        for a in ci.get('auth_asym_ids') or []:
            if a in want[p]:
                org=';'.join(sorted({o.get('ncbi_scientific_name') or '' for o in (pe.get('rcsb_entity_source_organism') or [])}))
                rows.append(dict(pdb=p,chain=a,ab_org=org,entity_desc=(pe.get('rcsb_polymer_entity') or {}).get('pdbx_description')))
pd.DataFrame(rows).to_csv(f'{SP}/deeper/cache/verify_capsid_confounding/chain_ab_org.csv',index=False)
pd.DataFrame(auth).to_csv(f'{SP}/deeper/cache/verify_capsid_confounding/entry_authors.csv',index=False)
print(len(rows),len(auth))
