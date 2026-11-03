# Verification of "capsid_target" (antigen_competition lens): extract per-entry auxiliary data
# - antibody-entity source organisms (bovine detection), citation author lists, audit authors
# Reads the investigator's GraphQL cache read-only; writes a compact table to this verify directory.
import json, re
import pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C = f'{SP}/deeper/cache/antigen_competition'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_integrity'
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain'])
want = set(ent.pdb)
abchains = ch.groupby('pdb').chain.apply(set).to_dict()
rows = []
for l in open(f'{C}/gql_entries.jsonl'):
    e = json.loads(l)
    p = e['rcsb_id'].lower()
    if p not in want:
        continue
    cit = e.get('rcsb_primary_citation') or {}
    aa = [a['name'] for a in sorted(e.get('audit_author') or [], key=lambda x: x.get('pdbx_ordinal') or 0)]
    ab_orgs = set()
    for pe in e.get('polymer_entities') or []:
        auth = set((pe.get('rcsb_polymer_entity_container_identifiers') or {}).get('auth_asym_ids') or [])
        if auth & abchains.get(p, set()):
            for o in pe.get('rcsb_entity_source_organism') or []:
                ab_orgs.add(o.get('ncbi_scientific_name') or '')
    rows.append(dict(pdb=p, audit_authors='; '.join(aa), cit_authors='; '.join(cit.get('rcsb_authors') or []),
                     ab_orgs='; '.join(sorted(ab_orgs))))
X = pd.DataFrame(rows).drop_duplicates('pdb')
X.to_csv(f'{OUT}/entry_aux.csv.gz', index=False)
print('entries with aux', len(X), 'of', len(want))
print(X.ab_orgs.value_counts().head(25).to_string())
