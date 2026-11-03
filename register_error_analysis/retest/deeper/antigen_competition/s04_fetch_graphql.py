# Fetch entity metadata + assembly-1 interfaces for all antibody-containing PDB entries (RCSB GraphQL), cached as jsonl
import json, os, time, sys, requests, threading
from concurrent.futures import ThreadPoolExecutor
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/antigen_competition'
ents=json.load(open(f'{C}/ab_entries_pdbwide.json'))
out=f'{C}/gql_entries.jsonl'
done=set()
if os.path.exists(out):
    for l in open(out):
        try: done.add(json.loads(l)['rcsb_id'].lower())
        except Exception: pass
todo=[e for e in ents if e not in done]
print('todo',len(todo),flush=True)
Q='''{entries(entry_ids:%s){rcsb_id rcsb_accession_info{deposit_date initial_release_date}
 exptl{method} rcsb_entry_info{resolution_combined}
 audit_author{name pdbx_ordinal}
 rcsb_primary_citation{pdbx_database_id_DOI pdbx_database_id_PubMed journal_abbrev rcsb_authors}
 polymer_entities{rcsb_id entity_poly{rcsb_entity_polymer_type rcsb_sample_sequence_length}
  rcsb_polymer_entity{pdbx_description}
  rcsb_polymer_entity_container_identifiers{entity_id auth_asym_ids asym_ids uniprot_ids}
  rcsb_entity_source_organism{ncbi_taxonomy_id ncbi_scientific_name taxonomy_lineage{name depth}}
  rcsb_polymer_entity_annotation{type annotation_id name}
  rcsb_polymer_entity_group_membership{aggregation_method similarity_cutoff group_id}}
 assemblies{rcsb_id interfaces{rcsb_interface_info{interface_area polymer_composition} rcsb_interface_partner{interface_partner_identifier{entity_id asym_id}}}}}}'''
lock=threading.Lock()
B=20
def trim(e):
    a=e.get('assemblies') or []
    e['assemblies']=a[:1]
    for pe in e.get('polymer_entities') or []:
        pe['rcsb_polymer_entity_annotation']=[x for x in (pe.get('rcsb_polymer_entity_annotation') or []) if x['type'] in ('Pfam','InterPro')]
        for o in pe.get('rcsb_entity_source_organism') or []:
            o['taxonomy_lineage']=[t['name'] for t in (o.get('taxonomy_lineage') or []) if t.get('depth') is not None and t['depth']<=6]
    return e
def job(i, fo):
    ids=[x.upper() for x in todo[i:i+B]]
    data=None
    for attempt in range(5):
        try:
            r=requests.post('https://data.rcsb.org/graphql',json={'query':Q%json.dumps(ids)},timeout=180)
            r.raise_for_status(); data=r.json()['data']['entries']; break
        except Exception as ex:
            print('retry',i,ex,flush=True); time.sleep(5*(attempt+1))
    if data is None: return
    lines=[json.dumps(trim(e))+'\n' for e in data if e]
    with lock:
        fo.writelines(lines); fo.flush()
    if (i//B)%25==0: print(i+len(ids),len(todo),flush=True)
with open(out,'a') as fo, ThreadPoolExecutor(5) as ex:
    list(ex.map(lambda i: job(i,fo),range(0,len(todo),B)))
print('done')
