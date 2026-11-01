import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ids=open(sys.argv[1]).read().split(); out=sys.argv[2]
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id software{classification name} em_software{category name}
pdbx_initial_refinement_model{type source_name accession_code} em_3d_fitting{ref_protocol}
pdbx_audit_revision_group{group revision_ordinal} pdbx_audit_revision_history{ordinal revision_date major_revision}
rcsb_primary_citation{pdbx_database_id_DOI pdbx_database_id_PubMed journal_abbrev}
polymer_entities{rcsb_polymer_entity{pdbx_description} rcsb_polymer_entity_container_identifiers{uniprot_ids auth_asym_ids}}}}'''
def fetch(b):
    for k in range(6):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            return json.load(urllib.request.urlopen(r,timeout=300))['data']['entries']
        except Exception as e: time.sleep(2**k)
    return []
with open(out,'w') as f, ThreadPoolExecutor(4) as ex:
    for res in ex.map(fetch,[ids[i:i+100] for i in range(0,len(ids),100)]):
        for e in res:
            if e: f.write(json.dumps(e)+'\n')
print('done')
