import json, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
ids=open(sys.argv[1]).read().split(); out=sys.argv[2]; B=int(sys.argv[3])
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id struct{title} struct_keywords{pdbx_keywords text}
rcsb_primary_citation{journal_abbrev year pdbx_database_id_DOI pdbx_database_id_PubMed}
rcsb_accession_info{deposit_date initial_release_date} rcsb_entry_info{resolution_combined molecular_weight}
audit_author{name pdbx_ordinal}
polymer_entities{rcsb_polymer_entity{pdbx_description} entity_poly{type pdbx_seq_one_letter_code_can}
 rcsb_polymer_entity_container_identifiers{uniprot_ids auth_asym_ids} rcsb_entity_source_organism{ncbi_taxonomy_id scientific_name}}}}'''
def fetch(b):
    for t in range(6):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            d=json.load(urllib.request.urlopen(r,timeout=300)); return d['data']['entries']
        except Exception as e:
            print('retry',t,e,file=sys.stderr,flush=True); time.sleep(2**t)
    print('FAILED',b[0],file=sys.stderr); return []
batches=[ids[i:i+B] for i in range(0,len(ids),B)]
with open(out,'w') as f, ThreadPoolExecutor(4) as ex:
    for i,res in enumerate(ex.map(fetch,batches)):
        for e in res: f.write(json.dumps(e)+'\n')
        if i%50==0: print(i,len(batches),file=sys.stderr,flush=True)
print('done',file=sys.stderr)
