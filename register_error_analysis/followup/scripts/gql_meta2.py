import json, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
ids=open(sys.argv[1]).read().split(); out=sys.argv[2]; B=int(sys.argv[3]) if len(sys.argv)>3 else 60
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id struct{title} struct_keywords{pdbx_keywords text}
rcsb_primary_citation{journal_abbrev year pdbx_database_id_DOI pdbx_database_id_PubMed title rcsb_authors}
rcsb_accession_info{deposit_date initial_release_date} audit_author{name pdbx_ordinal}
pdbx_audit_revision_history{ordinal major_revision minor_revision revision_date} pdbx_audit_revision_group{group revision_ordinal}
rcsb_entry_info{molecular_weight deposited_polymer_monomer_count resolution_combined polymer_entity_count_protein}
exptl{method} em_3d_reconstruction{resolution symmetry_type}
polymer_entities{rcsb_id rcsb_polymer_entity{pdbx_description} entity_poly{type rcsb_sample_sequence_length}
 rcsb_polymer_entity_container_identifiers{uniprot_ids auth_asym_ids} rcsb_entity_source_organism{ncbi_taxonomy_id scientific_name}
 polymer_entity_instances{rcsb_polymer_entity_instance_container_identifiers{auth_asym_id} rcsb_polymer_instance_feature_summary{type count coverage}}}}}'''
def fetch(b):
    for t in range(6):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            d=json.load(urllib.request.urlopen(r,timeout=300))
            if 'errors' in d and not d.get('data'): raise Exception(str(d['errors'])[:200])
            return d['data']['entries']
        except Exception as e:
            print('retry',t,e,file=sys.stderr,flush=True); time.sleep(2**t)
    print('FAILED batch',b[0],file=sys.stderr); return []
batches=[ids[i:i+B] for i in range(0,len(ids),B)]
with open(out,'w') as f, ThreadPoolExecutor(4) as ex:
    for i,res in enumerate(ex.map(fetch,batches)):
        for e in res: f.write(json.dumps(e)+'\n')
        if i%20==0: print(i,len(batches),file=sys.stderr,flush=True)
print('done',file=sys.stderr)
