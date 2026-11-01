import json, sys, urllib.request, time
ids=open(sys.argv[1]).read().split()
out=open(sys.argv[2],'w')
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id exptl{method} rcsb_entry_info{resolution_combined}
rcsb_accession_info{deposit_date}
polymer_entities{entity_poly{type rcsb_sample_sequence_length} rcsb_polymer_entity_container_identifiers{auth_asym_ids}}}}'''
for i in range(0,len(ids),300):
    b=ids[i:i+300]
    for t in range(4):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            d=json.load(urllib.request.urlopen(r,timeout=120)); break
        except Exception as e:
            print('retry',e,file=sys.stderr); time.sleep(2**t)
    for e in d['data']['entries']:
        out.write(json.dumps(e)+'\n')
    print(i,file=sys.stderr)
