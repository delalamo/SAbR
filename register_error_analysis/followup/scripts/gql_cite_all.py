import json, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
ids=[l.split('\t')[0] for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:] if l.strip()]
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id rcsb_primary_citation{journal_abbrev year pdbx_database_id_DOI pdbx_database_id_PubMed}
rcsb_accession_info{deposit_date initial_release_date} exptl{method}}}'''
def fetch(b):
    for t in range(6):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            return json.load(urllib.request.urlopen(r,timeout=300))['data']['entries']
        except Exception as e:
            print('retry',t,e,file=sys.stderr,flush=True); time.sleep(2**t)
    print('FAILED',b[0],file=sys.stderr); return []
batches=[ids[i:i+500] for i in range(0,len(ids),500)]
with open(sys.argv[1],'w') as f, ThreadPoolExecutor(4) as ex:
    for i,res in enumerate(ex.map(fetch,batches)):
        for e in res:
            if e: f.write(json.dumps(e)+'\n')
        if i%50==0: print(i,len(batches),file=sys.stderr,flush=True)
print('done',file=sys.stderr)
