import json, sys, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
ids=open(sys.argv[1]).read().split(); out=sys.argv[2]; B=int(sys.argv[3])
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id polymer_entities{entity_poly{type}
 polymer_entity_instances{rcsb_polymer_entity_instance_container_identifiers{auth_asym_id auth_to_entity_poly_seq_mapping}
 rcsb_polymer_instance_feature{type feature_positions{beg_seq_id values}}}}}}'''
KEEP={'Q_SCORE','RSCC','OWAB','RSRZ'}
def fetch(b):
    for t in range(6):
        try:
            r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':b}}).encode(),{'Content-Type':'application/json'})
            d=json.load(urllib.request.urlopen(r,timeout=300))
            res=[]
            for e in d['data']['entries']:
                for p in e['polymer_entities'] or []:
                    if p['entity_poly']['type']!='polypeptide(L)': continue
                    for inst in p['polymer_entity_instances'] or []:
                        c=inst['rcsb_polymer_entity_instance_container_identifiers']
                        feats={}
                        for f in inst['rcsb_polymer_instance_feature'] or []:
                            if f['type'] in KEEP:
                                v=feats.setdefault(f['type'],{})
                                for fp in f['feature_positions']:
                                    for k,x in enumerate(fp['values'] or []): v[fp['beg_seq_id']+k]=x
                        res.append({'pdb':e['rcsb_id'].lower(),'chain':c['auth_asym_id'],'auth_map':c['auth_to_entity_poly_seq_mapping'],'feats':feats})
            return res
        except Exception as ex:
            print('retry',t,ex,file=sys.stderr,flush=True); time.sleep(2**t)
    print('FAILED',b[0],file=sys.stderr); return []
batches=[ids[i:i+B] for i in range(0,len(ids),B)]
with open(out,'w') as f, ThreadPoolExecutor(4) as ex:
    for i,res in enumerate(ex.map(fetch,batches)):
        for e in res: f.write(json.dumps(e)+'\n')
        if i%50==0: print(i,len(batches),file=sys.stderr,flush=True)
print('done',file=sys.stderr)
