"""Fetch per-instance CIS-PEPTIDE and SHEET feature positions for every EM antibody V chain (RCSB GraphQL)."""
import json, urllib.request, sys, time, os, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT=f'{SP}/cache/pdb_meta/cis_sheet.jsonl'
ch=pd.read_csv(f'{SP}/chains.csv.gz',usecols=['pdb','chain','method'])
want=set(zip(ch.pdb.str.upper(),ch.chain))
amap={}
for l in open(f'{SP}/cache/pdb_meta/rcsb_meta.jsonl'):
    e=json.loads(l)
    for pe in e.get('polymer_entities') or []:
        for inst in pe.get('polymer_entity_instances') or []:
            ci=inst['rcsb_polymer_entity_instance_container_identifiers']
            if (e['rcsb_id'],ci['auth_asym_id']) in want: amap[f"{e['rcsb_id']}.{ci['asym_id']}"]=ci['auth_asym_id']
ids=sorted(amap)
Q='''query($ids:[String!]!){ polymer_entity_instances(instance_ids:$ids){ rcsb_id
  rcsb_polymer_entity_instance_container_identifiers{auth_asym_id}
  rcsb_polymer_instance_feature{ type provenance_source feature_positions{beg_seq_id end_seq_id} } } }'''
done=set()
if os.path.exists(OUT): done={json.loads(l)['rcsb_id'] for l in open(OUT)}
todo=[i for i in ids if i not in done]; print(len(ids),len(todo),file=sys.stderr)
with open(OUT,'a') as f:
    for k in range(0,len(todo),200):
        req=urllib.request.Request('https://data.rcsb.org/graphql',data=json.dumps({'query':Q,'variables':{'ids':todo[k:k+200]}}).encode(),headers={'Content-Type':'application/json'})
        for a in range(5):
            try: r=json.load(urllib.request.urlopen(req,timeout=300)); break
            except Exception as ex: print('retry',ex,file=sys.stderr); time.sleep(5*(a+1))
        for x in (r.get('data') or {}).get('polymer_entity_instances') or []:
            feats=[ft for ft in (x.get('rcsb_polymer_instance_feature') or []) if ft['type'] in ('CIS-PEPTIDE','SHEET','UNOBSERVED_RESIDUE_XYZ')]
            f.write(json.dumps({'rcsb_id':x['rcsb_id'],'auth':x['rcsb_polymer_entity_instance_container_identifiers']['auth_asym_id'],'feats':feats})+'\n')
        print(k,file=sys.stderr)
