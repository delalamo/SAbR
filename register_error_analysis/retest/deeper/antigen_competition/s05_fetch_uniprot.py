# Fetch UniProt keywords / names / EC / organism for every antigen-candidate accession seen in antibody entries (RCSB GraphQL uniprot)
import json, os, time, requests, threading
from concurrent.futures import ThreadPoolExecutor
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/antigen_competition'
accs=set()
for l in open(f'{C}/gql_entries.jsonl'):
    e=json.loads(l)
    for pe in e.get('polymer_entities') or []:
        accs.update(pe['rcsb_polymer_entity_container_identifiers'].get('uniprot_ids') or [])
out=f'{C}/uniprot_info.json'
res=json.load(open(out)) if os.path.exists(out) else {}
todo=sorted(a for a in accs if a not in res)
print('accessions',len(accs),'todo',len(todo),flush=True)
F='rcsb_id rcsb_uniprot_keyword{id value} rcsb_uniprot_protein{name{value} ec{number} source_organism{scientific_name taxonomy_id} gene{name{value}}}'
lock=threading.Lock()
def job(i):
    chunk=todo[i:i+40]
    q='{'+' '.join(f'u{j}:uniprot(uniprot_id:"{a}"){{{F}}}' for j,a in enumerate(chunk))+'}'
    for attempt in range(4):
        try:
            r=requests.post('https://data.rcsb.org/graphql',json={'query':q},timeout=120); d=r.json().get('data') or {}; break
        except Exception as ex:
            print('retry',ex,flush=True); time.sleep(5); d={}
    with lock:
        for j,a in enumerate(chunk):
            u=d.get(f'u{j}')
            if u:
                p=u.get('rcsb_uniprot_protein') or {}
                res[a]=dict(kw=[k['value'] for k in (u.get('rcsb_uniprot_keyword') or [])],
                    name=((p.get('name') or {}).get('value')), ec=[x['number'] for x in (p.get('ec') or [])],
                    org=((p.get('source_organism') or {}).get('scientific_name')), taxid=((p.get('source_organism') or {}).get('taxonomy_id')),
                    gene=[n['value'] for g in (p.get('gene') or []) for n in (g.get('name') or [])][:3])
            else: res[a]=None
with ThreadPoolExecutor(4) as ex: list(ex.map(job,range(0,len(todo),40)))
json.dump(res,open(out,'w'))
print('done',len(res),sum(v is None for v in res.values()))
