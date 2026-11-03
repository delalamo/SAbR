"""Fetch EMDB entry JSON (EBI EMDB REST API) for every EMDB id linked to the EM entries; raw JSON cached gzipped."""
import json, urllib.request, sys, time, os, gzip
from concurrent.futures import ThreadPoolExecutor
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
D=f'{SP}/cache/pdb_meta/emdb'; os.makedirs(D,exist_ok=True)
kind=sys.argv[1] if len(sys.argv)>1 else 'entry'   # entry | analysis
ids=set()
for l in open(f'{SP}/cache/pdb_meta/rcsb_meta.jsonl'):
    e=json.loads(l)
    ci=e.get('rcsb_entry_container_identifiers') or {}
    for x in (ci.get('emdb_ids') or []): ids.add(x)
ids=sorted(ids)
def get(i):
    fn=f'{D}/{kind}_{i}.json.gz'
    if os.path.exists(fn): return i,'cached'
    url=f'https://www.ebi.ac.uk/emdb/api/{kind}/{i}'
    for a in range(4):
        try:
            data=urllib.request.urlopen(url,timeout=120).read()
            with gzip.open(fn,'wb') as f: f.write(data)
            return i,'ok'
        except urllib.error.HTTPError as e:
            if e.code==404: return i,'404'
            time.sleep(3*(a+1))
        except Exception as e:
            time.sleep(3*(a+1))
    return i,'fail'
print(len(ids),'ids',file=sys.stderr)
with ThreadPoolExecutor(4) as ex:
    for n,(i,s) in enumerate(ex.map(get,ids)):
        if s not in ('ok','cached') or n%200==0: print(n,i,s,file=sys.stderr,flush=True)
